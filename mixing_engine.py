# mixing_engine.py
import os
import json
import io
import tempfile
from datetime import datetime, timedelta

import numpy as np
import librosa
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter, normalize

SONGS_DIR = "./songs"

# ---------------------------
# Utility conversions
# ---------------------------
def audio_segment_to_np(segment: AudioSegment):
    """Convert AudioSegment to mono float32 numpy array in range [-1, 1]."""
    samples = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    sr = segment.frame_rate
    # pydub uses signed 16-bit PCM by default
    y = samples.astype(np.float32) / 32768.0
    return y, sr

def np_to_audio_segment(y: np.ndarray, sr: int):
    """Convert mono float32 numpy array in [-1,1] to pydub AudioSegment (mono, 16-bit)."""
    # clip and convert to int16
    y_clipped = np.clip(y, -1.0, 1.0)
    y_int16 = (y_clipped * 32767.0).astype(np.int16)
    return AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

def load_audiofile_to_segment(path):
    """Load audio file to pydub AudioSegment (mp3/wav etc)."""
    return AudioSegment.from_file(path)

# ---------------------------
# Beat / onset helpers
# ---------------------------
def get_onset_envelope(y, sr, hop_length=512):
    """Return onset strength envelope (librosa) normalized."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if onset_env.size == 0:
        return None, hop_length
    onset_env = onset_env / (np.max(onset_env) + 1e-9)
    return onset_env, hop_length

def estimate_tempo_and_beats(y, sr):
    """Estimate tempo (bpm) and beat_times (seconds)."""
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        return float(tempo), beats
    except Exception:
        return 0.0, np.array([])

def find_best_alignment(y1, sr1, y2, sr2, match_duration_sec=15.0):
    """
    Try to find best alignment (lag in seconds) to match onsets between two signals.
    We compute onset envelopes for a window (match_duration_sec) taken from the tails/heads,
    then compute cross-correlation to find the best offset.
    Returns lag_seconds such that shifting y2 by lag aligns it to y1 (positive means y2 delayed).
    """
    # Convert durations to samples
    try:
        # take the last match_duration_sec of y1 and first match_duration_sec of y2
        n1 = min(len(y1), int(match_duration_sec * sr1))
        n2 = min(len(y2), int(match_duration_sec * sr2))
        if n1 < 1024 or n2 < 1024:
            return 0.0  # too short to align robustly

        tail1 = y1[-n1:]
        head2 = y2[:n2]

        # compute onset envelopes (resample to same hop_length for correlation)
        onset1, hop = get_onset_envelope(tail1, sr1)
        onset2, _ = get_onset_envelope(head2, sr2)
        if onset1 is None or onset2 is None:
            return 0.0

        # resample onset arrays to same length scale if different hop lengths or sr
        # simplest: normalize lengths by linear interpolation to comparable lengths
        minlen = min(len(onset1), len(onset2))
        if minlen < 8:
            return 0.0
        onset1_r = librosa.util.fix_length(onset1, size=minlen)
        onset2_r = librosa.util.fix_length(onset2, size=minlen)

        # cross-correlation
        corr = np.correlate(onset1_r - onset1_r.mean(), onset2_r - onset2_r.mean(), mode='full')
        lag_idx = corr.argmax() - (len(onset2_r) - 1)
        # each onset bin corresponds to hop_length/sr seconds; we used hop ~512 by default
        approx_hop_seconds = hop / float(sr1)
        lag_seconds = -lag_idx * approx_hop_seconds  # negative because we aligned head2 to tail1
        # Bound lag to reasonable values (±match_duration_sec)
        lag_seconds = float(np.clip(lag_seconds, -match_duration_sec, match_duration_sec))
        return lag_seconds
    except Exception:
        return 0.0

# ---------------------------
# Harmonic compatibility
# ---------------------------
def is_harmonic_key(from_key_semitone, to_key_semitone):
    compatible_shifts = [0, 1, 11, 7, 5]
    if from_key_semitone is None or to_key_semitone is None:
        return True
    key_diff = abs(int(from_key_semitone) - int(to_key_semitone)) % 12
    return key_diff in compatible_shifts or (12 - key_diff) in compatible_shifts

# ---------------------------
# OTAC (tempo adjust) helper
# ---------------------------
def compute_otac(song1_data, song2_data):
    tempo1, tempo2 = song1_data.get('bpm', 0), song2_data.get('bpm', 0)
    try:
        if tempo1 <= 0 or tempo2 <= 0:
            return 0.0
        # gentle coefficient scaled for seconds window
        otac = np.log(float(tempo2) / float(tempo1)) / 60.0
        return float(otac)
    except Exception:
        return 0.0

# ---------------------------
# Core transition application
# ---------------------------
def apply_transition(segment1: AudioSegment,
                     segment2: AudioSegment,
                     transition_type: str,
                     duration_ms: int = 8000,
                     early_ms: int = 2500,
                     otac: float = 0.0,
                     from_track: dict = None,
                     eq_match_duration_ms: int = 15000):
    """
    Apply different transition types between segment1 (outgoing mix) and segment2 (incoming track).
    - duration_ms: base overlap duration for crossfades (ms)
    - early_ms: how much earlier to start crossfade relative to end of segment1 (ms)
    - eq_match_duration_ms: target beat-match duration for EQ/beat align (ms)
    """
    try:
        # Convert incoming to numpy to optionally time-stretch
        y2, sr2 = audio_segment_to_np(segment2)
        y1, sr1 = audio_segment_to_np(segment1)  # used for matching decisions

        # Time-stretch incoming if otac significant
        if abs(otac) > 0.01:
            try:
                # compute effective rate for the transition window (gentle)
                rate = 1.0 + otac * (max(duration_ms, eq_match_duration_ms) / 1000.0) / 60.0
                # librosa expects float array, may be long - but okay
                y2_stretched = librosa.effects.time_stretch(y2, rate=rate)
            except Exception:
                y2_stretched = y2
        else:
            y2_stretched = y2

        segment2_stretched = np_to_audio_segment(y2_stretched, sr2)

        # ensure same frame rate and channels (mono)
        if segment2_stretched.frame_rate != segment1.frame_rate:
            segment2_stretched = segment2_stretched.set_frame_rate(segment1.frame_rate)
        if segment2_stretched.channels != 1:
            segment2_stretched = segment2_stretched.set_channels(1)

        # compute a safe overlap value
        base_overlap = min(duration_ms, len(segment1), len(segment2_stretched))
        overlap = int(min(len(segment1), len(segment2_stretched), duration_ms + early_ms))

        # Crossfade (early offset applied)
        if transition_type.lower() in ("crossfade", "cross fade"):
            overlap = int(min(len(segment1), len(segment2_stretched), duration_ms + early_ms))
            overlap = max(500, overlap)  # guard
            out_tail = segment1[-overlap:].fade_out(overlap)
            in_head = segment2_stretched[:overlap].fade_in(overlap)
            cross = out_tail.overlay(in_head)
            return segment1[:-overlap] + cross + segment2_stretched[overlap:]

        # EQ Sweep with beat-matching (long overlap ~15s)
        elif transition_type.lower() in ("eq sweep", "eq", "eq_sweep"):
            # want at least eq_match_duration_ms of overlap if possible
            eq_overlap = int(min(eq_match_duration_ms, len(segment1), len(segment2_stretched)))
            if eq_overlap < 2000:
                # fallback to simple crossfade
                eq_overlap = int(min(2000, len(segment1), len(segment2_stretched)))

            # Attempt beat alignment: compute lag in seconds to align incoming head to outgoing tail
            try:
                # convert to np arrays (mono)
                y1_full, sr_full = audio_segment_to_np(segment1)
                y2_full, _ = audio_segment_to_np(segment2_stretched)
                match_sec = eq_overlap / float(sr_full)
                lag_seconds = find_best_alignment(y1_full, sr_full, y2_full, sr_full, match_duration_sec=match_sec)
                # Convert lag_seconds to milliseconds shift for segment2 head: positive lag means delay incoming
                lag_ms = int(lag_seconds * 1000.0)
            except Exception:
                lag_ms = 0

            # Prepare incoming head (possibly shifted)
            incoming_head = segment2_stretched[:eq_overlap]
            if lag_ms > 0:
                incoming_head = AudioSegment.silent(duration=lag_ms) + incoming_head
            elif lag_ms < 0:
                # if negative lag, we advance incoming (cut from head)
                advance = min(-lag_ms, len(incoming_head) - 1)
                if advance > 0:
                    incoming_head = incoming_head[advance:]

            # apply gentle EQ: high-pass to outgoing, low-pass to incoming, then crossfade
            outgoing_tail = segment1[-eq_overlap:]
            outgoing_hp = high_pass_filter(outgoing_tail, cutoff=200)  # removes low rumble
            incoming_lp = low_pass_filter(incoming_head, cutoff=6000)  # smooth highs

            # create gradual crossfade envelope: outgoing fades out, incoming fades in over eq_overlap
            outgoing_faded = outgoing_hp.fade_out(eq_overlap)
            incoming_faded = incoming_lp.fade_in(eq_overlap)

            mixed = outgoing_faded.overlay(incoming_faded)
            # Compose: keep full mix until tail start, then add mixed portion, then rest of incoming (after eq_overlap)
            return segment1[:-eq_overlap] + mixed + segment2_stretched[eq_overlap:]

        # Echo-Drop style (quieter echo + quick fade in)
        elif transition_type.lower() in ("echo-drop", "echo drop"):
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            echo = (segment1[-overlap:] - 10).fade_out(int(overlap * 0.6))
            incoming = segment2_stretched[:overlap].fade_in(int(overlap * 0.6))
            return segment1[:-overlap] + echo.overlay(incoming) + segment2_stretched[overlap:]

        # Fade Out / Fade In with configurable durations (smoother for first track)
        elif transition_type.lower() in ("fade out/fade in", "fade out", "fade in", "fade_in"):
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            fade_out = segment1[-overlap:].fade_out(overlap)
            fade_in = segment2_stretched[:overlap].fade_in(overlap)
            return segment1[:-overlap] + fade_out + fade_in + segment2_stretched[overlap:]

        # Build Drop - simple overlay but keep energy
        elif transition_type.lower() == "build_drop":
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            mixed_overlap = segment1[-overlap:].overlay(segment2_stretched[:overlap])
            return segment1[:-overlap] + mixed_overlap + segment2_stretched[overlap:]

        # Loop / Backspin / Reverb fallbacks similar to original
        elif transition_type.lower() == "loop":
            beat_len_ms = 2000 if from_track is None else int(60.0 / max(0.1, from_track.get('bpm', 120)) * 1000 * 4)
            loop_end = min(beat_len_ms, len(segment1))
            loop_seg = segment1[-loop_end:] + segment1[-loop_end:]
            return loop_seg + segment2_stretched

        elif transition_type.lower() == "backspin":
            rewind_len_ms = int(min(4000, len(segment1)))
            rewind = segment1[-rewind_len_ms:].reverse().fade_out(1000)
            return rewind + segment2_stretched

        elif transition_type.lower() == "reverb":
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            delay_ms = 100
            reverb_out = segment1[-overlap:] + AudioSegment.silent(duration=delay_ms)
            reverb_out = reverb_out.overlay(segment1[-overlap:].shift(delay_ms), gain=-10)
            reverb_in = segment2_stretched[:overlap] + AudioSegment.silent(duration=delay_ms)
            reverb_in = reverb_in.overlay(segment2_stretched[:overlap].shift(delay_ms), gain=-10)
            crossfade = reverb_out.overlay(reverb_in)
            return segment1[:-overlap] + crossfade + segment2_stretched[overlap:]

        else:
            # default safe append
            return segment1 + segment2_stretched

    except Exception as e:
        print(f"[apply_transition] Exception: {e}")
        # fallback to append
        try:
            return segment1 + segment2
        except Exception:
            return AudioSegment.empty()

# ---------------------------
# High-level plan generator and mix exporter
# ---------------------------
def suggest_transition_type(from_track, to_track):
    """
    Simple suggestion logic reused from original with slight adjustments.
    """
    tempo_diff = abs(float(from_track.get('bpm', 0)) - float(to_track.get('bpm', 0)))
    key_compatible = is_harmonic_key(from_track.get('key_semitone'), to_track.get('key_semitone'))
    has_vocals = from_track.get('has_vocals', False) and to_track.get('has_vocals', False)
    if tempo_diff <= 3 and key_compatible and not has_vocals:
        return "Crossfade"
    if tempo_diff <= 6 and not key_compatible:
        return "EQ Sweep"
    if has_vocals:
        return "Fade Out/Fade In"
    return "Crossfade"

def generate_mixing_plan_and_mix(analyzed_setlist_json, first_fade_in_ms=5000, crossfade_early_ms=2500, eq_match_ms=15000):
    """
    Main function: given an analyzed setlist JSON (same format as earlier), produce:
    - mixing_plan.json with transitions and comments
    - mix.mp3 exported with normalized levels
    """
    try:
        analyzed_data = json.loads(analyzed_setlist_json)
        mixing_plan = []
        current_time = datetime.strptime("00:00:00", "%H:%M:%S")
        full_mix = AudioSegment.empty()

        for segment in analyzed_data.get("analyzed_setlist", []):
            time_range = segment.get("time", "")
            tracks = segment.get("analyzed_tracks", [])

            for i, track in enumerate(tracks):
                file_path = os.path.join(SONGS_DIR, track["file"])
                if not os.path.exists(file_path):
                    print(f"[generate_mixing_plan_and_mix] Missing file: {file_path}. Skipping track.")
                    continue

                audio = AudioSegment.from_file(file_path)
                start_str = current_time.strftime("%H:%M:%S")

                # first track of the entire mix or of this segment
                if len(full_mix) == 0:
                    # smoother fade-in (configurable), apply to start of track
                    transition_type = "Fade In"
                    comment = f"Start {track.get('notes','').split('.')[0].lower()} section."
                    mixing_plan.append({
                        "from_track": None,
                        "to_track": track.get("title"),
                        "start_time": start_str,
                        "transition_point": "downbeat align",
                        "transition_type": transition_type,
                        "comment": comment
                    })
                    # Append with long fade-in
                    fade_dur = int(min(first_fade_in_ms, len(audio)))
                    full_mix += audio.fade_in(fade_dur)
                else:
                    from_track = tracks[i-1] if i - 1 >= 0 else None
                    transition_type = suggest_transition_type(from_track if from_track else {}, track)
                    otac = compute_otac(from_track if from_track else {}, track)
                    comment = f"Transition {from_track.get('title') if from_track else 'prev'} -> {track.get('title')}. Suggested '{transition_type}'."
                    mixing_plan.append({
                        "from_track": from_track.get('title') if from_track else None,
                        "to_track": track.get('title'),
                        "start_time": start_str,
                        "transition_point": "beat grid match",
                        "transition_type": transition_type,
                        "otac": float(otac),
                        "comment": comment
                    })

                    # We must choose the chunk of existing mix to use for the transition.
                    # Prefer a chunk equal to max(duration, eq_match_ms) so beat matching can use enough context.
                    desired_overlap_ms = max(eq_match_ms, 8000)
                    available = len(full_mix)
                    overlap_chunk_ms = int(min(available, desired_overlap_ms))
                    if overlap_chunk_ms < 1000:
                        # not enough context: fallback to last 5s
                        overlap_chunk_ms = int(min(5000, available))

                    tail_chunk = full_mix[-overlap_chunk_ms:]
                    # apply transition between tail_chunk and the new track
                    trans_audio = apply_transition(tail_chunk, audio, transition_type,
                                                   duration_ms=8000,
                                                   early_ms=crossfade_early_ms,
                                                   otac=otac,
                                                   from_track=from_track,
                                                   eq_match_duration_ms=eq_match_ms)
                    # replace tail in full_mix
                    full_mix = full_mix[:-overlap_chunk_ms] + trans_audio

                # increment current_time by track duration
                duration_sec = len(audio) / 1000.0
                current_time += timedelta(seconds=duration_sec)

        # Save mixing_plan
        with open("mixing_plan.json", "w") as f:
            json.dump({"mixing_plan": mixing_plan}, f, indent=2)

        # normalize and export mix
        full_mix = normalize(full_mix)
        full_mix.export("mix.mp3", format="mp3")
        print("Mixing plan saved to 'mixing_plan.json'")
        print("Mix exported to 'mix.mp3'")

    except Exception as e:
        print(f"[generate_mixing_plan_and_mix] Error: {e}")
        raise

# ---------------------------
# Optional stem mixing utility (kept simple)
# ---------------------------
def mix_stems(stem_paths1, stem_paths2, output_path):
    """Mix individual stems if provided. 'stem_pathsX' are dicts e.g. {'drums': 'drums1.wav'}"""
    try:
        if 'drums' in stem_paths1 and 'drums' in stem_paths2:
            drums1 = AudioSegment.from_file(stem_paths1['drums'])
            drums2 = AudioSegment.from_file(stem_paths2['drums'])
            mixed_drums = drums1.append(drums2, crossfade=2000)
            mixed_drums.export(output_path, format='wav')
    except Exception as e:
        print(f"[mix_stems] Error: {e}")

# ---------------------------
# Example run for quick debugging
# ---------------------------
if __name__ == "__main__":
    sample_analyzed_setlist_json = '''
    {
        "analyzed_setlist": [
            {
                "time": "19:00–20:00",
                "analyzed_tracks": [
                    {
                        "title": "Tum Hi Ho",
                        "artist": "Arijit Singh",
                        "file": "Arijit Singh - Tum Hi Ho.mp3",
                        "bpm": 94,
                        "key_semitone": 9,
                        "scale": "major",
                        "genre": "bollywood",
                        "energy": 0.45,
                        "valence": 0.32,
                        "danceability": 0.52,
                        "has_vocals": true,
                        "segments": [{"label": "L"}],
                        "chroma_matrix": null,
                        "transition": "Fade In",
                        "notes": "Balanced Vibe track. Genre: bollywood."
                    },
                    {
                        "title": "No Scrubs",
                        "artist": "TLC",
                        "file": "TLC - No Scrubs.mp3",
                        "bpm": 93,
                        "key_semitone": 8,
                        "scale": "minor",
                        "genre": "r&b",
                        "energy": 0.7,
                        "valence": 0.6,
                        "danceability": 0.8,
                        "has_vocals": true,
                        "segments": [{"label": "H"}],
                        "chroma_matrix": null,
                        "transition": "Crossfade",
                        "notes": "Dance Floor Filler track. Genre: r&b."
                    }
                ]
            }
        ]
    }
    '''
    generate_mixing_plan_and_mix(sample_analyzed_setlist_json)
