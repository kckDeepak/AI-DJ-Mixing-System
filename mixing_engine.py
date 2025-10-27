# generate_mix.py
import os
import json
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
    y = samples.astype(np.float32) / 32768.0
    return y, sr

def np_to_audio_segment(y: np.ndarray, sr: int):
    """Convert mono float32 numpy array in [-1,1] to pydub AudioSegment (mono, 16-bit)."""
    y_clipped = np.clip(y, -1.0, 1.0)
    y_int16 = (y_clipped * 32767.0).astype(np.int16)
    return AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

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

def find_best_alignment(y1, sr1, y2, sr2, match_duration_sec=15.0):
    try:
        n1 = min(len(y1), int(match_duration_sec * sr1))
        n2 = min(len(y2), int(match_duration_sec * sr2))
        if n1 < 1024 or n2 < 1024:
            return 0.0
        tail1 = y1[-n1:]
        head2 = y2[:n2]
        onset1, hop = get_onset_envelope(tail1, sr1)
        onset2, _ = get_onset_envelope(head2, sr2)
        if onset1 is None or onset2 is None:
            return 0.0
        minlen = min(len(onset1), len(onset2))
        if minlen < 8:
            return 0.0
        onset1_r = librosa.util.fix_length(onset1, size=minlen)
        onset2_r = librosa.util.fix_length(onset2, size=minlen)
        corr = np.correlate(onset1_r - onset1_r.mean(), onset2_r - onset2_r.mean(), mode='full')
        lag_idx = corr.argmax() - (len(onset2_r) - 1)
        approx_hop_seconds = hop / float(sr1)
        lag_seconds = -lag_idx * approx_hop_seconds
        lag_seconds = float(np.clip(lag_seconds, -match_duration_sec, match_duration_sec))
        return lag_seconds
    except Exception:
        return 0.0

# ---------------------------
# Core transition application
# ---------------------------
def apply_transition(segment1: AudioSegment,
                     segment2: AudioSegment,
                     transition_type: str,
                     duration_ms: int = 8000,
                     early_ms: int = 5500,
                     otac: float = 0.0,
                     eq_match_duration_ms: int = 15000):
    try:
        y2, sr2 = audio_segment_to_np(segment2)
        y1, sr1 = audio_segment_to_np(segment1)
        if abs(otac) > 0.01:
            rate = 1.0 + otac * (max(duration_ms, eq_match_duration_ms) / 1000.0) / 60.0
            y2_stretched = librosa.effects.time_stretch(y2, rate=rate)
        else:
            y2_stretched = y2
        segment2_stretched = np_to_audio_segment(y2_stretched, sr2)
        if segment2_stretched.frame_rate != segment1.frame_rate:
            segment2_stretched = segment2_stretched.set_frame_rate(segment1.frame_rate)
        if segment2_stretched.channels != 1:
            segment2_stretched = segment2_stretched.set_channels(1)
        overlap = int(min(len(segment1), len(segment2_stretched), duration_ms + early_ms))
        overlap = max(500, overlap)

        if transition_type.lower() in ("crossfade", "cross fade"):
            out_tail = segment1[-overlap:].fade_out(overlap)
            in_head = segment2_stretched[:overlap].fade_in(overlap)
            cross = out_tail.overlay(in_head)
            return segment1[:-overlap] + cross + segment2_stretched[overlap:]

        elif transition_type.lower() in ("eq sweep", "eq", "eq_sweep"):
            eq_overlap = int(min(eq_match_duration_ms, len(segment1), len(segment2_stretched)))
            eq_overlap = max(2000, eq_overlap)
            y1_full, sr_full = audio_segment_to_np(segment1)
            y2_full, _ = audio_segment_to_np(segment2_stretched)
            match_sec = eq_overlap / float(sr_full)
            lag_seconds = find_best_alignment(y1_full, sr_full, y2_full, sr_full, match_duration_sec=match_sec)
            lag_ms = int(lag_seconds * 1000.0)
            # Support both delay and advance without skipping content
            outgoing_tail = segment1[-eq_overlap:]
            outgoing_hp = high_pass_filter(outgoing_tail, cutoff=200)
            outgoing_faded = outgoing_hp.fade_out(eq_overlap)

            head_for_filter = segment2_stretched[:eq_overlap]
            incoming_lp = low_pass_filter(head_for_filter, cutoff=6000)
            incoming_faded = incoming_lp.fade_in(eq_overlap)

            if lag_ms > 0:
                # Delay: prepend silence to incoming, but fade only the content
                silence = AudioSegment.silent(duration=lag_ms)
                incoming_head_faded = silence + incoming_faded
                mixed = outgoing_faded.overlay(incoming_head_faded)
                tail = segment2_stretched[eq_overlap:]
            elif lag_ms < 0:
                # Advance: prepend the skipped head, overlay the rest on outgoing
                advance_ms = -lag_ms
                pre_head = low_pass_filter(segment2_stretched[:advance_ms], cutoff=6000)
                # No fade on pre (full volume start)
                overlay_head = low_pass_filter(segment2_stretched[advance_ms : advance_ms + eq_overlap], cutoff=6000)
                overlay_faded = overlay_head.fade_in(eq_overlap)
                mixed = outgoing_faded.overlay(overlay_faded, position=0)
                incoming_head_faded = pre_head + mixed  # Prepend pre to the overlay
                tail = segment2_stretched[advance_ms + eq_overlap:]
            else:
                # No lag
                incoming_head_faded = incoming_faded
                mixed = outgoing_faded.overlay(incoming_head_faded)
                tail = segment2_stretched[eq_overlap:]

            return segment1[:-eq_overlap] + mixed + tail

        elif transition_type.lower() in ("echo-drop", "echo drop"):
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            echo = (segment1[-overlap:] - 10).fade_out(int(overlap * 0.6))
            incoming = segment2_stretched[:overlap].fade_in(int(overlap * 0.6))
            return segment1[:-overlap] + echo.overlay(incoming) + segment2_stretched[overlap:]

        elif transition_type.lower() in ("fade out/fade in", "fade out", "fade in", "fade_in"):
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            fade_out = segment1[-overlap:].fade_out(overlap)
            fade_in = segment2_stretched[:overlap].fade_in(overlap)
            return segment1[:-overlap] + fade_out + fade_in + segment2_stretched[overlap:]

        elif transition_type.lower() == "build_drop":
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            mixed_overlap = segment1[-overlap:].overlay(segment2_stretched[:overlap])
            return segment1[:-overlap] + mixed_overlap + segment2_stretched[overlap:]

        elif transition_type.lower() == "loop":
            beat_len_ms = 2000
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
            return segment1 + segment2_stretched

    except Exception as e:
        print(f"[apply_transition] Exception: {e}")
        return segment1 + segment2

# ---------------------------
# Mix generator
# ---------------------------
def generate_mix(analyzed_setlist_json, mixing_plan_json, first_fade_in_ms=5000, crossfade_early_ms=5500, eq_match_ms=15000):
    try:
        analyzed_data = json.loads(analyzed_setlist_json)
        mixing_plan = json.load(open(mixing_plan_json, 'r')).get("mixing_plan", [])
        full_mix = AudioSegment.empty()
        track_index = 0

        for segment in analyzed_data.get("analyzed_setlist", []):
            tracks = segment.get("analyzed_tracks", [])

            for track in tracks:
                file_path = os.path.join(SONGS_DIR, track["file"])
                if not os.path.exists(file_path):
                    print(f"[generate_mix] Missing file: {file_path}. Skipping track.")
                    track_index += 1
                    continue

                audio = AudioSegment.from_file(file_path)
                if track_index >= len(mixing_plan):
                    print(f"[generate_mix] Mixing plan too short, appending track {track['title']}.")
                    full_mix += audio
                    track_index += 1
                    continue

                plan_entry = mixing_plan[track_index]
                transition_type = plan_entry.get("transition_type", "Crossfade")
                otac = plan_entry.get("otac", 0.0)

                if len(full_mix) == 0:
                    fade_dur = int(min(first_fade_in_ms, len(audio)))
                    full_mix += audio.fade_in(fade_dur)
                else:
                    desired_overlap_ms = max(eq_match_ms if "eq" in transition_type.lower() else 8000, 8000)
                    available = len(full_mix)
                    overlap_chunk_ms = int(min(available, desired_overlap_ms))
                    if overlap_chunk_ms < 1000:
                        overlap_chunk_ms = int(min(5000, available))
                    tail_chunk = full_mix[-overlap_chunk_ms:]
                    trans_audio = apply_transition(tail_chunk, audio, transition_type,
                                                   duration_ms=8000,
                                                   early_ms=crossfade_early_ms,
                                                   otac=otac,
                                                   eq_match_duration_ms=eq_match_ms)
                    full_mix = full_mix[:-overlap_chunk_ms] + trans_audio

                track_index += 1

        # Normalize and export mix
        full_mix = normalize(full_mix)
        full_mix.export("mix.mp3", format="mp3")
        print("Mix exported to 'mix.mp3'")

    except Exception as e:
        print(f"[generate_mix] Error: {e}")
        raise

# ---------------------------
# Example run
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
    # Assumes mixing_plan.json exists; generate it first using generate_mixing_plan.py
    generate_mix(sample_analyzed_setlist_json, "mixing_plan.json")