# mixing_engine.py
import json
from datetime import datetime, timedelta
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter, normalize
import os
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import scipy.signal as signal
import tempfile
import io

SONGS_DIR = "./songs"

def audio_segment_to_np(segment):
    """Convert AudioSegment to numpy array."""
    samples = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)  # Convert to mono
    return samples.astype(np.float32) / 32768.0, segment.frame_rate

def np_to_audio_segment(y, sr):
    """Convert numpy array to AudioSegment."""
    y_int = (y * 32767).astype(np.int16)
    return AudioSegment(
        y_int.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

def is_harmonic_key(from_key_semitone, to_key_semitone):
    """Check harmonic compatibility using circle of fifths."""
    compatible_shifts = [0, 1, 11, 7, 5]  # Same, adjacent, fifths
    key_diff = abs(from_key_semitone - to_key_semitone) % 12
    return key_diff in compatible_shifts or (12 - key_diff) in compatible_shifts

def compute_chroma_similarity(chroma1, chroma2):
    """Mean cosine similarity on aligned chroma vectors. Returns 0 if invalid or mismatched."""
    # Check for None
    if chroma1 is None or chroma2 is None:
        return 0.0

    # Convert to numpy arrays
    chroma1 = np.array(chroma1)
    chroma2 = np.array(chroma2)

    # Skip if empty
    if chroma1.size == 0 or chroma2.size == 0:
        return 0.0

    # Ensure both are 2D
    if chroma1.ndim < 2 or chroma2.ndim < 2:
        return 0.0

    # Align lengths
    min_len = min(chroma1.shape[0], chroma2.shape[0])
    chroma1 = chroma1[:min_len]
    chroma2 = chroma2[:min_len]

    sims = []
    for i in range(min_len):
        # Skip frames with mismatched shape
        if chroma1[i].shape != chroma2[i].shape:
            continue
        # Compute cosine similarity only if both vectors are non-zero
        if np.linalg.norm(chroma1[i]) > 0 and np.linalg.norm(chroma2[i]) > 0:
            sims.append(1 - cosine(chroma1[i], chroma2[i]))

    return float(np.mean(sims)) if sims else 0.0

def compute_otac(song1_data, song2_data):
    """Optimal Tempo Adjustment Coefficient: Log ramp for gradual tempo change."""
    tempo1, tempo2 = song1_data['bpm'], song2_data['bpm']
    if tempo1 == 0 or tempo2 == 0:
        return 0.0
    otac = np.log(tempo2 / tempo1) / 60  # Ramp over 1 min
    return otac

def auto_transition_type(song1_data, song2_data):
    """Automatic transition selection based on features."""
    tempo_diff = abs(song1_data['bpm'] - song2_data['bpm'])
    energy_change = False
    if song1_data.get('segments') and song2_data.get('segments') and song1_data['segments'] and song2_data['segments']:
        energy_change = song1_data['segments'][-1].get('label') != song2_data['segments'][0].get('label')
    has_vocals = song1_data.get('has_vocals', False) and song2_data.get('has_vocals', False)
    key_compatible = is_harmonic_key(song1_data['key_semitone'], song2_data['key_semitone'])
    
    if tempo_diff < 5 and key_compatible and not has_vocals:
        return 'crossfade'
    elif energy_change:
        return 'build_drop'
    elif has_vocals:
        return 'eq'
    elif not key_compatible:
        return 'reverb'
    else:
        return 'quick_cut'

def suggest_transition_type(from_track, to_track):
    """Suggest transition type based on BPM difference and metadata (updated with advanced logic)."""
    transition_type = auto_transition_type(from_track, to_track)
    bpm_diff = abs(from_track['bpm'] - to_track['bpm'])
    
    if bpm_diff <= 3:
        return "Crossfade"
    elif 3 < bpm_diff <= 5:
        return "EQ Sweep" if to_track['energy'] > from_track['energy'] else "Echo-Drop"
    else:
        return "Fade Out/Fade In"

def generate_mixing_notes(from_track, to_track, time_segment, transition_type):
    """Generate contextual mixing notes for the transition (enhanced)."""
    notes = f"Smooth handoff into {to_track['notes'].split('.')[0].lower()} section."
    
    start_time = time_segment.split('–')[0]
    try:
        dt = datetime.strptime(start_time, "%H:%M")
        if dt.hour >= 21:
            if to_track['energy'] > from_track['energy'] + 0.2:
                notes = f"Build tension before drop at {start_time}. {notes}"
    except ValueError:
        pass  # Invalid time format, skip
    
    if not is_harmonic_key(from_track['key_semitone'], to_track['key_semitone']):
        notes += " Consider key modulation for smoother harmonic transition."
    
    similarity = compute_chroma_similarity(from_track.get('chroma_matrix'), to_track.get('chroma_matrix'))
    if similarity > 0.7:
        notes += f" High aural similarity ({similarity:.2f}) for seamless mashup."
    
    return notes

def apply_transition(segment1, segment2, transition_type, duration_ms=5000, otac=0, from_track=None):
    """Apply advanced DJ transitions using pydub and librosa for stretching."""
    try:
        # Convert segment2 to np for stretching
        y2, sr2 = audio_segment_to_np(segment2)
        
        # Apply time stretch if needed
        if abs(otac) > 0.01:  # Only if significant change
            rate = 1 + otac * (duration_ms / 1000) / 60  # Gradual adjust during transition
            y2_stretched = librosa.effects.time_stretch(y2, rate=rate)
        else:
            y2_stretched = y2
        
        # Convert back to AudioSegment
        segment2_stretched = np_to_audio_segment(y2_stretched, sr2)
        
        # Ensure same frame rate as segment1
        if segment2_stretched.frame_rate != segment1.frame_rate:
            segment2_stretched = segment2_stretched.set_frame_rate(segment1.frame_rate)
        
        if transition_type == "Crossfade":
            overlap = min(duration_ms, min(len(segment1), len(segment2_stretched)))
            crossfade_out = segment1[-overlap:].fade_out(overlap)
            crossfade_in = segment2_stretched[:overlap].fade_in(overlap)
            crossfade = crossfade_out.overlay(crossfade_in)
            return segment1[:-overlap] + crossfade + segment2_stretched[overlap:]
        elif transition_type == "EQ Sweep":
            overlap = min(duration_ms, min(len(segment1), len(segment2_stretched)))
            outgoing = high_pass_filter(segment1[-overlap:], 200)
            incoming = low_pass_filter(segment2_stretched[:overlap], 5000)
            crossfade = outgoing.overlay(incoming)
            return segment1[:-overlap] + crossfade + segment2_stretched[overlap:]
        elif transition_type == "Echo-Drop":
            overlap = min(duration_ms, min(len(segment1), len(segment2_stretched)))
            echo = segment1[-overlap:] - 10  # Quieter echo
            echo_fade = echo.fade_out(1000)
            incoming = segment2_stretched[:overlap].fade_in(1000)
            return segment1[:-overlap] + echo_fade.overlay(incoming) + segment2_stretched[overlap:]
        elif transition_type == "Fade Out/Fade In":
            overlap = min(duration_ms, min(len(segment1), len(segment2_stretched)))
            fade_out = segment1[-overlap:].fade_out(overlap)
            fade_in = segment2_stretched[:overlap].fade_in(overlap)
            return segment1[:-overlap] + fade_out + fade_in + segment2_stretched[overlap:]
        elif transition_type == "Quick Cut":
            cut_point = min(len(segment1) // 2, len(segment2_stretched) // 2)
            return segment1[:cut_point] + segment2_stretched[cut_point:]
        elif transition_type == "Build Drop":
            # Simple overlay for build
            overlap = min(duration_ms, min(len(segment1), len(segment2_stretched)))
            mixed_overlap = segment1[-overlap:].overlay(segment2_stretched[:overlap])
            return segment1[:-overlap] + mixed_overlap + segment2_stretched[overlap:]
        elif transition_type == "Loop":
            # Simple loop of last beat
            beat_len_ms = 2000 if from_track is None else int(60 / from_track['bpm'] * 1000 * 4)
            loop_end = min(beat_len_ms, len(segment1))
            loop_seg = segment1[-loop_end:] + segment1[-loop_end:]
            return loop_seg + segment2_stretched
        elif transition_type == "Backspin":
            rewind_len_ms = 4000
            rewind_end = min(rewind_len_ms, len(segment1))
            rewind = segment1[-rewind_end:].reverse().fade_out(1000)
            return rewind + segment2_stretched
        elif transition_type == "Reverb":
            # Simple reverb on overlap
            overlap = min(duration_ms, min(len(segment1), len(segment2_stretched)))
            delay_ms = 100
            reverb_out = segment1[-overlap:]
            reverb_out = reverb_out + AudioSegment.silent(duration=delay_ms)
            reverb_out = reverb_out.overlay(segment1[-overlap:].shift(delay_ms), gain=-10)
            reverb_in = segment2_stretched[:overlap]
            reverb_in = reverb_in + AudioSegment.silent(duration=delay_ms)
            reverb_in = reverb_in.overlay(segment2_stretched[:overlap].shift(delay_ms), gain=-10)
            crossfade = reverb_out.overlay(reverb_in)
            return segment1[:-overlap] + crossfade + segment2_stretched[overlap:]
        else:
            # Default crossfade
            return segment1 + segment2_stretched
    except Exception as e:
        print(f"Error in apply_transition: {e}")
        # Fallback to simple append
        return segment1 + segment2

def generate_mixing_plan_and_mix(analyzed_setlist_json):
    """Generate advanced mixing plan and create MP3 mix."""
    try:
        analyzed_data = json.loads(analyzed_setlist_json)
        mixing_plan = []
        current_time = datetime.strptime("00:00", "%H:%M")
        full_mix = AudioSegment.empty()
        
        for segment in analyzed_data["analyzed_setlist"]:
            time_range = segment["time"]
            tracks = segment["analyzed_tracks"]
            
            for i in range(len(tracks)):
                track = tracks[i]
                file_path = os.path.join(SONGS_DIR, track["file"])
                audio = AudioSegment.from_mp3(file_path)
                
                start_str = current_time.strftime("%H:%M:%S")
                
                if i == 0:
                    transition_type = "Fade In"
                    comment = f"Start {track['notes'].split('.')[0].lower()} section."
                    mixing_plan.append({
                        "from_track": None if i == 0 else tracks[i-1]["title"],
                        "to_track": track["title"],
                        "start_time": start_str,
                        "transition_point": "downbeat align",
                        "transition_type": transition_type,
                        "comment": comment
                    })
                    full_mix += audio.fade_in(2000)
                else:
                    from_track = tracks[i-1]
                    transition_type = suggest_transition_type(from_track, track)
                    otac = compute_otac(from_track, track)
                    comment = generate_mixing_notes(from_track, track, time_range, transition_type)
                    mixing_plan.append({
                        "from_track": from_track["title"],
                        "to_track": track["title"],
                        "start_time": start_str,
                        "transition_point": "beat grid match",
                        "transition_type": transition_type,
                        "otac": float(otac),
                        "comment": comment
                    })
                    trans_audio = apply_transition(full_mix[-5000:], audio, transition_type, otac=otac, from_track=from_track)
                    full_mix = full_mix[:-5000] + trans_audio
                
                duration_sec = len(audio) / 1000
                current_time += timedelta(seconds=duration_sec)
        
        with open("mixing_plan.json", "w") as f:
            json.dump({"mixing_plan": mixing_plan}, f, indent=2)
        print("Mixing plan saved to 'mixing_plan.json'")
        
        full_mix = normalize(full_mix)
        full_mix.export("mix.mp3", format="mp3")
        print("Mix exported to 'mix.mp3'")
        
    except Exception as e:
        print(f"Error in Mixing Engine: {str(e)}")
        raise

# Stem mixing placeholder
def mix_stems(stem_paths1, stem_paths2, output_path):
    """Mix individual stems if provided."""
    if 'drums' in stem_paths1 and 'drums' in stem_paths2:
        drums1 = AudioSegment.from_file(stem_paths1['drums'])
        drums2 = AudioSegment.from_file(stem_paths2['drums'])
        mixed_drums = drums1.append(drums2, crossfade=2000)
        mixed_drums.export(output_path, format='wav')

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
                        "has_vocals": True,
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
                        "has_vocals": True,
                        "segments": [{"label": "H"}],
                        "chroma_matrix": null,
                        "transition": "Crossfade",
                        "notes": "Dance Floor Filler track. Genre: r&b."
                    }
                ]
            },
            {
                "time": "20:00–21:00",
                "analyzed_tracks": [
                    {
                        "title": "Ye",
                        "artist": "Burna Boy",
                        "file": "Burna Boy - Ye.mp3",
                        "bpm": 100,
                        "key_semitone": 5,
                        "scale": "major",
                        "genre": "afrobeats",
                        "energy": 0.85,
                        "valence": 0.7,
                        "danceability": 0.9,
                        "has_vocals": True,
                        "segments": [{"label": "H"}],
                        "chroma_matrix": null,
                        "transition": "Fade In",
                        "notes": "Peak Energy track. Genre: afrobeats."
                    },
                    {
                        "title": "Essence",
                        "artist": "Wizkid ft. Tems",
                        "file": "Wizkid ft. Tems - Essence.mp3",
                        "bpm": 104,
                        "key_semitone": 0,
                        "scale": "major",
                        "genre": "afrobeats",
                        "energy": 0.8,
                        "valence": 0.65,
                        "danceability": 0.85,
                        "has_vocals": True,
                        "segments": [{"label": "H"}],
                        "chroma_matrix": null,
                        "transition": "EQ Sweep",
                        "notes": "Dance Floor Filler track. Genre: afrobeats."
                    }
                ]
            }
        ]
    }
    '''
    generate_mixing_plan_and_mix(sample_analyzed_setlist_json)