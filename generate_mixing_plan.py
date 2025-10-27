# generate_mixing_plan.py
import os
import json
import numpy as np
from datetime import datetime, timedelta
from pydub import AudioSegment

SONGS_DIR = "./songs"

# ---------------------------
# Estimated overlap for timing
# ---------------------------
def get_estimated_overlap(transition_type: str, crossfade_early_ms: int = 5500, duration_ms: int = 8000, eq_match_ms: int = 15000):
    t = transition_type.lower()
    if "fade in" in t:
        return 0.0
    elif "eq" in t:
        return eq_match_ms / 1000.0
    else:
        return (duration_ms + crossfade_early_ms) / 1000.0

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
        otac = np.log(float(tempo2) / float(tempo1)) / 60.0
        return float(otac)
    except Exception:
        return 0.0

# ---------------------------
# Transition suggestion
# ---------------------------
def suggest_transition_type(from_track, to_track):
    tempo_diff = abs(float(from_track.get('bpm', 0)) - float(to_track.get('bpm', 0)))
    key_compatible = is_harmonic_key(from_track.get('key_semitone'), to_track.get('key_semitone'))
    has_vocals = from_track.get('has_vocals', False) and to_track.get('has_vocals', False)
    if tempo_diff <= 3 and key_compatible and not has_vocals:
        return "Crossfade"
    if tempo_diff <= 6 and not key_compatible:
        return "EQ Sweep"
    if has_vocals:
        return "Crossfade"
    return "Crossfade"

# ---------------------------
# Mixing plan generator
# ---------------------------
def generate_mixing_plan(analyzed_setlist_json, first_fade_in_ms=5000, crossfade_early_ms=5500, eq_match_ms=15000):
    try:
        analyzed_data = json.loads(analyzed_setlist_json)
        mixing_plan = []
        current_mix_length_sec = 0.0
        last_track = None

        for segment in analyzed_data.get("analyzed_setlist", []):
            tracks = segment.get("analyzed_tracks", [])

            for track in tracks:
                file_path = os.path.join(SONGS_DIR, track["file"])
                if not os.path.exists(file_path):
                    print(f"[generate_mixing_plan] Missing file: {file_path}. Skipping track.")
                    # Use fallback duration to avoid breaking timing
                    duration_sec = 180.0
                    est_overlap_sec = get_estimated_overlap("Crossfade", crossfade_early_ms, 8000, eq_match_ms) if last_track else 0.0
                    start_sec = current_mix_length_sec
                    start_delta = timedelta(seconds=start_sec)
                    start_str = (datetime.min + start_delta).strftime("%H:%M:%S")
                    # Skip appending to plan if no file
                    current_mix_length_sec += duration_sec - est_overlap_sec
                    continue

                audio = AudioSegment.from_file(file_path)
                duration_sec = len(audio) / 1000.0

                if last_track is None:
                    transition_type = "Fade In"
                    comment = f"Start {track.get('notes','').split('.')[0].lower()} section."
                    from_track_title = None
                    otac_val = 0.0
                    transition_point = "downbeat align"
                    est_overlap_sec = 0.0
                else:
                    transition_type = suggest_transition_type(last_track, track)
                    otac_val = compute_otac(last_track, track)
                    comment = f"Transition {last_track.get('title')} -> {track.get('title')}. Suggested '{transition_type}'."
                    from_track_title = last_track.get('title')
                    transition_point = "beat grid match"
                    est_overlap_sec = get_estimated_overlap(transition_type, crossfade_early_ms, 8000, eq_match_ms)

                # Start time is the transition point / take-over time in the mix
                start_sec = current_mix_length_sec
                start_delta = timedelta(seconds=start_sec)
                start_str = (datetime.min + start_delta).strftime("%H:%M:%S")

                mixing_plan.append({
                    "from_track": from_track_title,
                    "to_track": track.get("title"),
                    "start_time": start_str,
                    "transition_point": transition_point,
                    "transition_type": transition_type,
                    "otac": float(otac_val),
                    "comment": comment
                })

                current_mix_length_sec += duration_sec - est_overlap_sec
                last_track = track

        # Save mixing_plan
        with open("mixing_plan.json", "w") as f:
            json.dump({"mixing_plan": mixing_plan}, f, indent=2)
        print("Mixing plan saved to 'mixing_plan.json'")

    except Exception as e:
        print(f"[generate_mixing_plan] Error: {e}")
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
    generate_mixing_plan(sample_analyzed_setlist_json)