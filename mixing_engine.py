# mixing_engine.py
import json
from datetime import datetime, timedelta
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter
import os

SONGS_DIR = "./songs"

def is_harmonic_key(from_key, to_key):
    """Check if two keys are harmonically compatible (simplified Camelot Wheel logic)."""
    key_map = {
        'C': ['C', 'G', 'F', 'Am', 'Em'],
        'C#m': ['C#m', 'G#m', 'F#m', 'E', 'B'],
        'D': ['D', 'A', 'G', 'F#m', 'Bm'],
        'D#m': ['D#m', 'A#m', 'G#m', 'F#', 'C#'],
        'E': ['E', 'B', 'A', 'G#m', 'C#m'],
        'F': ['F', 'C', 'A#m', 'Dm', 'Am'],
        'F#m': ['F#m', 'C#m', 'B', 'A', 'D#m'],
        'G': ['G', 'D', 'C', 'Bm', 'Em'],
        'G#': ['G#', 'D#', 'C#m', 'B', 'F#m'],
        'A': ['A', 'E', 'D', 'C#m', 'F#m'],
        'A#m': ['A#m', 'F', 'D#m', 'C', 'G#m'],
        'B': ['B', 'F#', 'E', 'D#m', 'G#m']
    }
    
    base_key = from_key.replace('m', '') if 'm' in from_key else from_key
    if base_key in key_map:
        return to_key in key_map[base_key]
    return True

def suggest_transition_type(from_track, to_track):
    """Suggest transition type based on BPM difference and metadata."""
    bpm_diff = abs(from_track['bpm'] - to_track['bpm'])
    
    if bpm_diff <= 3:
        return "Crossfade"
    elif 3 < bpm_diff <= 5:
        return "EQ Sweep" if to_track['energy'] > from_track['energy'] else "Echo-Drop"
    else:
        return "Fade Out/Fade In"

def generate_mixing_notes(from_track, to_track, time_segment, transition_type):
    """Generate contextual mixing notes for the transition."""
    notes = f"Smooth handoff into {to_track['notes'].split('.')[0].lower()} section."
    
    start_time = time_segment.split('–')[0]
    dt = datetime.strptime(start_time, "%H:%M")
    if dt.hour >= 21:
        if to_track['energy'] > from_track['energy'] + 0.2:
            notes = f"Build tension before drop at {start_time}. {notes}"
    
    if not is_harmonic_key(from_track['key'], to_track['key']):
        notes += " Consider key modulation for smoother harmonic transition."
    
    return notes

def apply_transition(segment1, segment2, transition_type, duration_ms=5000):
    """Apply DJ transition using pydub."""
    if transition_type == "Crossfade":
        overlap = duration_ms
        crossfade = segment1[-overlap:].fade_out(overlap).overlay(segment2[:overlap].fade_in(overlap))
        return segment1[:-overlap] + crossfade + segment2[overlap:]
    elif transition_type == "EQ Sweep":
        # Simulate EQ sweep: high pass on outgoing, low pass on incoming
        outgoing = high_pass_filter(segment1[-duration_ms:], 200)
        incoming = low_pass_filter(segment2[:duration_ms], 5000)
        return segment1[:-duration_ms] + outgoing.overlay(incoming) + segment2[duration_ms:]
    elif transition_type == "Echo-Drop":
        # Simulate echo-drop
        echo = segment1[-duration_ms:] - 10  # Quieter echo
        return segment1[:-duration_ms] + echo + segment2.fade_in(duration_ms)
    elif transition_type == "Fade Out/Fade In":
        return segment1.fade_out(duration_ms) + segment2.fade_in(duration_ms)
    else:
        return segment1 + segment2  # Default

def generate_mixing_plan_and_mix(analyzed_setlist_json):
    """Generate mixing plan for track pairs and save to JSON file, plus create MP3 mix."""
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
                        "from_track": None if i == 0 else tracks[i-1]["track"],
                        "to_track": track["track"],
                        "start_time": start_str,
                        "transition_point": "first chorus",
                        "transition_type": transition_type,
                        "comment": comment
                    })
                    full_mix += audio.fade_in(2000)
                else:
                    from_track = tracks[i-1]
                    transition_type = suggest_transition_type(from_track, track)
                    comment = generate_mixing_notes(from_track, track, time_range, transition_type)
                    mixing_plan.append({
                        "from_track": from_track["track"],
                        "to_track": track["track"],
                        "start_time": start_str,
                        "transition_point": "first chorus",
                        "transition_type": transition_type,
                        "comment": comment
                    })
                    trans_audio = apply_transition(full_mix[-5000:], audio, transition_type)
                    full_mix = full_mix[:-5000] + trans_audio
                
                duration_sec = len(audio) / 1000
                current_time += timedelta(seconds=duration_sec)
        
        with open("mixing_plan.json", "w") as f:
            json.dump({"mixing_plan": mixing_plan}, f, indent=2)
        print("Mixing plan saved to 'mixing_plan.json'")
        
        full_mix.export("mix.mp3", format="mp3")
        print("Mix exported to 'mix.mp3'")
        
    except Exception as e:
        print(f"Error in Mixing Engine: {str(e)}")
        raise

if __name__ == "__main__":
    sample_analyzed_setlist_json = '''
    {
        "analyzed_setlist": [
            {
                "time": "19:00–20:00",
                "analyzed_tracks": [
                    {
                        "track": "Tum Hi Ho",
                        "artist": "Arijit Singh",
                        "file": "Arijit Singh - Tum Hi Ho.mp3",
                        "bpm": 94,
                        "key": "A",
                        "genre": "bollywood",
                        "energy": 0.45,
                        "valence": 0.32,
                        "danceability": 0.52,
                        "transition": "Fade In",
                        "notes": "Balanced Vibe track. Genre: bollywood."
                    },
                    {
                        "track": "No Scrubs",
                        "artist": "TLC",
                        "file": "TLC - No Scrubs.mp3",
                        "bpm": 93,
                        "key": "G#m",
                        "genre": "r&b",
                        "energy": 0.7,
                        "valence": 0.6,
                        "danceability": 0.8,
                        "transition": "Crossfade",
                        "notes": "Dance Floor Filler track. Genre: r&b."
                    }
                ]
            },
            {
                "time": "20:00–21:00",
                "analyzed_tracks": [
                    {
                        "track": "Ye",
                        "artist": "Burna Boy",
                        "file": "Burna Boy - Ye.mp3",
                        "bpm": 100,
                        "key": "F",
                        "genre": "afrobeats",
                        "energy": 0.85,
                        "valence": 0.7,
                        "danceability": 0.9,
                        "transition": "Fade In",
                        "notes": "Peak Energy track. Genre: afrobeats."
                    },
                    {
                        "track": "Essence",
                        "artist": "Wizkid ft. Tems",
                        "file": "Wizkid ft. Tems - Essence.mp3",
                        "bpm": 104,
                        "key": "C",
                        "genre": "afrobeats",
                        "energy": 0.8,
                        "valence": 0.65,
                        "danceability": 0.85,
                        "transition": "EQ Sweep",
                        "notes": "Dance Floor Filler track. Genre: afrobeats."
                    }
                ]
            }
        ]
    }
    '''
    generate_mixing_plan_and_mix(sample_analyzed_setlist_json)