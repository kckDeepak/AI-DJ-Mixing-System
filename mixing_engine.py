import json
from datetime import datetime

def is_harmonic_key(from_key, to_key):
    """
    Check if two keys are harmonically compatible (simplified Camelot Wheel logic).
    Args:
        from_key (str): Musical key of the first track (e.g., 'C#m', 'G').
        to_key (str): Musical key of the second track.
    Returns:
        bool: True if keys are compatible, False otherwise.
    """
    # Simplified harmonic mixing: same key or adjacent keys (e.g., C#m -> D#m or B)
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
    return True  # Default to True if key not in map (fallback)

def suggest_transition_type(from_track, to_track):
    """
    Suggest transition type based on BPM difference and metadata.
    Args:
        from_track (dict): Metadata of the first track.
        to_track (dict): Metadata of the second track.
    Returns:
        str: Transition type (e.g., 'Crossfade', 'EQ Sweep', 'Fade Out/Fade In').
    """
    bpm_diff = abs(from_track['bpm'] - to_track['bpm'])
    
    if bpm_diff <= 3:
        return "Crossfade"
    elif 3 < bpm_diff <= 5:
        return "EQ Sweep" if to_track['energy'] > from_track['energy'] else "Echo-Drop"
    else:
        return "Fade Out/Fade In"

def generate_mixing_notes(from_track, to_track, time_segment, transition_type):
    """
    Generate contextual mixing notes for the transition.
    Args:
        from_track (dict): Metadata of the first track.
        to_track (dict): Metadata of the second track.
        time_segment (str): Time range (e.g., '19:00–20:00').
        transition_type (str): Type of transition.
    Returns:
        str: Contextual notes for the DJ.
    """
    notes = f"Smooth handoff into {to_track['notes'].split('.')[0].lower()} section."
    
    # Adjust notes based on time segment and energy
    start_time = time_segment.split('–')[0]
    dt = datetime.strptime(start_time, "%H:%M")
    if dt.hour >= 21:  # Evening peak (e.g., dancing at 9pm)
        if to_track['energy'] > from_track['energy'] + 0.2:
            notes = f"Build tension before drop at {start_time}. {notes}"
    
    # Add harmonic mixing advice
    if not is_harmonic_key(from_track['key'], to_track['key']):
        notes += " Consider key modulation for smoother harmonic transition."
    
    return notes

def generate_mixing_plan(analyzed_setlist_json):
    """
    Generate a mixing plan for track pairs based on metadata.
    Args:
        analyzed_setlist_json (str): JSON string from Track Analysis Engine.
    Returns:
        str: JSON string with mixing plan for each track pair.
    """
    analyzed_data = json.loads(analyzed_setlist_json)
    mixing_plan = []
    
    for segment in analyzed_data["analyzed_setlist"]:
        time_range = segment["time"]
        tracks = segment["analyzed_tracks"]
        
        # Generate transitions between consecutive tracks in the segment
        for i in range(len(tracks) - 1):
            from_track = tracks[i]
            to_track = tracks[i + 1]
            
            transition_type = suggest_transition_type(from_track, to_track)
            notes = generate_mixing_notes(from_track, to_track, time_range, transition_type)
            
            transition = {
                "from_track": from_track["track"],
                "to_track": to_track["track"],
                "transition_point": "first chorus",  # Default as per requirement
                "transition_type": transition_type,
                "comment": notes
            }
            mixing_plan.append(transition)
        
        # If the segment has tracks but no transitions (e.g., single track), add a note
        if len(tracks) == 1:
            mixing_plan.append({
                "from_track": tracks[0]["track"],
                "to_track": None,
                "transition_point": "first chorus",
                "transition_type": "Fade In",
                "comment": f"Start {tracks[0]['notes'].split('.')[0].lower()} section."
            })
    
    return json.dumps({"mixing_plan": mixing_plan}, indent=2)

# Example usage
if __name__ == "__main__":
    # Sample input from Track Analysis Engine
    sample_analyzed_setlist_json = '''
    {
        "analyzed_setlist": [
    {
      "time": "19:00\u201320:00",
      "analyzed_tracks": [
        {
          "track": "Kal Ho Naa Ho",
          "artist": "Sonu Nigam",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Fade In",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Tum Hi Ho",
          "artist": "Arijit Singh",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Kal Ho Naa Ho",
          "artist": "Sonu Nigam",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        }
      ]
    },
    {
      "time": "20:00\u201321:00",
      "analyzed_tracks": [
        {
          "track": "Tum Hi Ho",
          "artist": "Arijit Singh",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Fade In",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Joro",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Ye",
          "artist": "Burna Boy",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "On the Low",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "No Scrubs",
          "artist": "TLC",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "No Scrubs",
          "artist": "TLC",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Tum Hi Ho",
          "artist": "Arijit Singh",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Joro",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "On the Low",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        }
      ]
    },
    {
      "time": "21:00\u201322:00",
      "analyzed_tracks": [
        {
          "track": "Tum Hi Ho",
          "artist": "Arijit Singh",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Fade In",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "On the Low",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "No Scrubs",
          "artist": "TLC",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Joro",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Ye",
          "artist": "Burna Boy",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "On the Low",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Joro",
          "artist": "Wizkid",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Ye",
          "artist": "Burna Boy",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "No Scrubs",
          "artist": "TLC",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        },
        {
          "track": "Tum Hi Ho",
          "artist": "Arijit Singh",
          "bpm": 120,
          "key": "C#m",
          "genre": "Unknown",
          "energy": 0.7,
          "valence": 0.6,
          "danceability": 0.8,
          "transition": "Seamless Beatmatch",
          "notes": "Balanced Vibe track. Genre: Unknown."
        }
      ]
    }
  ]
    }
    '''
    
    result = generate_mixing_plan(sample_analyzed_setlist_json)
    print(result)