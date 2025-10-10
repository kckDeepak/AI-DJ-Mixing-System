# track_analysis_engine.py
import json
import os
import librosa
from librosa.feature import rms, spectral_centroid, spectral_contrast
import numpy as np

SONGS_DIR = "./songs"

def analyze_track(title, artist, filename):
    """Analyze local MP3 file to extract audio features using librosa."""
    file_path = os.path.join(SONGS_DIR, filename)
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo))
        
        # Key (simple estimation)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_idx = np.argmax(np.mean(chroma, axis=1))
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = key_names[key_idx]
        
        # Genre (placeholder: assume from filename or improve with ML if needed)
        genre = "Unknown"  # Can use tags or classifier
        
        # Energy (mean RMS, no scaling to allow variation 0-~0.3)
        energy_rms = np.mean(rms(y=y))
        energy = round(float(energy_rms), 2)
        
        # Valence (approx from spectral centroid)
        centroid = np.mean(spectral_centroid(y=y, sr=sr))
        valence = round(float(min(centroid / 5000, 1)), 2)  # Adjusted for more variation
        
        # Danceability (approx from spectral contrast)
        contrast = np.mean(spectral_contrast(y=y, sr=sr))
        danceability = round(float(min(contrast / 40, 1)), 2)  # Adjusted for 0-1 range
        
        return {
            "bpm": bpm,
            "key": key,
            "genre": genre,
            "energy": energy,
            "valence": valence,
            "danceability": danceability
        }
    except Exception as e:
        print(f"Error analyzing '{title}' by '{artist}': {e}")
        return {
            "bpm": 120,
            "key": "C",
            "genre": "Unknown",
            "energy": 0.7,
            "valence": 0.6,
            "danceability": 0.8
        }

def compute_vibe_label(energy, valence, danceability):
    """Compute contextual vibe label based on metadata."""
    if energy > 0.2 and danceability > 0.7:
        return "Peak Energy"
    elif energy < 0.1 and valence > 0.6:
        return "Sunset Chill"
    elif danceability > 0.8 and valence > 0.5:
        return "Dance Floor Filler"
    elif energy > 0.15 and valence < 0.4:
        return "Intense Build"
    return "Balanced Vibe"

def suggest_transition(prev_track_data, current_track_data):
    """Suggest transition type based on previous and current track metadata."""
    if not prev_track_data:
        return "Fade In"
    
    prev_bpm = prev_track_data.get('bpm', 120)
    curr_bpm = current_track_data.get('bpm', 120)
    prev_energy = prev_track_data.get('energy', 0.15)
    curr_energy = current_track_data.get('energy', 0.15)
    bpm_diff = abs(prev_bpm - curr_bpm)
    
    if bpm_diff <= 2:
        return "Seamless Beatmatch"
    elif bpm_diff <= 5:
        return "Pitch Bend Adjustment"
    elif curr_energy > prev_energy + 0.05:  # Lower threshold for variation
        return "Energy Build (EQ Sweep)"
    elif curr_energy < prev_energy - 0.05:
        return "Fade Out Transition"
    return "Crossfade"

def generate_notes(vibe_label, energy, danceability, genre):
    """Generate contextual notes for the track."""
    notes = f"{vibe_label} track. "
    if energy > 0.2:
        notes += "High energy suitable for peak moments. "
    if danceability > 0.8:
        notes += "Excellent for dancing. "
    notes += f"Genre: {genre}."
    return notes

def analyze_tracks_in_setlist(setlist_json):
    """Analyze tracks from setlist using local MP3s, then sort by BPM for better transitions."""
    try:
        setlist_data = json.loads(setlist_json)
        analyzed_setlist = []
        
        for segment in setlist_data["setlist"]:
            time_range = segment["time"]
            tracks = segment["tracks"]
            temp_analyzed = []
            
            # First, analyze all tracks
            for track in tracks:
                title = track["title"]
                artist = track["artist"]
                filename = track["file"]
                
                metadata = analyze_track(title, artist, filename)
                
                vibe_label = compute_vibe_label(
                    metadata["energy"], 
                    metadata["valence"], 
                    metadata["danceability"]
                )
                notes = generate_notes(vibe_label, metadata["energy"], metadata["danceability"], metadata["genre"])
                
                analyzed_track = {
                    "track": title,
                    "artist": artist,
                    "file": filename,
                    "bpm": metadata["bpm"],
                    "key": metadata["key"],
                    "genre": metadata["genre"],
                    "energy": metadata["energy"],
                    "valence": metadata["valence"],
                    "danceability": metadata["danceability"],
                    "notes": notes,
                    "transition": None  # To be set after sorting
                }
                temp_analyzed.append(analyzed_track)
            
            # Sort by BPM for minimal differences
            temp_analyzed.sort(key=lambda x: x['bpm'])
            
            # Now set transitions based on new order
            prev_track_data = None
            analyzed_tracks = []
            for analyzed_track in temp_analyzed:
                metadata = {
                    "bpm": analyzed_track["bpm"],
                    "energy": analyzed_track["energy"]
                }
                transition = suggest_transition(prev_track_data, metadata)
                analyzed_track["transition"] = transition
                analyzed_tracks.append(analyzed_track)
                prev_track_data = metadata
            
            analyzed_setlist.append({
                "time": time_range,
                "analyzed_tracks": analyzed_tracks
            })
        
        output = {"analyzed_setlist": analyzed_setlist}
        with open("analyzed_setlist.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Analyzed setlist saved to 'analyzed_setlist.json'")
        
        return output
    except Exception as e:
        print(f"Error in Track Analysis Engine: {str(e)}")
        raise

if __name__ == "__main__":
    sample_setlist_json = '''
    {
        "setlist": [
            {
                "time": "19:00–20:00",
                "tracks": [
                    {"title": "Tum Hi Ho", "artist": "Arijit Singh", "file": "Arijit Singh - Tum Hi Ho.mp3"},
                    {"title": "No Scrubs", "artist": "TLC", "file": "TLC - No Scrubs.mp3"}
                ]
            },
            {
                "time": "20:00–21:00",
                "tracks": [
                    {"title": "Ye", "artist": "Burna Boy", "file": "Burna Boy - Ye.mp3"},
                    {"title": "Essence", "artist": "Wizkid ft. Tems", "file": "Wizkid ft. Tems - Essence.mp3"}
                ]
            }
        ]
    }
    '''
    analyze_tracks_in_setlist(sample_setlist_json)