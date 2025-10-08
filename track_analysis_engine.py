import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# --- Spotify API Configuration ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# --- Sanity Check for Credentials ---
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise ValueError("FATAL ERROR: SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET is missing or empty in the .env file. Please verify your .env file path and contents.")

# --------------------------------------------------------------------------
# Application Authentication (Client Credentials Flow)
# This is a robust, non-expiring token manager for public endpoints.
# --------------------------------------------------------------------------
try:
    ccm = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    
    # Explicitly get the token here and print confirmation
    initial_token = ccm.get_access_token(as_dict=False)
    if not initial_token:
        raise Exception("Client Credentials Manager failed to retrieve an access token (token was None).")
        
    print(f"--- SUCCESS: Initial application token retrieved (starts with: {initial_token[:10]}...) ---")

    # Create the Spotipy instance using the verified manager
    sp_app = spotipy.Spotify(client_credentials_manager=ccm)
    
except Exception as e:
    print("\n\n--- FATAL SPOTIFY AUTHENTICATION ERROR (CCM FAILED) ---")
    print("The application failed to initialize Client Credentials. Check ID/Secret validity or network.")
    print(f"Underlying error: {e}")
    raise ConnectionError("Spotify Client Credentials initialization failed. Cannot proceed with API calls.")


def search_track(title, artist):
    """Search for a track on Spotify and return its audio features."""
    try:
        # 1. Search (uses sp_app)
        results = sp_app.search(q=f'track:{title} artist:{artist}', type='track', limit=1)
        
        if not results['tracks']['items']:
            raise ValueError(f"Track '{title}' by '{artist}' not found on Spotify.")
        
        track = results['tracks']['items'][0]
        track_id = track['id']

        # 2. Audio Features (uses sp_app)
        audio_features = sp_app.audio_features(tracks=[track_id])[0]
        
        if not audio_features:
            raise ValueError(f"Audio features not available for '{title}'.")
        
        # 3. Artist Info (uses sp_app)
        artist_id = track['artists'][0]['id']
        artist_info = sp_app.artist(artist_id)
        genres = artist_info.get('genres', [])
        primary_genre = genres[0] if genres else "Unknown"
        
        # Map key and mode to readable name
        key_num = audio_features['key']
        mode = audio_features['mode']
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_name = key_names[key_num] if 0 <= key_num < len(key_names) else "Unknown"
        if mode == 0 and key_name != "Unknown": # mode 0 is minor
            key_name += "m"
        
        return {
            "bpm": round(audio_features['tempo']),
            "key": key_name,
            "genre": primary_genre,
            "energy": round(audio_features['energy'], 2),
            "valence": round(audio_features['valence'], 2),
            "danceability": round(audio_features['danceability'], 2)
        }
    except Exception as e:
        # Fallback for API failures or tracks not found
        # We catch the 403 error here and print a cleaner message
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
             print(f"Error fetching data for '{title}' by '{artist}': API call rejected (403 Forbidden).")
        else:
             print(f"Error fetching data for '{title}' by '{artist}': {error_msg}")
             
        return {
            "bpm": 120,
            "key": "C#m",
            "genre": "Unknown",
            "energy": 0.7,
            "valence": 0.6,
            "danceability": 0.8
        }

def compute_vibe_label(energy, valence, danceability):
    """Compute contextual vibe label based on metadata."""
    if energy > 0.8 and danceability > 0.7:
        return "Peak Energy"
    elif energy < 0.4 and valence > 0.6:
        return "Sunset Chill"
    elif danceability > 0.8 and valence > 0.5:
        return "Dance Floor Filler"
    elif energy > 0.6 and valence < 0.4:
        return "Intense Build"
    return "Balanced Vibe"

def suggest_transition(prev_track_data, current_track_data):
    """Suggest transition type based on previous and current track metadata."""
    if not prev_track_data:
        return "Fade In"
    
    prev_bpm = prev_track_data.get('bpm', 120)
    curr_bpm = current_track_data.get('bpm', 120)
    prev_energy = prev_track_data.get('energy', 0.5)
    curr_energy = current_track_data.get('energy', 0.5)
    bpm_diff = abs(prev_bpm - curr_bpm)
    
    if bpm_diff <= 2:
        return "Seamless Beatmatch"
    elif bpm_diff <= 5:
        return "Pitch Bend Adjustment"
    elif curr_energy > prev_energy + 0.2:
        return "Energy Build (EQ Sweep)"
    elif curr_energy < prev_track_data.get('energy', 0.5) - 0.2:
        return "Fade Out Transition"
    return "Crossfade"

def generate_notes(vibe_label, energy, danceability, genre):
    """Generate contextual notes for the track."""
    notes = f"{vibe_label} track. "
    if energy > 0.7:
        notes += "High energy suitable for peak moments. "
    if danceability > 0.8:
        notes += "Excellent for dancing. "
    notes += f"Genre: {genre}."
    return notes

def analyze_tracks_in_setlist(setlist_json):
    """Analyze tracks from Engine 1 setlist and save to JSON file."""
    try:
        print("Running Track Identification Engine...")
        setlist_data = json.loads(setlist_json)
        with open("setlist.json", "w") as f:
            json.dump(setlist_data, f, indent=2)
        print("Setlist saved to 'setlist.json'")
        
        print("Running Track Analysis Engine...")
        analyzed_setlist = []
        
        for segment in setlist_data["setlist"]:
            time_range = segment["time"]
            tracks = segment["tracks"]
            analyzed_tracks = []
            prev_track_data = None
            
            for track in tracks:
                title = track["title"]
                artist = track["artist"]
                
                # The core track analysis function now uses the stable sp_app
                metadata = search_track(title, artist)
                
                vibe_label = compute_vibe_label(
                    metadata["energy"], 
                    metadata["valence"], 
                    metadata["danceability"]
                )
                transition = suggest_transition(prev_track_data, metadata)
                notes = generate_notes(vibe_label, metadata["energy"], metadata["danceability"], metadata["genre"])
                
                analyzed_track = {
                    "track": title,
                    "artist": artist,
                    "bpm": metadata["bpm"],
                    "key": metadata["key"],
                    "genre": metadata["genre"],
                    "energy": metadata["energy"],
                    "valence": metadata["valence"],
                    "danceability": metadata["danceability"],
                    "transition": transition,
                    "notes": notes
                }
                
                analyzed_tracks.append(analyzed_track)
                prev_track_data = metadata
            
            analyzed_setlist.append({
                "time": time_range,
                "analyzed_tracks": analyzed_tracks
            })
        
        with open("analyzed_setlist.json", "w") as f:
            json.dump({"analyzed_setlist": analyzed_setlist}, f, indent=2)
        print("Analyzed setlist saved to 'analyzed_setlist.json'")
        
        return {"analyzed_setlist": analyzed_setlist}

    except Exception as e:
        print(f"Fatal Error in Track Analysis Engine: {str(e)}")
        raise

def run_mixing_engine(analyzed_setlist):
    """
    Simulates the Mixing Engine based on the analyzed setlist.
    """
    mixing_plan = {
        "plan_summary": "Auto-generated mix plan based on track metadata.",
        "mix_segments": []
    }
    
    for segment in analyzed_setlist["analyzed_setlist"]:
        mix_plan_tracks = []
        for track in segment["analyzed_tracks"]:
            mix_plan_tracks.append({
                "track": track["track"],
                "artist": track["artist"],
                "BPM": track["bpm"],
                "Key": track["key"],
                "Transition_Type": track["transition"]
            })
        
        mixing_plan["mix_segments"].append({
            "time": segment["time"],
            "tracks": mix_plan_tracks
        })

    with open("mixing_plan.json", "w") as f:
        json.dump(mixing_plan, f, indent=2)
    print("Mixing plan saved to 'mixing_plan.json'")


if __name__ == "__main__":
    sample_setlist_json = '''
    {
        "setlist": [
            {
                "time": "19:00–20:00",
                "tracks": [
                    {"title": "Tum Hi Ho", "artist": "Arijit Singh"},
                    {"title": "No Scrubs", "artist": "TLC"}
                ]
            },
            {
                "time": "20:00–21:00",
                "tracks": [
                    {"title": "Ye", "artist": "Burna Boy"},
                    {"title": "Essence", "artist": "Wizkid ft. Tems"}
                ]
            }
        ]
    }
    '''
    
    # 1. Analyze the tracks
    analyzed_setlist = analyze_tracks_in_setlist(sample_setlist_json)
    
    # 2. Run the mixing engine simulation
    print("Running Mixing Engine...")
    run_mixing_engine(analyzed_setlist)
    
    print("Pipeline complete. Check 'setlist.json', 'analyzed_setlist.json', and 'mixing_plan.json'.")
