import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify API Configuration
SPOTIFY_CLIENT_ID = ""  # Replace with your Spotify Client ID
SPOTIFY_CLIENT_SECRET = ""  # Replace with your Spotify Client Secret
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8000/callback"  # Replace if needed
SPOTIFY_SCOPE = "playlist-read-private playlist-read-collaborative"

# Initialize Spotify client with OAuth
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SPOTIFY_SCOPE
))

def search_track(title, artist):
    """
    Search for a track on Spotify and return its audio features.
    Args:
        title (str): Track title.
        artist (str): Track artist.
    Returns:
        dict: Metadata with bpm, key, genre, energy, valence, danceability.
    """
    try:
        # Search for the track
        results = sp.search(q=f'track:{title} artist:{artist}', type='track', limit=1)
        if not results['tracks']['items']:
            raise ValueError(f"Track '{title}' by '{artist}' not found on Spotify.")
        
        track = results['tracks']['items'][0]
        track_id = track['id']
        
        # Fetch audio features
        audio_features = sp.audio_features(tracks=[track_id])[0]
        if not audio_features:
            raise ValueError(f"Audio features not available for '{title}'.")
        
        # Fetch artist genres
        artist_id = track['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        genres = artist_info.get('genres', [])
        primary_genre = genres[0] if genres else "Unknown"
        
        # Convert numerical key to musical key name (e.g., 1 major = C#)
        key_num = audio_features['key']
        mode = audio_features['mode']  # 0 = minor, 1 = major
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_name = key_names[key_num] if key_num < len(key_names) else "Unknown"
        if mode == 0:
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
        print(f"Error fetching data for '{title}' by '{artist}': {str(e)}")
        # Fallback metadata if Spotify API fails
        return {
            "bpm": 120,
            "key": "C#m",
            "genre": "Unknown",
            "energy": 0.7,
            "valence": 0.6,
            "danceability": 0.8
        }

def compute_vibe_label(energy, valence, danceability):
    """
    Compute contextual vibe label based on metadata.
    Args:
        energy (float): Track energy (0-1).
        valence (float): Track valence (0-1).
        danceability (float): Track danceability (0-1).
    Returns:
        str: Vibe label (e.g., 'Peak Energy', 'Sunset Chill').
    """
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
    """
    Suggest transition type based on previous and current track metadata.
    Args:
        prev_track_data (dict): Metadata of previous track (or None if first).
        current_track_data (dict): Metadata of current track.
    Returns:
        str: Transition type (e.g., 'Seamless Beatmatch', 'Fade In').
    """
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
    elif curr_energy < prev_energy - 0.2:
        return "Fade Out Transition"
    return "Crossfade"

def generate_notes(vibe_label, energy, danceability, genre):
    """
    Generate contextual notes for the track.
    Args:
        vibe_label (str): Computed vibe label.
        energy (float): Track energy.
        danceability (float): Track danceability.
        genre (str): Track genre.
    Returns:
        str: Descriptive notes for DJ use.
    """
    notes = f"{vibe_label} track. "
    if energy > 0.7:
        notes += "High energy suitable for peak moments. "
    if danceability > 0.8:
        notes += "Excellent for dancing. "
    notes += f"Genre: {genre}."
    return notes

def analyze_tracks_in_setlist(setlist_json):
    """
    Analyze tracks from Engine 1 setlist using Spotify API.
    Args:
        setlist_json (str): JSON string from Track Identification Engine.
    Returns:
        str: JSON string with analyzed tracks and metadata.
    """
    setlist_data = json.loads(setlist_json)
    analyzed_setlist = []
    
    for segment in setlist_data["setlist"]:
        time_range = segment["time"]
        tracks = segment["tracks"]
        analyzed_tracks = []
        prev_track_data = None  # Track previous track for transitions
        
        for track in tracks:
            title = track["title"]
            artist = track["artist"]
            
            # Fetch metadata from Spotify
            metadata = search_track(title, artist)
            
            # Compute custom attributes
            vibe_label = compute_vibe_label(
                metadata["energy"], 
                metadata["valence"], 
                metadata["danceability"]
            )
            transition = suggest_transition(prev_track_data, metadata)
            notes = generate_notes(vibe_label, metadata["energy"], metadata["danceability"], metadata["genre"])
            
            # Build analyzed track object
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
            prev_track_data = metadata  # Update for next transition
        
        analyzed_setlist.append({
            "time": time_range,
            "analyzed_tracks": analyzed_tracks
        })
    
    return json.dumps({"analyzed_setlist": analyzed_setlist}, indent=2)

# Example usage
if __name__ == "__main__":
    # Sample input from Track Identification Engine
    sample_setlist_json = '''
    {
        "setlist": [
    {
      "time": "19:00\u201320:00",
      "tracks": [
        {
          "title": "Kal Ho Naa Ho",
          "artist": "Sonu Nigam"
        },
        {
          "title": "Tum Hi Ho",
          "artist": "Arijit Singh"
        },
        {
          "title": "Kal Ho Naa Ho",
          "artist": "Sonu Nigam"
        }
      ]
    },
    {
      "time": "20:00\u201321:00",
      "tracks": [
        {
          "title": "Tum Hi Ho",
          "artist": "Arijit Singh"
        },
        {
          "title": "Joro",
          "artist": "Wizkid"
        },
        {
          "title": "Ye",
          "artist": "Burna Boy"
        },
        {
          "title": "On the Low",
          "artist": "Wizkid"
        },
        {
          "title": "No Scrubs",
          "artist": "TLC"
        },
        {
          "title": "No Scrubs",
          "artist": "TLC"
        },
        {
          "title": "Tum Hi Ho",
          "artist": "Arijit Singh"
        },
        {
          "title": "Joro",
          "artist": "Wizkid"
        },
        {
          "title": "On the Low",
          "artist": "Wizkid"
        }
      ]
    },
    {
      "time": "21:00\u201322:00",
      "tracks": [
        {
          "title": "Tum Hi Ho",
          "artist": "Arijit Singh"
        },
        {
          "title": "On the Low",
          "artist": "Wizkid"
        },
        {
          "title": "No Scrubs",
          "artist": "TLC"
        },
        {
          "title": "Joro",
          "artist": "Wizkid"
        },
        {
          "title": "Ye",
          "artist": "Burna Boy"
        },
        {
          "title": "On the Low",
          "artist": "Wizkid"
        },
        {
          "title": "Joro",
          "artist": "Wizkid"
        },
        {
          "title": "Ye",
          "artist": "Burna Boy"
        },
        {
          "title": "No Scrubs",
          "artist": "TLC"
        },
        {
          "title": "Tum Hi Ho",
          "artist": "Arijit Singh"
        }
      ]
    }
  ]
    }
    '''
    
    result = analyze_tracks_in_setlist(sample_setlist_json)
    print(result)