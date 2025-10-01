import json
import google.generativeai as genai
from datetime import datetime, timedelta
import random

# Configure Gemini API (replace with your API key)
genai.configure(api_key="AIzaSyA3NhSUvHbESoedS0O4TLKRbQoI6XyXM5s")  # Replace with actual API key

# Simulated BPM data for songs (since we don't have Spotify API here)
# In a real implementation, this would come from the Track Analysis Engine
SONG_BPM_DB = {
    "R&B": [
        {"title": "Adore You", "artist": "Miley Cyrus", "bpm": 120},
        {"title": "No Scrubs", "artist": "TLC", "bpm": 93},
        {"title": "Say My Name", "artist": "Destiny's Child", "bpm": 138},
        {"title": "Blinding Lights", "artist": "The Weeknd", "bpm": 171}
    ],
    "Bollywood": [
        {"title": "Tum Hi Ho", "artist": "Arijit Singh", "bpm": 94},
        {"title": "Kal Ho Naa Ho", "artist": "Sonu Nigam", "bpm": 84},
        {"title": "Badtameez Dil", "artist": "Benny Dayal", "bpm": 106},
        {"title": "Dilli Wali Girlfriend", "artist": "Arijit Singh", "bpm": 115}
    ],
    "Afrobeats": [
        {"title": "Ye", "artist": "Burna Boy", "bpm": 100},
        {"title": "On the Low", "artist": "Wizkid", "bpm": 97},
        {"title": "Essence", "artist": "Wizkid ft. Tems", "bpm": 104},
        {"title": "Joro", "artist": "Wizkid", "bpm": 98}
    ]
}

import json
import re
import google.generativeai as genai

# ... (rest of your existing code)

def parse_time_segments(user_input):
    """Parse user input to extract time segments and preferences."""
    model = genai.GenerativeModel("gemini-2.5-pro")
    
    prompt = f"""
    Parse the following user input into a structured JSON format with time segments, preferred genres, and specific songs (if any).
    Input: "{user_input}"
    
    Output format:
    {{
        "time_segments": [
            {{"start": "HH:MM", "end": "HH:MM", "description": "string"}},
            ...
        ],
        "genres": ["genre1", "genre2", ...],
        "specific_songs": [{{"title": "string", "artist": "string"}}, ...]
    }}
    Provide ONLY the JSON object in your response. Do not include any additional text, explanations, or code block markers.
    """
    
    response = model.generate_content(prompt)
    
    # Attempt to find the JSON within a potential code block
    json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = response.text.strip() # Fallback to the whole response

    try:
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"DEBUG: Failed to parse JSON. Raw response text: '{response.text}'")
        print(f"DEBUG: String being parsed: '{json_string}'")
        raise ValueError("Failed to parse Gemini response into JSON") from e

def generate_setlist(parsed_data):
    """Generate a setlist ensuring BPM difference of ±5 between consecutive tracks."""
    setlist = []
    
    for segment in parsed_data["time_segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        description = segment["description"].lower()
        
        # Determine duration of segment in minutes
        start_dt = datetime.strptime(start_time, "%H:%M")
        end_dt = datetime.strptime(end_time, "%H:%M")
        duration = (end_dt - start_dt).total_seconds() / 60
        
        # Estimate number of tracks (assuming ~3.5 minutes per track)
        num_tracks = int(duration // 3.5)
        if num_tracks < 1:
            num_tracks = 1
        
        # Collect candidate tracks
        candidate_tracks = []
        genres = parsed_data["genres"]
        specific_songs = parsed_data["specific_songs"]
        
        # Add specific songs first
        for song in specific_songs:
            for genre, tracks in SONG_BPM_DB.items():
                if any(s["title"] == song["title"] and s["artist"] == song["artist"] for s in tracks):
                    candidate_tracks.append(song)
        
        # Add genre-based tracks
        for genre in genres:
            if genre in SONG_BPM_DB:
                candidate_tracks.extend(SONG_BPM_DB[genre])
        
        # Shuffle candidates to add variety
        random.shuffle(candidate_tracks)
        
        # Select tracks with BPM constraints
        selected_tracks = []
        last_bpm = None
        
        for track in candidate_tracks:
            if len(selected_tracks) >= num_tracks:
                break
                
            current_bpm = track.get("bpm", random.randint(80, 140))  # Fallback BPM if not provided
            if last_bpm is None or abs(last_bpm - current_bpm) <= 5:
                selected_tracks.append({"title": track["title"], "artist": track["artist"]})
                last_bpm = current_bpm
        
        # If not enough tracks, fill with random tracks from genres
        while len(selected_tracks) < num_tracks and candidate_tracks:
            track = random.choice(candidate_tracks)
            current_bpm = track.get("bpm", random.randint(80, 140))
            if last_bpm is None or abs(last_bpm - current_bpm) <= 5:
                selected_tracks.append({"title": track["title"], "artist": track["artist"]})
                last_bpm = current_bpm
            candidate_tracks.remove(track)
        
        setlist.append({
            "time": f"{start_time}–{end_time}",
            "tracks": selected_tracks
        })
    
    return setlist

def track_identification_engine(user_input):
    """Main function to process user input and generate setlist."""
    # Step 1: Parse user input
    parsed_data = parse_time_segments(user_input)
    
    # Step 2: Generate setlist
    setlist = generate_setlist(parsed_data)
    
    # Step 3: Output JSON
    output = {"setlist": setlist}
    return json.dumps(output, indent=2)

# Example usage
if __name__ == "__main__":
    user_input = (
        "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
        "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
        "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
        "{'title': 'Ye', 'artist': 'Burna Boy'}]."
    )
    result = track_identification_engine(user_input)
    print(result)