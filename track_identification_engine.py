# track_identification_engine.py
import json
import google.generativeai as genai
from datetime import datetime
import random
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Directory for local MP3 songs
SONGS_DIR = "./songs"

def get_available_songs():
    """Scan the songs directory and return a list of available songs."""
    available_songs = []
    for filename in os.listdir(SONGS_DIR):
        if filename.lower().endswith(".mp3"):
            # Clean filename to extract artist and title
            clean_name = filename[:-4]
            if clean_name.startswith("[iSongs.info] "):
                clean_name = clean_name.split(" - ", 1)[-1] if " - " in clean_name else clean_name.split(" ", 2)[-1]
            parts = clean_name.split(" - ", 1)
            if len(parts) == 2:
                artist, title = parts
            else:
                artist = "Unknown"
                title = clean_name
            available_songs.append({"title": title, "artist": artist, "file": filename})
    return available_songs

def parse_time_segments_and_generate_setlist(user_input, available_songs):
    """Parse user input and generate setlist using LLM, considering available local songs."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    available_songs_str = json.dumps(available_songs, indent=2)
    
    prompt = f"""
    Parse the following user input into time segments, preferred genres, and specific songs.
    Then, generate a structured setlist by selecting a pool of songs (unordered) from the available local songs list below for each time segment.
    Prioritize specific songs if they match available ones. Select songs that fit the genres, vibe, and description.
    Ensure the number of tracks approximately covers the time duration (3-4 min per track). Do not order the tracks yet; provide them as an unordered list for each segment.
    
    Available local songs: {available_songs_str}
    
    Input: "{user_input}"
    
    Output format:
    {{
        "time_segments": [
            {{"start": "HH:MM", "end": "HH:MM", "description": "string"}},
            ...
        ],
        "genres": ["genre1", "genre2", ...],
        "specific_songs": [{{"title": "string", "artist": "string"}}, ...],
        "setlist": [
            {{
                "time": "HH:MM–HH:MM",
                "tracks": [
                    {{"title": "string", "artist": "string", "file": "string.mp3"}},
                    ...
                ]
            }},
            ...
        ]
    }}
    Provide ONLY the JSON object in your response.
    """
    
    response = model.generate_content(prompt)
    
    json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = response.text.strip()

    try:
        parsed_data = json.loads(json_string)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"DEBUG: Failed to parse JSON. Raw response text: '{response.text}'")
        raise ValueError("Failed to parse Gemini response into JSON") from e

def track_identification_engine(user_input):
    """Process user input and save setlist to JSON file."""
    try:
        available_songs = get_available_songs()
        data = parse_time_segments_and_generate_setlist(user_input, available_songs)
        output = {
            "setlist": data["setlist"],
            "genres": data["genres"],
            "specific_songs": data["specific_songs"]
        }
        
        with open("setlist.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Setlist saved to 'setlist.json'")
    except Exception as e:
        print(f"Error in Track Identification Engine: {str(e)}")
        raise

if __name__ == "__main__":
    user_input = (
        "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
        "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
        "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
        "{'title': 'Ye', 'artist': 'Burna Boy'}]."
    )
    track_identification_engine(user_input)