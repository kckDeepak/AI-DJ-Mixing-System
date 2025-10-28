# track_identification_engine.py
"""
This module implements a track identification engine that processes user input to generate a structured setlist
for events, leveraging the Google Gemini AI model. It scans a local directory for MP3 files, parses user preferences
(including time segments, genres, and specific song requests), and generates an unordered pool of available tracks
that match the criteria. The output is saved as a JSON file for further use.

Key features:
- Scans local MP3 files and extracts metadata (artist and title).
- Uses Gemini AI to interpret user input and select appropriate tracks.
- Handles unavailable songs by logging them separately.
- Ensures track selections approximate the desired time duration (assuming 3-4 minutes per track).

Dependencies:
- google-generativeai: For interacting with the Gemini API.
- python-dotenv: For loading environment variables.
- Standard library: json, datetime, random, os, re.
"""

import json  # Used for parsing and serializing JSON data, including available songs and setlist output.
import google.generativeai as genai  # Google Generative AI library for accessing the Gemini model.
from datetime import datetime  # Imported but not currently used; reserved for potential timestamping in future expansions.
import random  # Imported but not currently used; reserved for potential randomization of track selections if needed.
from dotenv import load_dotenv  # Library for loading environment variables from a .env file.
import os  # Standard library for file and directory operations, such as listing songs and accessing environment variables.
import re  # Regular expressions module for extracting JSON from the AI response if wrapped in code blocks.


# Load environment variables from the .env file to securely manage API keys and configurations.
load_dotenv()

# Configure the Gemini API by setting the API key retrieved from the environment variable.
# This key is expected to be set in the .env file as GEMINI_API_KEY.
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the directory path where local MP3 song files are stored.
# This path is relative to the script's execution directory.
SONGS_DIR = "./songs"


def get_available_songs():
    """
    Scans the specified songs directory (SONGS_DIR) for MP3 files and returns a list of dictionaries
    containing metadata for each available song.

    For each MP3 file:
    - Extracts a clean name by removing the .mp3 extension and handling common prefixes like "[iSongs.info] ".
    - Attempts to split the clean name into artist and title using " - " as a delimiter.
    - If splitting fails, defaults the artist to "Unknown".

    Returns:
        list: A list of dictionaries, each with keys 'title' (str), 'artist' (str), and 'file' (str, original filename).
    """
    available_songs = []  # Initialize an empty list to store song metadata dictionaries.
    for filename in os.listdir(SONGS_DIR):  # Iterate over all files in the songs directory.
        if filename.lower().endswith(".mp3"):  # Check if the file is an MP3 (case-insensitive).
            # Clean filename to extract artist and title:
            # Remove the .mp3 extension.
            clean_name = filename[:-4]
            # Handle common download prefixes like "[iSongs.info] " by removing it and extracting the relevant part.
            if clean_name.startswith("[iSongs.info] "):
                # Split on " - " and take the last part if it exists; otherwise, take the last two space-separated parts.
                clean_name = clean_name.split(" - ", 1)[-1] if " - " in clean_name else clean_name.split(" ", 2)[-1]
            # Attempt to split the clean name into artist and title using " - " as the delimiter.
            parts = clean_name.split(" - ", 1)
            if len(parts) == 2:  # If exactly two parts are found (artist and title).
                artist, title = parts  # Assign the parts to artist and title variables.
            else:  # If splitting fails (e.g., no " - " delimiter).
                artist = "Unknown"  # Default artist to "Unknown".
                title = clean_name  # Use the entire clean name as the title.
            # Append a dictionary with the extracted metadata and original filename to the list.
            available_songs.append({"title": title, "artist": artist, "file": filename})
    return available_songs  # Return the populated list of available songs.


def parse_time_segments_and_generate_setlist(user_input, available_songs):
    """
    Uses the Gemini AI model to parse the user input into time segments, genres, and specific song requests,
    then generates an unordered setlist of available local songs that match the criteria.

    Process:
    - Converts the available_songs list to a JSON string for inclusion in the prompt.
    - Constructs a detailed prompt instructing the model to:
      - Parse time segments (start/end times and descriptions).
      - Extract preferred genres and specific songs.
      - Select matching tracks from available local songs, prioritizing exact matches.
      - Log unavailable specific songs separately.
      - Estimate track counts based on time durations (3-4 min/track).
      - Output in a strict JSON format.
    - Generates content using the "gemini-2.5-flash" model.
    - Extracts the JSON from the response (handling potential Markdown code block wrappers).
    - Parses the JSON and returns the data.

    Args:
        user_input (str): The user's natural language description of the event, including times, genres, and songs.
        available_songs (list): List of song dictionaries from get_available_songs().

    Returns:
        dict: Parsed JSON data with keys: time_segments, genres, specific_songs, unavailable_songs, setlist.

    Raises:
        ValueError: If the AI response cannot be parsed into valid JSON.
    """
    # Initialize the GenerativeModel instance with the specified Gemini model variant.
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Serialize the available_songs list to a formatted JSON string for easy inclusion in the prompt.
    available_songs_str = json.dumps(available_songs, indent=2)
    
    # Construct the prompt as a multi-line f-string, providing instructions and context to the AI model.
    prompt = f"""
    Parse the following user input into time segments, preferred genres, and specific songs.
    Then, generate a structured setlist by selecting a pool of songs (unordered) from the available local songs list below for each time segment.
    Prioritize specific songs if they match available ones exactly by title and artist. 
    For specific songs mentioned in the input that are not in available local songs, list them in unavailable_songs with reason 'not found'.
    Do not include unavailable songs in the setlist tracks.
    Select songs that fit the genres, vibe, and description.
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
        "unavailable_songs": [{{"title": "string", "artist": "string", "reason": "string"}}, ...],
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
    
    # Generate the AI response using the constructed prompt.
    response = model.generate_content(prompt)
    
    # Attempt to extract JSON from the response text, handling cases where it may be wrapped in ```json ... ``` Markdown blocks.
    json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
    if json_match:  # If a Markdown-wrapped JSON block is found.
        json_string = json_match.group(1)  # Extract the inner JSON content.
    else:  # If no wrapper is found, use the raw response text (stripped of whitespace).
        json_string = response.text.strip()

    # Parse the extracted string into a Python dictionary.
    try:
        parsed_data = json.loads(json_string)
        return parsed_data  # Return the successfully parsed data.
    except json.JSONDecodeError as e:  # Handle JSON parsing errors.
        # Print debug information: the raw response text for troubleshooting.
        print(f"DEBUG: Failed to parse JSON. Raw response text: '{response.text}'")
        # Raise a ValueError with a user-friendly message, chaining the original exception for details.
        raise ValueError("Failed to parse Gemini response into JSON") from e


def track_identification_engine(user_input):
    """
    Main entry point for the track identification engine. Orchestrates the process of generating and saving a setlist.

    Process:
    - Retrieves the list of available local songs.
    - Parses the user input using the AI model to generate setlist data.
    - Constructs an output dictionary, including the setlist, genres, specific songs, and unavailable songs.
    - Saves the output to a file named "setlist.json" in the current directory.
    - Prints a success message upon completion.

    Args:
        user_input (str): The user's natural language input describing the event.

    Raises:
        Exception: Any errors during processing (e.g., file I/O, AI parsing) are caught, logged, and re-raised.
    """
    try:  # Wrap the entire process in a try-except block for comprehensive error handling.
        # Retrieve the list of available songs from the local directory.
        available_songs = get_available_songs()
        # Parse the user input and generate the setlist data using the AI model.
        data = parse_time_segments_and_generate_setlist(user_input, available_songs)
        # Construct the final output dictionary, extracting relevant fields from the parsed data.
        # Note: unavailable_songs defaults to an empty list if not present in the parsed data.
        output = {
            "setlist": data["setlist"],
            "genres": data["genres"],
            "specific_songs": data["specific_songs"],
            "unavailable_songs": data.get("unavailable_songs", [])
        }
        
        # Open a file named "setlist.json" in write mode and serialize the output dictionary to it with indentation.
        with open("setlist.json", "w") as f:
            json.dump(output, f, indent=2)
        # Print a confirmation message to indicate successful file creation.
        print("Setlist saved to 'setlist.json'")
    except Exception as e:  # Catch any exceptions during execution.
        # Print an error message with the exception details for logging.
        print(f"Error in Track Identification Engine: {str(e)}")
        # Re-raise the exception to allow the caller to handle it if needed.
        raise


if __name__ == "__main__":  # Standard Python idiom to execute code only if the script is run directly (not imported).
    # Define a sample user input string for demonstration purposes.
    # This example includes time segments, event descriptions, genres, and specific song requests.
    user_input = (
        "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
        "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
        "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
        "{'title': 'Ye', 'artist': 'Burna Boy'}]."
    )
    # Invoke the main engine function with the sample user input to generate and save the setlist.
    track_identification_engine(user_input)