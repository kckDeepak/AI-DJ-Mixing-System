# AI DJ Mixing System

## Overview

This project is an AI-powered DJ system that generates and mixes a curated setlist based on natural language user prompts (e.g., venue, vibe, time schedule, audience preferences). It processes your local MP3 songs, analyzes them for metadata like BPM, key, energy, valence, and danceability, and creates a seamless DJ mix with transitions. The system outputs JSON files for the setlist, analysis, and mixing plan, along with a final mixed MP3 file.

The system is composed of three main engines:
1. **Track Identification Engine**: Parses user input using Gemini AI to select songs from your local library that fit the prompt.
2. **Track Analysis Engine**: Analyzes selected MP3s using Librosa to extract audio features and sorts tracks for optimal transitions.
3. **Mixing Engine**: Applies DJ transitions (e.g., crossfade, EQ sweep) using PyDub and generates the final mix MP3.

This is designed for personal use with your own legally obtained MP3 songs. It runs offline except for the initial Gemini API call for setlist generation.

## Prerequisites

To run this project, you need:
- **Python 3.8+**: Installed on your system.
- **FFmpeg**: Required for audio processing with PyDub. Download and install from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH.
- **Google Gemini API Key**: Sign up at [ai.google.dev](https://ai.google.dev) to get a free API key for the Gemini model. Add it to a `.env` file in the project root as `GEMINI_API_KEY=your_key_here`.
- **Local MP3 Songs**: You must have your own MP3 files in the `./songs` directory. The system assumes filenames like "Artist - Title.mp3" or similar (it cleans names like "[iSongs.info] 01 - Song.mp3"). Without songs, the pipeline won't work. Collect legal MP3s matching your preferred genres (e.g., R&B, Bollywood, Afrobeats).

Note: Processing a 3-hour mix takes approximately 10 minutes, depending on your hardware (due to audio analysis and mixing).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-dj-mixing-system.git
   cd ai-dj-mixing-system
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies from `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist yet, create it with these packages:
   ```
   google-generativeai
   librosa
   numpy
   pydub
   python-dotenv
   ```

## Usage

1. Place your MP3 songs in the `./songs` folder.

2. Update the user prompt in `run_pipeline.py` (or pass it as needed). Example prompt:
   ```
   user_input = (
       "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
       "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
       "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
       "{'title': 'Ye', 'artist': 'Burna Boy'}]."
   )
   ```

3. Run the pipeline:
   ```
   python run_pipeline.py
   ```

   This will:
   - Generate `setlist.json`: Selected songs (unordered pool) per time segment.
   - Generate `analyzed_setlist.json`: Analyzed features and sorted order for smooth transitions.
   - Generate `mixing_plan.json`: Timings, song names, transition types (e.g., crossfade, EQ sweep), and comments.
   - Export `mix.mp3`: The final DJ-mixed audio file.

## Code Structure

- **track_identification_engine.py**: Uses Gemini AI to parse prompts and select songs from your local `./songs` directory.
- **track_analysis_engine.py**: Analyzes MP3s with Librosa, computes features, and sorts tracks by BPM for better mixing.
- **mixing_engine.py**: Applies transitions with PyDub based on analysis and exports the MP3 mix.
- **run_pipeline.py**: Orchestrates the full process from prompt to output.

The pipeline is modular: Identification selects songs, Analysis extracts/sorts, Mixing builds the plan and audio.

## Output Example

- **setlist.json**: Time segments with selected songs (e.g., tracks as list of {"title", "artist", "file"}).
- **analyzed_setlist.json**: Features like BPM, energy, and sorted order per segment.
- **mixing_plan.json**: Array of transitions with start times, types (varied like crossfade, EQ sweep), and notes.
- **mix.mp3**: A single MP3 file with the full mix, including DJ effects.

For a 3-hour casino mix, expect varied transitions beyond just fades, thanks to BPM/energy-based logic.

## License

MIT License. Feel free to modify and use! If you encounter issues, check console logs for errors like missing files or API keys.