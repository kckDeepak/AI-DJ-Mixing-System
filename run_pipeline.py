# run_pipeline.py
"""
This module orchestrates the full AI DJ pipeline, integrating multiple stages to generate a DJ mix from user input.
It coordinates the track identification, track analysis, mixing plan generation, and final mix creation, producing
a seamless MP3 mix based on user preferences and local audio files.

Key features:
- Executes four main stages: track identification, track analysis, mixing plan generation, and mix generation.
- Uses logging to track progress and errors for debugging and monitoring.
- Validates the existence of intermediate output files (JSON and MP3) to ensure pipeline integrity.
- Handles errors gracefully, logging issues and raising exceptions for caller handling.

Dependencies:
- json: For reading intermediate JSON files.
- os: For file path operations and environment variable management.
- warnings: To suppress non-critical Numba warnings.
- logging: For detailed logging of pipeline execution.
- track_identification_engine: Generates the initial setlist from user input.
- track_analysis_engine: Analyzes tracks for audio features (e.g., BPM, key).
- generate_mixing_plan: Creates a mixing plan with transition types and timings.
- mixing_engine: Generates the final MP3 mix with applied transitions.
"""

import json  # Used for reading intermediate JSON files (setlist, analyzed setlist, mixing plan).
import os  # Used for file path operations and setting environment variables.
os.environ["NUMBA_DISABLE_JIT"] = "1"  # Disable Numba JIT compilation to avoid potential performance issues.
import warnings  # Used to suppress non-critical warnings from Numba.
warnings.filterwarnings("ignore", category=UserWarning, module="numba")  # Ignore Numba-related UserWarnings.
import logging  # Used for logging pipeline progress and errors.
from track_identification_engine import track_identification_engine  # Generates setlist from user input.
from track_analysis_engine import analyze_tracks_in_setlist  # Analyzes tracks for audio features.
from generate_mixing_plan import generate_mixing_plan  # Creates a mixing plan with transitions.
from mixing_engine import generate_mix  # Generates the final MP3 mix.

# Setup logging with DEBUG level and a custom format including timestamp, level, and message.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('colorlogger')  # Create a logger instance named 'colorlogger'.

# Define the directory path where local MP3 song files are stored (relative to the script's execution directory).
SONGS_DIR = "./songs"


def run_pipeline(user_input):
    """
    Runs the full AI DJ pipeline to generate a DJ mix from user input.

    Process:
    1. Track Identification: Generates a setlist based on user input and local songs.
    2. Track Analysis: Analyzes tracks for audio features (e.g., BPM, key, energy).
    3. Mixing Plan: Creates a mixing plan with transition types, timings, and OTAC values.
    4. Mix Generation: Produces an MP3 mix with applied transitions.
    - Validates the existence of intermediate output files (JSON and MP3).
    - Logs progress and errors for debugging and monitoring.

    Args:
        user_input (str): User-provided description of the desired mix, including time segments, genres, and specific songs.

    Raises:
        FileNotFoundError: If any expected output file (setlist.json, analyzed_setlist.json, mixing_plan.json, mix.mp3) is not created.
        Exception: For other errors during pipeline execution, logged and re-raised for caller handling.
    """
    try:
        # Step 1: Generate setlist from local songs using the track identification engine.
        logger.info("Running Track Identification Engine...")
        track_identification_engine(user_input)  # Generate setlist.json based on user input.
        
        # Verify that the setlist file was created.
        if not os.path.exists("setlist.json"):
            raise FileNotFoundError("setlist.json not created.")
        logger.info("Setlist generated successfully.")
        
        # Step 2: Analyze tracks in the setlist for audio features.
        logger.info("Running Track Analysis Engine...")
        with open("setlist.json", "r") as f:
            setlist_json = f.read()  # Read the generated setlist JSON.
        analyze_tracks_in_setlist(setlist_json)  # Analyze tracks and save to analyzed_setlist.json.
        
        # Verify that the analyzed setlist file was created.
        if not os.path.exists("analyzed_setlist.json"):
            raise FileNotFoundError("analyzed_setlist.json not created.")
        logger.info("Track analysis completed. Using advanced analysis with structural segmentation.")
        
        # Step 3: Generate a mixing plan based on the analyzed setlist.
        logger.info("Running Mixing Plan Generator...")
        with open("analyzed_setlist.json", "r") as f:
            analyzed_setlist_json = f.read()  # Read the analyzed setlist JSON.
        generate_mixing_plan(analyzed_setlist_json, eq_match_ms=15000)  # Generate mixing_plan.json with 15s EQ transitions.
        
        # Verify that the mixing plan file was created.
        if not os.path.exists("mixing_plan.json"):
            raise FileNotFoundError("mixing_plan.json not created.")
        logger.info("Mixing plan generated successfully.")
        
        # Step 4: Generate the final MP3 mix using the analyzed setlist and mixing plan.
        logger.info("Running Mix Generator...")
        generate_mix(analyzed_setlist_json, "mixing_plan.json", eq_match_ms=15000)  # Generate mix.mp3.
        
        # Verify that the final mix file was created.
        if not os.path.exists("mix.mp3"):
            raise FileNotFoundError("mix.mp3 not created.")
        
        # Log successful completion with a summary of output files.
        logger.info("Pipeline complete. Check 'setlist.json', 'analyzed_setlist.json', 'mixing_plan.json', and 'mix.mp3'.")
    except Exception as e:
        # Log any errors encountered during pipeline execution and re-raise for caller handling.
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Entry point for testing the AI DJ pipeline with a sample user input.

    Defines a sample user input specifying a mix for a casino event with time segments, preferred genres,
    and specific song requests. Calls run_pipeline() to execute the full pipeline and generate the mix.
    """
    user_input = (
        "Just create a mix between i ain't worried, Closer."
    )
    # Execute the pipeline with the sample user input.
    run_pipeline(user_input)