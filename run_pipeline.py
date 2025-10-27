# run_pipeline.py
import json
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
import logging
from track_identification_engine import track_identification_engine
from track_analysis_engine import analyze_tracks_in_setlist
from generate_mixing_plan import generate_mixing_plan
from mixing_engine import generate_mix

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('colorlogger')

SONGS_DIR = "./songs"

def run_pipeline(user_input):
    """Run the full AI DJ pipeline: setlist, analysis, mixing plan, and generate MP3 mix."""
    try:
        # Step 1: Generate setlist from local songs
        logger.info("Running Track Identification Engine...")
        track_identification_engine(user_input)
        
        if not os.path.exists("setlist.json"):
            raise FileNotFoundError("setlist.json not created.")
        logger.info("Setlist generated successfully.")
        
        # Step 2: Analyze tracks
        logger.info("Running Track Analysis Engine...")
        with open("setlist.json", "r") as f:
            setlist_json = f.read()
        analyze_tracks_in_setlist(setlist_json)
        
        if not os.path.exists("analyzed_setlist.json"):
            raise FileNotFoundError("analyzed_setlist.json not created.")
        logger.info("Track analysis completed. Using advanced analysis with structural segmentation.")
        
        # Step 3: Generate mixing plan
        logger.info("Running Mixing Plan Generator...")
        with open("analyzed_setlist.json", "r") as f:
            analyzed_setlist_json = f.read()
        generate_mixing_plan(analyzed_setlist_json, eq_match_ms=15000)
        
        if not os.path.exists("mixing_plan.json"):
            raise FileNotFoundError("mixing_plan.json not created.")
        logger.info("Mixing plan generated successfully.")
        
        # Step 4: Generate MP3 mix
        logger.info("Running Mix Generator...")
        generate_mix(analyzed_setlist_json, "mixing_plan.json", eq_match_ms=15000)
        
        if not os.path.exists("mix.mp3"):
            raise FileNotFoundError("mix.mp3 not created.")
        
        logger.info("Pipeline complete. Check 'setlist.json', 'analyzed_setlist.json', 'mixing_plan.json', and 'mix.mp3'.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    user_input = (
        "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
        "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
        "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
        "{'title': 'Ye', 'artist': 'Burna Boy'}]."
    )
    run_pipeline(user_input)