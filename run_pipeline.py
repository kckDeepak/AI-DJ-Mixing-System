# run_pipeline.py
import json
from track_identification_engine import track_identification_engine
from track_analysis_engine import analyze_tracks_in_setlist
from mixing_engine import generate_mixing_plan_and_mix
import os

def run_pipeline(user_input):
    """Run the full AI DJ pipeline: setlist, analysis, mixing plan, and generate MP3 mix."""
    try:
        # Step 1: Generate setlist from local songs
        print("Running Track Identification Engine...")
        track_identification_engine(user_input)
        
        if not os.path.exists("setlist.json"):
            raise FileNotFoundError("setlist.json not created.")
        
        # Step 2: Analyze tracks
        print("Running Track Analysis Engine...")
        with open("setlist.json", "r") as f:
            setlist_json = f.read()
        analyzed_setlist = analyze_tracks_in_setlist(setlist_json)
        
        if not os.path.exists("analyzed_setlist.json"):
            raise FileNotFoundError("analyzed_setlist.json not created.")
        
        # Step 3: Generate mixing plan and MP3 mix
        print("Running Mixing Engine...")
        analyzed_setlist_json = json.dumps(analyzed_setlist)
        generate_mixing_plan_and_mix(analyzed_setlist_json)
        
        if not os.path.exists("mixing_plan.json") or not os.path.exists("mix.mp3"):
            raise FileNotFoundError("mixing_plan.json or mix.mp3 not created.")
        
        print("Pipeline complete. Check 'setlist.json', 'analyzed_setlist.json', 'mixing_plan.json', and 'mix.mp3'.")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    user_input = (
        "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
        "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
        "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
        "{'title': 'Ye', 'artist': 'Burna Boy'}]."
    )
    run_pipeline(user_input)