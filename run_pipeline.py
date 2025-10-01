import json
from track_identification_engine import track_identification_engine
from track_analysis_engine import analyze_tracks_in_setlist
from mixing_engine import generate_mixing_plan
import os

def run_pipeline(user_input):
    """Run the full AI DJ pipeline and generate JSON files."""
    try:
        # Step 1: Generate setlist
        print("Running Track Identification Engine...")
        track_identification_engine(user_input)
        
        # Verify setlist.json exists
        if not os.path.exists("setlist.json"):
            raise FileNotFoundError("setlist.json was not created. Check Track Identification Engine.")
        
        # Step 2: Analyze setlist
        print("Running Track Analysis Engine...")
        with open("setlist.json", "r") as f:
            setlist_json = f.read()
        analyze_tracks_in_setlist(setlist_json)
        
        # Verify analyzed_setlist.json exists
        if not os.path.exists("analyzed_setlist.json"):
            raise FileNotFoundError("analyzed_setlist.json was not created. Check Track Analysis Engine.")
        
        # Step 3: Generate mixing plan
        print("Running Mixing Engine...")
        with open("analyzed_setlist.json", "r") as f:
            analyzed_setlist_json = f.read()
        generate_mixing_plan(analyzed_setlist_json)
        
        # Verify mixing_plan.json exists
        if not os.path.exists("mixing_plan.json"):
            raise FileNotFoundError("mixing_plan.json was not created. Check Mixing Engine.")
        
        print("Pipeline complete. Check 'setlist.json', 'analyzed_setlist.json', and 'mixing_plan.json'.")
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