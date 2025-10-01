from track_identification_engine import track_identification_engine
from track_analysis_engine import analyze_tracks_in_setlist
from mixing_engine import generate_mixing_plan

user_input = (
    "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
    "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
    # "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
    # "{'title': 'Ye', 'artist': 'Burna Boy'}]."
)
setlist_json = track_identification_engine(user_input)
analyzed_setlist_json = analyze_tracks_in_setlist(setlist_json)
mixing_plan_json = generate_mixing_plan(analyzed_setlist_json)
print(mixing_plan_json)