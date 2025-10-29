# generate_mixing_plan.py
"""
This module generates a DJ mixing plan based on an analyzed setlist, producing a sequence of track transitions with
timing, transition types, and tempo adjustments. It uses audio metadata (e.g., BPM, key, vocals, choruses) to suggest optimal
transition types and calculates start times and overlaps for a seamless mix. The output is saved as a JSON file for use
in DJ software or event planning.

Key features:
- Estimates transition overlaps based on transition type (e.g., crossfade, EQ sweep, Chorus Beatmatch).
- Checks harmonic compatibility between tracks using key semitone differences.
- Computes OTAC (Optimal Tempo Adjustment Coefficient) for smooth tempo transitions.
- Suggests transition types based on BPM, key compatibility, vocal presence, and chorus availability for mid-song transitions.
- Supports "Chorus Beatmatch" transitions at the end of the outgoing track's chorus if beats align (BPM diff <= 2).
- Generates a mixing plan with start times, transition points, cut offsets, and comments for each track.

Dependencies:
- json: For parsing input setlist and serializing the mixing plan.
- os: For file path operations.
- numpy: For numerical computations (e.g., OTAC calculation).
- datetime, timedelta: For handling time calculations and formatting.
- pydub: For loading audio files to determine track durations.
"""

import os  # Used for constructing file paths to access MP3 files.
import json  # Used for parsing input setlist JSON and serializing output to JSON.
import numpy as np  # Used for numerical computations, such as OTAC calculation.
from datetime import datetime, timedelta  # Used for calculating and formatting track start times.
from pydub import AudioSegment  # Used to load MP3 files and determine their durations.

# Define the directory path where local MP3 song files are stored (relative to the script's execution directory).
SONGS_DIR = "./songs"


# ---------------------------
# Estimated overlap for timing
# ---------------------------
def get_estimated_overlap(transition_type: str, crossfade_early_ms: int = 5500, duration_ms: int = 8000, eq_match_ms: int = 15000):
    """
    Estimates the overlap duration (in seconds) for a transition based on the transition type.

    Transition types and overlaps:
    - Fade In: No overlap (0.0 seconds).
    - EQ Sweep: Uses eq_match_ms (default 15 seconds) for longer transitions.
    - Chorus Beatmatch: Same as Crossfade (default 13.5 seconds).
    - Other transitions (e.g., Crossfade): Uses crossfade_early_ms + duration_ms (default 13.5 seconds).

    Args:
        transition_type (str): Type of transition (e.g., 'Fade In', 'Crossfade', 'EQ Sweep', 'Chorus Beatmatch').
        crossfade_early_ms (int, optional): Early crossfade duration in milliseconds. Defaults to 5500.
        duration_ms (int, optional): Base crossfade duration in milliseconds. Defaults to 8000.
        eq_match_ms (int, optional): EQ sweep transition duration in milliseconds. Defaults to 15000.

    Returns:
        float: Overlap duration in seconds.
    """
    t = transition_type.lower()  # Convert transition type to lowercase for case-insensitive comparison.
    if "fade in" in t:
        return 0.0  # No overlap for fade-in transitions (first track).
    elif "eq" in t:
        return eq_match_ms / 1000.0  # Convert EQ sweep duration to seconds.
    elif "chorus" in t:
        return (duration_ms + crossfade_early_ms) / 1000.0  # Same as crossfade for chorus transitions.
    else:
        return (duration_ms + crossfade_early_ms) / 1000.0  # Sum crossfade durations and convert to seconds.


# ---------------------------
# Harmonic compatibility
# ---------------------------
def is_harmonic_key(from_key_semitone, to_key_semitone):
    """
    Determines if two tracks are harmonically compatible based on their key semitone indices.

    Process:
    - Checks if the semitone difference (modulo 12) is in a list of compatible shifts (same key, up/down one semitone,
      perfect fifth, or perfect fourth).
    - Returns True if either key is None (no key data) to avoid rejecting transitions unnecessarily.

    Args:
        from_key_semitone (int or None): Semitone index of the first track's key (0-23).
        to_key_semitone (int or None): Semitone index of the second track's key (0-23).

    Returns:
        bool: True if the keys are harmonically compatible or if either key is None, False otherwise.
    """
    compatible_shifts = [0, 1, 11, 7, 5]  # Compatible key shifts: same, ±1 semitone, perfect fifth (7), perfect fourth (5).
    if from_key_semitone is None or to_key_semitone is None:
        return True  # Allow transition if key data is missing.
    key_diff = abs(int(from_key_semitone) - int(to_key_semitone)) % 12  # Compute absolute semitone difference (mod 12).
    return key_diff in compatible_shifts or (12 - key_diff) in compatible_shifts  # Check both forward and backward shifts.


# ---------------------------
# OTAC (tempo adjust) helper
# ---------------------------
def compute_otac(song1_data, song2_data):
    """
    Computes the Optimal Tempo Adjustment Coefficient (OTAC) for transitioning between two tracks.

    Process:
    - OTAC is calculated as log(tempo2 / tempo1) / 60, representing the rate of tempo change per second.
    - Handles edge cases (e.g., zero or negative BPM) by returning 0.0.
    - Ensures floating-point arithmetic for precision.

    Args:
        song1_data (dict): Metadata of the first track, including 'bpm' (beats per minute).
        song2_data (dict): Metadata of the second track, including 'bpm'.

    Returns:
        float: OTAC value (tempo change rate per second), or 0.0 if calculation fails.
    """
    tempo1, tempo2 = song1_data.get('bpm', 0), song2_data.get('bpm', 0)  # Extract BPM, default to 0 if missing.
    try:
        if tempo1 <= 0 or tempo2 <= 0:
            return 0.0  # Return 0.0 for invalid BPM values.
        otac = np.log(float(tempo2) / float(tempo1)) / 60.0  # Compute OTAC using logarithmic ratio.
        return float(otac)  # Ensure float output.
    except Exception:
        return 0.0  # Return 0.0 if calculation fails (e.g., division by zero).


# ---------------------------
# Transition suggestion
# ---------------------------
def suggest_transition_type(from_track, to_track):
    """
    Suggests a transition type for moving from one track to another based on BPM difference, key compatibility, vocal presence, and chorus availability.

    Transition logic:
    - Chorus Beatmatch: BPM difference ≤ 2, both tracks have choruses (for mid-song beat-aligned transitions).
    - Crossfade: Small BPM differences (≤ 3), compatible keys, no vocals in both tracks, or both have vocals.
    - EQ Sweep: Moderate BPM differences (≤ 6) with incompatible keys.
    - Crossfade (default): Used in all other cases for simplicity and safety.

    Args:
        from_track (dict): Metadata of the first track (bpm, key_semitone, has_vocals, choruses).
        to_track (dict): Metadata of the second track (bpm, key_semitone, has_vocals, choruses).

    Returns:
        str: Suggested transition type ('Crossfade', 'EQ Sweep', 'Chorus Beatmatch').
    """
    bpm_diff = abs(float(from_track.get('bpm', 0)) - float(to_track.get('bpm', 0)))  # Compute BPM difference.
    has_choruses_from = bool(from_track.get('choruses', []))  # Check if outgoing track has choruses.
    has_choruses_to = bool(to_track.get('choruses', []))  # Check if incoming track has choruses.
    if bpm_diff <= 2 and has_choruses_from and has_choruses_to:
        return "Chorus Beatmatch"  # Use chorus transition if BPMs align closely and choruses available.
    key_compatible = is_harmonic_key(from_track.get('key_semitone'), to_track.get('key_semitone'))  # Check key compatibility.
    has_vocals = from_track.get('has_vocals', False) and to_track.get('has_vocals', False)  # Check if both tracks have vocals.
    if bpm_diff <= 3 and key_compatible and not has_vocals:
        return "Crossfade"  # Smooth transition for similar tempos and compatible keys without vocals.
    if bpm_diff <= 6 and not key_compatible:
        return "EQ Sweep"  # Use EQ sweep for moderate tempo differences with incompatible keys.
    if has_vocals:
        return "Crossfade"  # Prefer crossfade when both tracks have vocals to avoid clashing.
    return "Crossfade"  # Default to crossfade for all other cases.


# ---------------------------
# Time formatting helper
# ---------------------------
def format_time(seconds: float) -> str:
    """
    Formats seconds into HH:MM:SS string.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Formatted time string (e.g., '00:01:23').
    """
    delta = timedelta(seconds=seconds)
    return (datetime.min + delta).strftime("%H:%M:%S")


# ---------------------------
# Mixing plan generator
# ---------------------------
def generate_mixing_plan(analyzed_setlist_json, first_fade_in_ms=5000, crossfade_early_ms=5500, eq_match_ms=15000):
    """
    Generates a DJ mixing plan from an analyzed setlist, including start times, transition types, cut offsets, and OTAC values.

    Process:
    - Parses the input setlist JSON.
    - For each track, loads the audio to determine duration and calculates start times considering overlaps and cuts.
    - Suggests transition types, including "Chorus Beatmatch" for mid-song transitions at chorus ends.
    - Adjusts outgoing cut points for chorus transitions and computes aligned start times.
    - Handles missing files by using a fallback duration and skipping problematic tracks.
    - Saves the mixing plan as a JSON file ('mixing_plan.json').

    Args:
        analyzed_setlist_json (str): JSON string containing the analyzed setlist with track metadata.
        first_fade_in_ms (int, optional): Fade-in duration for the first track in milliseconds. Defaults to 5000.
        crossfade_early_ms (int, optional): Early crossfade duration in milliseconds. Defaults to 5500.
        eq_match_ms (int, optional): EQ sweep transition duration in milliseconds. Defaults to 15000.

    Raises:
        Exception: If JSON parsing or processing fails, logs the error and re-raises.
    """
    try:
        # Parse the input analyzed setlist JSON into a Python dictionary.
        analyzed_data = json.loads(analyzed_setlist_json)
        mixing_plan = []  # Initialize list to store mixing plan entries.
        last_track_meta = None  # Track the previous track metadata for transition calculations.
        last_start_sec = None  # Track the start time of the last track in mix seconds.
        last_full_duration_sec = 0.0  # Track the full duration of the last track.

        # Iterate over each time segment in the analyzed setlist.
        for segment in analyzed_data.get("analyzed_setlist", []):
            tracks = segment.get("analyzed_tracks", [])  # Get the list of analyzed tracks in the segment.

            # Process each track in the segment.
            for track in tracks:
                file_path = os.path.join(SONGS_DIR, track["file"])  # Construct full file path.
                if not os.path.exists(file_path):
                    # Log missing file and use fallback duration (180 seconds).
                    print(f"[generate_mixing_plan] Missing file: {file_path}. Skipping track.")
                    # For missing files, assume full play and default transition (no cut).
                    if last_track_meta is not None:
                        overlap_sec = get_estimated_overlap("Crossfade", crossfade_early_ms, 8000, eq_match_ms)
                        start_sec = last_start_sec + last_full_duration_sec - overlap_sec
                        mixing_plan.append({
                            "from_track": last_track_meta.get("title"),
                            "to_track": track.get("title"),
                            "start_time": format_time(start_sec),
                            "transition_point": "end of track",
                            "transition_type": "Crossfade",
                            "outgoing_cut_sec": last_full_duration_sec,
                            "overlap_sec": overlap_sec,
                            "otac": 0.0,
                            "comment": f"Transition {last_track_meta.get('title')} -> {track.get('title')}. Missing file fallback."
                        })
                        last_start_sec = start_sec
                    continue  # Skip adding duration update for missing.

                # Load audio file to determine its duration.
                audio = AudioSegment.from_file(file_path)
                duration_sec = len(audio) / 1000.0  # Convert duration to seconds.

                if last_track_meta is None:
                    # First track uses a fade-in transition.
                    start_sec = 0.0
                    transition_type = "Fade In"
                    outgoing_cut_sec = None  # No outgoing for first track.
                    overlap_sec = 0.0
                    from_track_title = None
                    otac_val = 0.0
                    transition_point = "downbeat align"
                    comment = f"Start {track.get('notes','').split('.')[0].lower()} section."
                else:
                    # Suggest transition type based on track metadata.
                    transition_type = suggest_transition_type(last_track_meta, track)
                    otac_val = compute_otac(last_track_meta, track)
                    overlap_sec = get_estimated_overlap(transition_type, crossfade_early_ms, 8000, eq_match_ms)
                    if transition_type == "Chorus Beatmatch":
                        # Use end of last chorus as cut point.
                        choruses = last_track_meta.get("choruses", [])
                        if choruses:
                            outgoing_cut_sec = choruses[-1]["end"]
                            # Ensure enough room for overlap; fallback if not.
                            if outgoing_cut_sec < overlap_sec:
                                outgoing_cut_sec = overlap_sec
                                transition_type = "Crossfade"
                                overlap_sec = get_estimated_overlap(transition_type, crossfade_early_ms, 8000, eq_match_ms)
                                transition_point = "end of track"
                            else:
                                transition_point = "chorus end beat align"
                        else:
                            # Fallback if no choruses.
                            outgoing_cut_sec = last_full_duration_sec
                            transition_type = "Crossfade"
                            overlap_sec = get_estimated_overlap(transition_type, crossfade_early_ms, 8000, eq_match_ms)
                            transition_point = "end of track"
                    else:
                        outgoing_cut_sec = last_full_duration_sec
                        transition_point = "beat grid match"
                    
                    # Calculate start time for the track (when it begins playing in the mix).
                    trans_start_mix_sec = last_start_sec + outgoing_cut_sec
                    start_sec = trans_start_mix_sec - overlap_sec
                    from_track_title = last_track_meta.get('title')
                    comment = f"Transition {from_track_title} -> {track.get('title')}. Suggested '{transition_type}' at {transition_point}."

                # Format start time as string.
                start_str = format_time(start_sec)

                # Add the transition to the mixing plan.
                mixing_plan.append({
                    "from_track": from_track_title,
                    "to_track": track.get("title"),
                    "start_time": start_str,
                    "transition_point": transition_point,
                    "transition_type": transition_type,
                    "outgoing_cut_sec": outgoing_cut_sec,
                    "overlap_sec": overlap_sec,
                    "otac": float(otac_val),
                    "comment": comment
                })

                # Update last track info for next iteration.
                last_start_sec = start_sec
                last_full_duration_sec = duration_sec
                last_track_meta = track

        # Save the mixing plan to a JSON file.
        with open("mixing_plan.json", "w") as f:
            json.dump({"mixing_plan": mixing_plan}, f, indent=2)
        print("Mixing plan saved to 'mixing_plan.json'")

    except Exception as e:
        # Log any errors during processing and re-raise for caller handling.
        print(f"[generate_mixing_plan] Error: {e}")
        raise


# ---------------------------
# Example run
# ---------------------------
if __name__ == "__main__":
    """
    Entry point for testing the mixing plan generator with a sample analyzed setlist.

    Defines a sample analyzed setlist JSON string with one time segment and two tracks.
    Calls generate_mixing_plan() to process the setlist and save the output to 'mixing_plan.json'.
    """
    sample_analyzed_setlist_json = '''
    {
        "analyzed_setlist": [
            {
                "time": "19:00–20:00",
                "analyzed_tracks": [
                    {
                        "title": "Tum Hi Ho",
                        "artist": "Arijit Singh",
                        "file": "Arijit Singh - Tum Hi Ho.mp3",
                        "bpm": 94,
                        "key_semitone": 9,
                        "scale": "major",
                        "genre": "bollywood",
                        "energy": 0.45,
                        "valence": 0.32,
                        "danceability": 0.52,
                        "has_vocals": true,
                        "segments": [{"label": "L"}],
                        "choruses": [],
                        "chroma_matrix": null,
                        "transition": "Fade In",
                        "notes": "Balanced Vibe track. Genre: bollywood."
                    },
                    {
                        "title": "No Scrubs",
                        "artist": "TLC",
                        "file": "TLC - No Scrubs.mp3",
                        "bpm": 93,
                        "key_semitone": 8,
                        "scale": "minor",
                        "genre": "r&b",
                        "energy": 0.7,
                        "valence": 0.6,
                        "danceability": 0.8,
                        "has_vocals": true,
                        "segments": [{"label": "H"}],
                        "choruses": [],
                        "chroma_matrix": null,
                        "transition": "Crossfade",
                        "notes": "Dance Floor Filler track. Genre: r&b."
                    }
                ]
            }
        ]
    }
    '''
    # Process the sample analyzed setlist and generate the mixing plan.
    generate_mixing_plan(sample_analyzed_setlist_json)