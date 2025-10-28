# generate_mix.py
"""
This module generates a continuous DJ mix by applying transitions between tracks based on an analyzed setlist and a mixing plan.
It uses audio processing techniques to align tracks, apply tempo adjustments, and create smooth transitions (e.g., crossfades, EQ sweeps).
The resulting mix is normalized and exported as an MP3 file.

Key features:
- Converts audio between pydub's AudioSegment and NumPy arrays for processing.
- Aligns tracks using onset strength correlation for beat matching.
- Supports multiple transition types (e.g., crossfade, EQ sweep, echo-drop, reverb) with customizable parameters.
- Applies tempo stretching based on OTAC (Optimal Tempo Adjustment Coefficient) from the mixing plan.
- Handles missing files gracefully and normalizes the final mix for consistent loudness.

Dependencies:
- os: For file path operations.
- json: For parsing input setlist and mixing plan JSON files.
- numpy: For numerical operations on audio data.
- librosa: For audio feature extraction (e.g., onset strength).
- pydub: For audio manipulation (e.g., fades, filters, normalization).
- pydub.effects: For applying audio effects like high-pass and low-pass filters.
"""

import os  # Used for constructing file paths to access MP3 files.
import json  # Used for parsing input setlist and mixing plan JSON files.
import numpy as np  # Used for numerical operations on audio data (e.g., array conversions, correlations).
import librosa  # Used for audio feature extraction, such as onset strength for beat alignment.
from pydub import AudioSegment  # Used for loading, manipulating, and exporting audio files.
from pydub.effects import high_pass_filter, low_pass_filter, normalize  # Used for audio effects and normalization.

# Define the directory path where local MP3 song files are stored (relative to the script's execution directory).
SONGS_DIR = "./songs"


# ---------------------------
# Utility conversions
# ---------------------------
def audio_segment_to_np(segment: AudioSegment):
    """
    Converts a pydub AudioSegment to a mono float32 NumPy array with values in [-1, 1].

    Process:
    - Extracts raw sample data from the AudioSegment.
    - If stereo, averages the two channels to create a mono signal.
    - Normalizes the samples to the range [-1, 1] for compatibility with audio processing libraries.

    Args:
        segment (AudioSegment): Input audio segment (mono or stereo).

    Returns:
        tuple: (y, sr), where y is a NumPy array (mono, float32, [-1, 1]) and sr is the sampling rate (int).
    """
    samples = np.array(segment.get_array_of_samples())  # Get raw sample data as a NumPy array.
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)  # Average stereo channels to mono.
    sr = segment.frame_rate  # Extract sampling rate.
    y = samples.astype(np.float32) / 32768.0  # Convert to float32 and normalize to [-1, 1].
    return y, sr


def np_to_audio_segment(y: np.ndarray, sr: int):
    """
    Converts a mono float32 NumPy array in [-1, 1] to a pydub AudioSegment (mono, 16-bit).

    Process:
    - Clips the input array to ensure values stay within [-1, 1].
    - Converts to 16-bit integers for compatibility with AudioSegment.
    - Creates a mono AudioSegment with the specified sampling rate.

    Args:
        y (np.ndarray): Mono audio array (float32, [-1, 1]).
        sr (int): Sampling rate of the audio.

    Returns:
        AudioSegment: Mono, 16-bit audio segment.
    """
    y_clipped = np.clip(y, -1.0, 1.0)  # Ensure values are within [-1, 1] to avoid distortion.
    y_int16 = (y_clipped * 32767.0).astype(np.int16)  # Convert to 16-bit integers.
    return AudioSegment(
        y_int16.tobytes(),  # Convert samples to bytes.
        frame_rate=sr,  # Set sampling rate.
        sample_width=2,  # Use 16-bit samples (2 bytes).
        channels=1  # Mono audio.
    )


# ---------------------------
# Beat / onset helpers
# ---------------------------
def get_onset_envelope(y, sr, hop_length=512):
    """
    Computes the normalized onset strength envelope of an audio signal using Librosa.

    Process:
    - Uses librosa.onset.onset_strength to calculate onset strength (indicating potential beat or event locations).
    - Normalizes the envelope to [0, 1] to ensure consistent correlation calculations.
    - Returns None if the envelope is empty to handle edge cases.

    Args:
        y (np.ndarray): Mono audio time series (float32, [-1, 1]).
        sr (int): Sampling rate of the audio.
        hop_length (int, optional): Hop length for onset detection. Defaults to 512.

    Returns:
        tuple: (onset_env, hop_length), where onset_env is a normalized NumPy array or None, and hop_length is the hop size.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)  # Compute onset strength envelope.
    if onset_env.size == 0:
        return None, hop_length  # Return None for empty envelope (e.g., silent audio).
    onset_env = onset_env / (np.max(onset_env) + 1e-9)  # Normalize to [0, 1], avoiding division by zero.
    return onset_env, hop_length


def find_best_alignment(y1, sr1, y2, sr2, match_duration_sec=15.0):
    """
    Finds the best alignment (lag in seconds) between two audio tracks using onset strength correlation.

    Process:
    - Extracts the last match_duration_sec of the first track and the first match_duration_sec of the second track.
    - Computes normalized onset strength envelopes for both segments.
    - Uses cross-correlation to find the lag that maximizes alignment of onsets (e.g., beats).
    - Converts the lag to seconds, clipping to the match duration to avoid extreme offsets.

    Args:
        y1 (np.ndarray): Audio time series of the first track.
        sr1 (int): Sampling rate of the first track.
        y2 (np.ndarray): Audio time series of the second track.
        sr2 (int): Sampling rate of the second track.
        match_duration_sec (float, optional): Duration (in seconds) to analyze for alignment. Defaults to 15.0.

    Returns:
        float: Lag in seconds (positive for delay, negative for advance) for optimal alignment, or 0.0 if alignment fails.
    """
    try:
        # Limit analysis to match_duration_sec samples to reduce computation.
        n1 = min(len(y1), int(match_duration_sec * sr1))
        n2 = min(len(y2), int(match_duration_sec * sr2))
        if n1 < 1024 or n2 < 1024:
            return 0.0  # Return 0.0 if segments are too short for reliable alignment.
        tail1 = y1[-n1:]  # Last portion of the first track.
        head2 = y2[:n2]  # First portion of the second track.
        onset1, hop = get_onset_envelope(tail1, sr1)  # Onset envelope for first track.
        onset2, _ = get_onset_envelope(head2, sr2)  # Onset envelope for second track.
        if onset1 is None or onset2 is None:
            return 0.0  # Return 0.0 if onset detection fails.
        minlen = min(len(onset1), len(onset2))  # Use shortest envelope length.
        if minlen < 8:
            return 0.0  # Return 0.0 if envelopes are too short for correlation.
        onset1_r = librosa.util.fix_length(onset1, size=minlen)  # Resize to common length.
        onset2_r = librosa.util.fix_length(onset2, size=minlen)
        # Compute cross-correlation of mean-subtracted onset envelopes.
        corr = np.correlate(onset1_r - onset1_r.mean(), onset2_r - onset2_r.mean(), mode='full')
        lag_idx = corr.argmax() - (len(onset2_r) - 1)  # Find lag with maximum correlation.
        approx_hop_seconds = hop / float(sr1)  # Convert hop size to seconds.
        lag_seconds = -lag_idx * approx_hop_seconds  # Convert lag index to seconds.
        # Clip lag to avoid extreme offsets.
        lag_seconds = float(np.clip(lag_seconds, -match_duration_sec, match_duration_sec))
        return lag_seconds
    except Exception:
        return 0.0  # Return 0.0 if alignment calculation fails.


# ---------------------------
# Core transition application
# ---------------------------
def apply_transition(segment1: AudioSegment,
                     segment2: AudioSegment,
                     transition_type: str,
                     duration_ms: int = 8000,
                     early_ms: int = 5500,
                     otac: float = 0.0,
                     eq_match_duration_ms: int = 15000):
    """
    Applies a specified transition between two audio segments, incorporating tempo stretching and beat alignment.

    Supported transition types:
    - Crossfade: Fades out the first segment and fades in the second over the overlap duration.
    - EQ Sweep: Applies high-pass to outgoing and low-pass to incoming, with beat alignment.
    - Echo-Drop: Reduces outgoing volume and crossfades with incoming.
    - Fade Out/Fade In: Sequential fade-out and fade-in without overlap.
    - Build Drop: Overlays segments without fading.
    - Loop: Repeats a portion of the outgoing segment before appending the incoming.
    - Backspin: Reverses a portion of the outgoing segment.
    - Reverb: Adds reverb-like delay to both segments during overlap.
    - Default: Appends segments without transition.

    Args:
        segment1 (AudioSegment): Outgoing audio segment.
        segment2 (AudioSegment): Incoming audio segment.
        transition_type (str): Type of transition to apply.
        duration_ms (int, optional): Base transition duration in milliseconds. Defaults to 8000.
        early_ms (int, optional): Early crossfade duration in milliseconds. Defaults to 5500.
        otac (float, optional): Optimal Tempo Adjustment Coefficient for tempo stretching. Defaults to 0.0.
        eq_match_duration_ms (int, optional): Duration for EQ sweep transitions in milliseconds. Defaults to 15000.

    Returns:
        AudioSegment: Resulting audio segment with the applied transition.

    Raises:
        Exception: Logs and returns concatenated segments if transition fails.
    """
    try:
        # Convert audio segments to NumPy arrays for processing.
        y2, sr2 = audio_segment_to_np(segment2)
        y1, sr1 = audio_segment_to_np(segment1)
        
        # Apply tempo stretching to the incoming track based on OTAC.
        if abs(otac) > 0.01:
            rate = 1.0 + otac * (max(duration_ms, eq_match_duration_ms) / 1000.0) / 60.0  # Calculate stretch rate.
            y2_stretched = librosa.effects.time_stretch(y2, rate=rate)  # Stretch the incoming track.
        else:
            y2_stretched = y2  # No stretching if OTAC is small.
        segment2_stretched = np_to_audio_segment(y2_stretched, sr2)  # Convert back to AudioSegment.
        # Ensure consistent frame rate and mono channel for compatibility.
        if segment2_stretched.frame_rate != segment1.frame_rate:
            segment2_stretched = segment2_stretched.set_frame_rate(segment1.frame_rate)
        if segment2_stretched.channels != 1:
            segment2_stretched = segment2_stretched.set_channels(1)
        # Calculate overlap duration, ensuring a minimum of 500ms.
        overlap = int(min(len(segment1), len(segment2_stretched), duration_ms + early_ms))
        overlap = max(500, overlap)

        if transition_type.lower() in ("crossfade", "cross fade"):
            # Crossfade: Fade out the outgoing segment and fade in the incoming segment.
            out_tail = segment1[-overlap:].fade_out(overlap)
            in_head = segment2_stretched[:overlap].fade_in(overlap)
            cross = out_tail.overlay(in_head)  # Combine faded segments.
            return segment1[:-overlap] + cross + segment2_stretched[overlap:]

        elif transition_type.lower() in ("eq sweep", "eq", "eq_sweep"):
            # EQ Sweep: Apply high-pass to outgoing and low-pass to incoming with beat alignment.
            eq_overlap = int(min(eq_match_duration_ms, len(segment1), len(segment2_stretched)))
            eq_overlap = max(2000, eq_overlap)  # Ensure minimum overlap of 2 seconds.
            y1_full, sr_full = audio_segment_to_np(segment1)
            y2_full, _ = audio_segment_to_np(segment2_stretched)
            match_sec = eq_overlap / float(sr_full)
            
            # Compute beat alignment lag for better synchronization.
            lag_seconds = find_best_alignment(y1_full, sr_full, y2_full, sr_full, match_duration_sec=match_sec)
            lag_ms = int(lag_seconds * 1000.0)
            outgoing_tail = segment1[-eq_overlap:]
            outgoing_hp = high_pass_filter(outgoing_tail, cutoff=200)  # High-pass filter outgoing audio.
            outgoing_faded = outgoing_hp.fade_out(eq_overlap)  # Fade out the filtered audio.
            head_for_filter = segment2_stretched[:eq_overlap]
            incoming_lp = low_pass_filter(head_for_filter, cutoff=6000)  # Low-pass filter incoming audio.
            incoming_faded = incoming_lp.fade_in(eq_overlap)  # Fade in the filtered audio.

            if lag_ms > 0:
                # Delay incoming track: Prepend silence to align.
                silence = AudioSegment.silent(duration=lag_ms)
                incoming_head_faded = silence + incoming_faded
                mixed = outgoing_faded.overlay(incoming_head_faded)
                tail = segment2_stretched[eq_overlap:]
            elif lag_ms < 0:
                # Advance incoming track: Prepend earlier portion and align overlap.
                advance_ms = -lag_ms
                pre_head = low_pass_filter(segment2_stretched[:advance_ms], cutoff=6000)
                overlay_head = low_pass_filter(segment2_stretched[advance_ms : advance_ms + eq_overlap], cutoff=6000)
                overlay_faded = overlay_head.fade_in(eq_overlap)
                mixed = outgoing_faded.overlay(overlay_faded, position=0)
                incoming_head_faded = pre_head + mixed
                tail = segment2_stretched[advance_ms + eq_overlap:]
            else:
                # No lag: Directly overlay faded segments.
                incoming_head_faded = incoming_faded
                mixed = outgoing_faded.overlay(incoming_head_faded)
                tail = segment2_stretched[eq_overlap:]

            return segment1[:-eq_overlap] + mixed + tail

        elif transition_type.lower() in ("echo-drop", "echo drop"):
            # Echo-Drop: Reduce outgoing volume and crossfade with incoming.
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            echo = (segment1[-overlap:] - 10).fade_out(int(overlap * 0.6))  # Lower volume and fade out.
            incoming = segment2_stretched[:overlap].fade_in(int(overlap * 0.6))  # Fade in incoming.
            return segment1[:-overlap] + echo.overlay(incoming) + segment2_stretched[overlap:]

        elif transition_type.lower() in ("fade out/fade in", "fade out", "fade in", "fade_in"):
            # Fade Out/Fade In: Sequential fade-out and fade-in without overlap.
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            fade_out = segment1[-overlap:].fade_out(overlap)
            fade_in = segment2_stretched[:overlap].fade_in(overlap)
            return segment1[:-overlap] + fade_out + fade_in + segment2_stretched[overlap:]

        elif transition_type.lower() == "build_drop":
            # Build Drop: Overlay segments without fading for a dramatic effect.
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            mixed_overlap = segment1[-overlap:].overlay(segment2_stretched[:overlap])
            return segment1[:-overlap] + mixed_overlap + segment2_stretched[overlap:]

        elif transition_type.lower() == "loop":
            # Loop: Repeat a portion of the outgoing segment before appending the incoming.
            beat_len_ms = 2000  # Default loop duration (2 seconds).
            loop_end = min(beat_len_ms, len(segment1))
            loop_seg = segment1[-loop_end:] + segment1[-loop_end:]  # Repeat the last portion.
            return loop_seg + segment2_stretched

        elif transition_type.lower() == "backspin":
            # Backspin: Reverse a portion of the outgoing segment for a DJ rewind effect.
            rewind_len_ms = int(min(4000, len(segment1)))  # Default rewind duration (4 seconds).
            rewind = segment1[-rewind_len_ms:].reverse().fade_out(1000)  # Reverse and fade out.
            return rewind + segment2_stretched

        elif transition_type.lower() == "reverb":
            # Reverb: Simulate reverb by overlaying delayed copies of both segments.
            overlap = int(min(duration_ms, len(segment1), len(segment2_stretched)))
            delay_ms = 100  # Short delay for reverb effect.
            reverb_out = segment1[-overlap:] + AudioSegment.silent(duration=delay_ms)
            reverb_out = reverb_out.overlay(segment1[-overlap:].shift(delay_ms), gain=-10)  # Add delayed copy.
            reverb_in = segment2_stretched[:overlap] + AudioSegment.silent(duration=delay_ms)
            reverb_in = reverb_in.overlay(segment2_stretched[:overlap].shift(delay_ms), gain=-10)  # Add delayed copy.
            crossfade = reverb_out.overlay(reverb_in)  # Combine reverb effects.
            return segment1[:-overlap] + crossfade + segment2_stretched[overlap:]

        else:
            # Default: Concatenate segments without transition.
            return segment1 + segment2_stretched

    except Exception as e:
        # Log error and return concatenated segments as a fallback.
        print(f"[apply_transition] Exception: {e}")
        return segment1 + segment2


# ---------------------------
# Mix generator
# ---------------------------
def generate_mix(analyzed_setlist_json, mixing_plan_json, first_fade_in_ms=5000, crossfade_early_ms=5500, eq_match_ms=15000):
    """
    Generates a continuous DJ mix by processing an analyzed setlist and applying transitions from a mixing plan.

    Process:
    - Loads the analyzed setlist and mixing plan JSON files.
    - Iterates through tracks, applying transitions as specified in the mixing plan.
    - Handles missing files by skipping them and logging errors.
    - Applies a fade-in to the first track and normalizes the final mix for consistent loudness.
    - Exports the mix as an MP3 file ('mix.mp3').

    Args:
        analyzed_setlist_json (str): JSON string containing the analyzed setlist with track metadata.
        mixing_plan_json (str): Path to the JSON file containing the mixing plan.
        first_fade_in_ms (int, optional): Fade-in duration for the first track in milliseconds. Defaults to 5000.
        crossfade_early_ms (int, optional): Early crossfade duration in milliseconds. Defaults to 5500.
        eq_match_ms (int, optional): EQ sweep transition duration in milliseconds. Defaults to 15000.

    Raises:
        Exception: Logs and re-raises any errors during processing.
    """
    try:
        # Parse the analyzed setlist JSON string.
        analyzed_data = json.loads(analyzed_setlist_json)
        # Load the mixing plan from the specified JSON file.
        mixing_plan = json.load(open(mixing_plan_json, 'r')).get("mixing_plan", [])
        full_mix = AudioSegment.empty()  # Initialize an empty AudioSegment for the mix.
        track_index = 0  # Track index to align with mixing plan entries.

        # Iterate over each time segment in the analyzed setlist.
        for segment in analyzed_data.get("analyzed_setlist", []):
            tracks = segment.get("analyzed_tracks", [])  # Get the list of analyzed tracks.

            # Process each track in the segment.
            for track in tracks:
                file_path = os.path.join(SONGS_DIR, track["file"])  # Construct full file path.
                if not os.path.exists(file_path):
                    # Log missing file and skip the track.
                    print(f"[generate_mix] Missing file: {file_path}. Skipping track.")
                    track_index += 1
                    continue

                # Load the audio file.
                audio = AudioSegment.from_file(file_path)
                if track_index >= len(mixing_plan):
                    # If mixing plan is too short, append the track without transition.
                    print(f"[generate_mix] Mixing plan too short, appending track {track['title']}.")
                    full_mix += audio
                    track_index += 1
                    continue

                # Get the corresponding mixing plan entry.
                plan_entry = mixing_plan[track_index]
                transition_type = plan_entry.get("transition_type", "Crossfade")  # Default to Crossfade.
                otac = plan_entry.get("otac", 0.0)  # Get OTAC for tempo adjustment.

                if len(full_mix) == 0:
                    # First track: Apply fade-in.
                    fade_dur = int(min(first_fade_in_ms, len(audio)))
                    full_mix += audio.fade_in(fade_dur)
                else:
                    # Subsequent tracks: Apply transition.
                    desired_overlap_ms = max(eq_match_ms if "eq" in transition_type.lower() else 8000, 8000)
                    available = len(full_mix)
                    overlap_chunk_ms = int(min(available, desired_overlap_ms))  # Determine overlap duration.
                    if overlap_chunk_ms < 1000:
                        overlap_chunk_ms = int(min(5000, available))  # Use minimum overlap if too short.
                    tail_chunk = full_mix[-overlap_chunk_ms:]  # Extract tail of current mix.
                    # Apply the specified transition.
                    trans_audio = apply_transition(tail_chunk, audio, transition_type,
                                                  duration_ms=8000,
                                                  early_ms=crossfade_early_ms,
                                                  otac=otac,
                                                  eq_match_duration_ms=eq_match_ms)
                    full_mix = full_mix[:-overlap_chunk_ms] + trans_audio  # Replace tail with transitioned audio.

                track_index += 1

        # Normalize the final mix for consistent loudness and export as MP3.
        full_mix = normalize(full_mix)
        full_mix.export("mix.mp3", format="mp3")
        print("Mix exported to 'mix.mp3'")

    except Exception as e:
        # Log any errors and re-raise for caller handling.
        print(f"[generate_mix] Error: {e}")
        raise


# ---------------------------
# Example run
# ---------------------------
if __name__ == "__main__":
    """
    Entry point for testing the mix generator with a sample analyzed setlist and mixing plan.

    Defines a sample analyzed setlist JSON string with one time segment and two tracks.
    Assumes 'mixing_plan.json' exists (generated by generate_mixing_plan.py).
    Calls generate_mix() to create and export the mix as 'mix.mp3'.
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
                        "chroma_matrix": null,
                        "transition": "Crossfade",
                        "notes": "Dance Floor Filler track. Genre: r&b."
                    }
                ]
            }
        ]
    }
    '''
    # Process the sample setlist and mixing plan to generate the mix.
    generate_mix(sample_analyzed_setlist_json, "mixing_plan.json")