# track_analysis_engine.py
"""
This module implements a track analysis engine that processes MP3 files from a setlist to extract comprehensive audio
features using the Librosa library. It analyzes tracks for tempo, key, energy, valence, danceability, and structural
segments, and enhances a setlist with these features, vibe labels, and transition suggestions. The output is saved as a
JSON file for use in DJ or event planning applications.

Key features:
- Extracts audio features such as BPM, key, energy, valence, danceability, and structural segments.
- Uses Krumhansl-Kessler profiles for key estimation and heuristic-based vocal detection.
- Applies beat tracking and optional machine learning for downbeat detection (with fallback logic).
- Suggests DJ transitions between tracks based on BPM and energy differences.
- Sorts tracks within setlist segments by BPM for smoother transitions.

Dependencies:
- librosa: For audio feature extraction (e.g., chroma, MFCC, RMS).
- numpy: For numerical computations on audio features.
- sklearn: For PCA and StandardScaler (used in feature extraction).
- tensorflow: For loading a pre-trained downbeat detection model (optional).
- json, os: For handling JSON data and file operations.
- warnings: To suppress non-critical Numba/Librosa warnings.
"""

import json  # Used for parsing input setlist JSON and serializing output to JSON.
import os  # Used for file path operations and environment variable management.
import warnings  # Used to suppress non-critical warnings from Numba/Librosa during initialization.

# Suppress Numba/Librosa warnings that often occur during library initialization to keep console output clean.
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import librosa  # Core library for audio analysis, providing feature extraction like chroma, MFCC, and beat tracking.
from librosa.feature import rms, spectral_centroid, spectral_contrast, mfcc, chroma_stft  # Specific Librosa feature extraction functions.
import numpy as np  # Used for numerical operations on audio features (e.g., means, correlations).
from sklearn.decomposition import PCA  # Imported but not used in this code; reserved for potential dimensionality reduction.
from sklearn.preprocessing import StandardScaler  # Imported but not used; reserved for potential feature scaling.
import tensorflow as tf  # Used for loading a pre-trained downbeat detection model (if available).

# Suppress TensorFlow oneDNN optimization warning to reduce console clutter.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define the directory path where local MP3 song files are stored (relative to the script's execution directory).
SONGS_DIR = "./songs"

# Attempt to load a pre-trained downbeat detection model from a file named 'downbeat_model.h5'.
# If the file does not exist, set to None to use fallback logic.
downbeat_model = tf.keras.models.load_model('downbeat_model.h5', compile=False) if os.path.exists('downbeat_model.h5') else None


# --- Core Feature Extraction Functions ---

def estimate_key(chroma_mean):
    """
    Estimates the musical key and scale (major/minor) of a track using Krumhansl-Kessler key profiles.

    Process:
    - Uses predefined Krumhansl-Kessler profiles for major and minor keys, which represent expected pitch class distributions.
    - Normalizes the input chroma mean vector and profiles for fair correlation comparison.
    - Computes correlation between the chroma vector and major/minor profiles to determine the scale.
    - Finds the best key match by rolling the chroma vector to test all 12 pitch classes.

    Args:
        chroma_mean (np.ndarray): Mean chroma vector (12 pitch classes) derived from chroma_stft.

    Returns:
        tuple: (key, scale), where key is a string (e.g., 'C', 'C#') and scale is 'major' or 'minor'.
    """
    # Define Krumhansl-Kessler profiles for major and minor keys (weights for each of the 12 pitch classes).
    major_profile = np.array([6.35, 2.26, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles and input chroma vector to sum to 1 for consistent correlation calculations.
    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)
    chroma_mean_norm = chroma_mean / np.sum(chroma_mean)
    
    # Compute correlation between normalized chroma vector and major/minor profiles starting at C.
    major_corr_c = np.corrcoef(chroma_mean_norm, major_profile)[0, 1]
    minor_corr_c = np.corrcoef(chroma_mean_norm, minor_profile)[0, 1]
    
    # Determine scale by comparing major and minor correlations.
    if major_corr_c > minor_corr_c:
        # For major scale, compute correlations for all 12 possible key shifts (C to B).
        corrs = [np.corrcoef(np.roll(chroma_mean_norm, -i), major_profile)[0, 1] for i in range(12)]
        key_idx = np.argmax(corrs)  # Select the key with the highest correlation.
        scale = 'major'
    else:
        # For minor scale, compute correlations for all 12 possible key shifts.
        corrs = [np.corrcoef(np.roll(chroma_mean_norm, -i), minor_profile)[0, 1] for i in range(12)]
        key_idx = np.argmax(corrs)  # Select the key with the highest correlation.
        scale = 'minor'
    
    # Map the key index to a key name (C to B).
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = keys[key_idx]
    
    return key, scale


def robust_tempo(y, sr):
    """
    Estimates the tempo (BPM) of an audio track using Librosa's beat tracking algorithm.

    Process:
    - Uses librosa.beat.beat_track to estimate tempo and beat frames.
    - Handles cases where tempo is returned as an array (e.g., variable tempo) by taking the mean.
    - Rounds the tempo to an integer for simplicity.
    - Includes fallback logic returning 120 BPM if beat tracking fails.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        int: Estimated tempo in beats per minute (BPM).
    """
    try:
        # Perform beat tracking to estimate tempo and beat frames.
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Handle cases where tempo is an array (variable tempo) by taking the mean.
        bpm = round(float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo))
        return bpm
    except Exception as e:
        # Log error and return a fallback tempo of 120 BPM if beat tracking fails.
        print(f"Tempo detection failed: {e}")
        return 120


def _detect_downbeats(y, sr, beats):
    """
    Detects downbeats (first beats of measures) in an audio track using a machine learning model (if available)
    or a fallback heuristic assuming a 4/4 time signature.

    Process:
    - If a pre-trained downbeat model is available, it would be used (placeholder logic).
    - Otherwise, assumes 4/4 time and selects every 4th beat as a downbeat.
    - Converts beat frames to time (seconds) using the sampling rate.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.
        beats (np.ndarray): Beat frame indices from librosa.beat.beat_track.

    Returns:
        np.ndarray: Array of downbeat times in seconds.
    """
    if downbeat_model is None:
        # Fallback to heuristic: assume 4/4 time and select every 4th beat as a downbeat.
        if len(beats) > 0:
            downbeats = beats[::4]  # Select every 4th beat.
        else:
            downbeats = np.array([0])  # Default to time 0 if no beats are detected.
        return librosa.frames_to_time(downbeats, sr=sr)
    
    # Placeholder for machine learning model-based downbeat detection.
    print("Using placeholder logic for ML model.")
    if len(beats) > 0:
        downbeats = beats[::4]  # Fallback to every 4th beat.
    else:
        downbeats = np.array([0])  # Default to time 0 if no beats.
    return librosa.frames_to_time(downbeats, sr=sr)


def _structural_segmentation(y, sr):
    """
    Performs structural segmentation of an audio track based on beat boundaries, labeling segments as high or low energy.

    Process:
    - Uses Librosa's beat tracking to identify segment boundaries.
    - Ensures segments cover the full track duration by adding start (0.0) and end times if needed.
    - Computes RMS energy for each segment and labels it as 'High' or 'Low' based on the median energy.
    - Returns a list of segments with start/end times and energy labels.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        list: List of dictionaries with keys 'start' (float), 'end' (float), and 'label' (str: 'High' or 'Low').
              Returns a fallback segment if analysis fails.
    """
    try:
        # Perform beat tracking to get beat boundaries.
        _, boundaries = librosa.beat.beat_track(y=y, sr=sr)
        segments = []
        duration = len(y) / sr  # Calculate total track duration in seconds.
        
        # Convert beat boundaries from frames to time (seconds).
        boundary_times = librosa.frames_to_time(boundaries, sr=sr).tolist()
        # Ensure the track starts at 0.0 seconds.
        if not boundary_times or boundary_times[0] > 0.1:
            boundary_times.insert(0, 0.0)
        # Ensure the track ends at its full duration.
        if boundary_times[-1] < duration - 0.1:
            boundary_times.append(duration)

        # Create segments between consecutive boundaries.
        for i in range(1, len(boundary_times)):
            start_time = boundary_times[i-1]
            end_time = boundary_times[i]
            
            # Extract audio for the segment to compute energy later.
            start_sample = int(start_time * sr)
            end_sample = int(min(end_time * sr, len(y)))
            seg_audio = y[start_sample:end_sample]
            
            segments.append({'start': round(start_time, 2), 'end': round(end_time, 2), 'audio': seg_audio})
        
        # Post-process segments to assign energy labels.
        if segments:
            # Compute RMS energy for each segment.
            energies = [np.mean(rms(y=seg['audio'])) for seg in segments]
            if energies:
                # Label segments as 'High' or 'Low' based on median energy.
                med = np.median(energies)
                labels = ['High' if e > med else 'Low' for e in energies]
            else:
                labels = ['Low'] * len(segments)
            
            # Create cleaned segment list with only start, end, and label (discard audio data).
            cleaned_segments = []
            for i, seg in enumerate(segments):
                cleaned_segments.append({
                    'start': seg['start'], 
                    'end': seg['end'], 
                    'label': labels[i]
                })
            return cleaned_segments

        # Fallback: return a single segment covering the entire track.
        return [{'start': 0.0, 'end': duration, 'label': 'Full Track'}]

    except Exception as e:
        # Log error and return a fallback segment if analysis fails.
        print(f"Segmentation failed: {e}")
        duration = len(y) / sr
        return [{'start': 0.0, 'end': duration, 'label': 'Full Track'}]


def _extract_similarity_features(y, sr):
    """
    Extracts audio features for similarity matching between tracks.

    Features extracted:
    - Mean RMS energy (overall loudness).
    - Mean spectral centroid (brightness of sound).
    - Mean spectral rolloff (frequency below which most energy lies).
    - Mean zero-crossing rate (noisiness or percussiveness).

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        np.ndarray: Array of four feature values (energy, centroid, rolloff, zero-crossing rate).
    """
    energy = np.mean(rms(y=y))  # Mean RMS energy for loudness.
    spectral_centroid_mean = np.mean(spectral_centroid(y=y, sr=sr))  # Mean spectral centroid for brightness.
    spectral_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))  # Mean spectral rolloff for frequency distribution.
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))  # Mean zero-crossing rate for noisiness.
    return np.array([energy, spectral_centroid_mean, spectral_rolloff_mean, zero_crossing])


def _compute_theme_descriptor(y, sr):
    """
    Computes a compact theme vector for a track using spectral and MFCC features.

    Features extracted:
    - Mean spectral centroid (brightness).
    - Mean spectral rolloff (frequency distribution).
    - Mean onset strength (event density).
    - Mean of 13 MFCC coefficients (timbre).

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        np.ndarray: Array of 16 features (3 spectral + 13 MFCCs).
                  Returns zeros if extraction fails.
    """
    try:
        # Initialize feature list with spectral features.
        features = []
        features.append(np.mean(spectral_centroid(y=y, sr=sr)))  # Brightness.
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))  # Frequency distribution.
        features.append(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))  # Event density.
        
        # Extract 13 MFCC coefficients (timbre) and compute their means.
        mfccs_feat = mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs_feat, axis=1))  # Append mean of each MFCC coefficient.

        # Return features as a numpy array (length: 3 spectral + 13 MFCC = 16).
        return np.array(features)
    except Exception as e:
        # Log error and return a zero-filled array if extraction fails.
        print(f"Theme descriptor failed: {e}")
        return np.zeros(16)


def _key_to_semitone(key, scale):
    """
    Maps a musical key and scale to a semitone index (0-11 for major, 12-23 for minor).

    Args:
        key (str): Musical key (e.g., 'C', 'C#').
        scale (str): Scale type ('major' or 'minor').

    Returns:
        int: Semitone index (0-11 for major, 12-23 for minor).
    """
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = keys.index(key)  # Get index of the key (0-11).
    if scale == 'minor':
        idx += 12  # Add 12 for minor keys to differentiate (12-23).
    return idx


def _detect_vocals(y, sr):
    """
    Detects the presence of vocals in an audio track using a heuristic based on energy in the human vocal frequency range.

    Process:
    - Computes the Short-Time Fourier Transform (STFT) to get frequency content.
    - Analyzes energy in the 200 Hz to 4000 Hz range (typical for human vocals).
    - Applies an empirical threshold to determine vocal presence.

    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of the audio.

    Returns:
        bool: True if vocals are detected, False otherwise.
    """
    S = np.abs(librosa.stft(y))  # Compute magnitude of STFT.
    freqs = librosa.fft_frequencies(sr=sr)  # Get frequency bins.
    # Select frequency indices between 200 Hz (low male voice) and 4000 Hz (high female voice/harmonics).
    mid_idx = np.where((freqs > 200) & (freqs < 4000))[0]
    mid_energy = np.mean(S[mid_idx, :])  # Compute mean energy in vocal range.
    # Use empirical threshold to determine if vocals are present.
    return bool(mid_energy > 0.01)


# --- Main Analysis Function ---

def analyze_track(title, artist, filename):
    """
    Analyzes a local MP3 file to extract comprehensive audio features using Librosa.

    Features extracted:
    - Tempo (BPM), beat positions, downbeats.
    - Key, scale, and semitone index.
    - Energy, valence, danceability (heuristic approximations).
    - Structural segments with energy labels.
    - Similarity features (energy, centroid, rolloff, zero-crossing rate).
    - Chroma matrix (for harmonic content).
    - Vocal presence (heuristic).
    - Theme vector (spectral and MFCC features).

    Args:
        title (str): Song title.
        artist (str): Artist name.
        filename (str): Name of the MP3 file in SONGS_DIR.

    Returns:
        dict: Dictionary containing all extracted features.
              Returns a fallback dictionary with default values if the file is missing or analysis fails.
    """
    file_path = os.path.join(SONGS_DIR, filename)  # Construct full file path.
    
    # Check if the file exists; return a fallback dictionary if not.
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Returning fallback data.")
        return {
            "title": title, "artist": artist, "file": filename, 
            "bpm": 0, "key": "N/A", "key_semitone": 0, "scale": "N/A", 
            "energy": 0.0, "valence": 0.0, "danceability": 0.0,
            "beat_positions": [], "downbeats": [], "segments": [],
            "similarity_features": [0.0] * 4, "chroma_matrix": None, "has_vocals": False, 
            "theme_vector": [0.0] * 5, "genre": "File Missing"
        }
    
    try:
        # Load the audio file with its native sampling rate.
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract core timing and tonal features.
        bpm = robust_tempo(y, sr)  # Estimate tempo.
        _, beats = librosa.beat.beat_track(y=y, sr=sr)  # Get beat frames.
        beat_positions = librosa.frames_to_time(beats, sr=sr)  # Convert beats to time (seconds).
        downbeats = _detect_downbeats(y, sr, beats)  # Detect downbeats.

        # Compute chroma features and estimate key.
        chroma_mat = chroma_stft(y=y, sr=sr)  # Chromagram for harmonic content.
        chroma_mean = np.mean(chroma_mat, axis=1)  # Mean chroma for key estimation.
        key, scale = estimate_key(chroma_mean)  # Estimate key and scale.
        key_semitone = _key_to_semitone(key, scale)  # Map to semitone index.
        key_name = f"{key}m" if scale == 'minor' else key  # Format key name (e.g., 'Cm' or 'C').

        # Compute energy and vibe heuristics.
        energy_rms = np.mean(rms(y=y))  # Mean RMS energy for loudness.
        energy = round(float(energy_rms), 2)  # Round for cleaner output.
        centroid = np.mean(spectral_centroid(y=y, sr=sr))  # Mean spectral centroid for brightness.
        valence = round(float(min(centroid / 5000, 1)), 2)  # Approximate valence (mood) based on centroid.
        contrast_mean = np.mean(spectral_contrast(y=y, sr=sr))  # Mean spectral contrast for dynamics.
        danceability = round(float(min(contrast_mean / 40, 1)), 2)  # Approximate danceability based on contrast.

        # Extract advanced structural and similarity features.
        segments = _structural_segmentation(y, sr)  # Segment track and label energy.
        similarity_features = _extract_similarity_features(y, sr)  # Features for track similarity.
        has_vocals = _detect_vocals(y, sr)  # Detect vocal presence.
        theme_vector = _compute_theme_descriptor(y, sr)  # Compact theme vector.
        
        # Return a dictionary with all extracted features.
        return {
            "title": title, "artist": artist, "file": filename, 
            "bpm": round(bpm), "beat_positions": beat_positions.tolist(), "downbeats": downbeats.tolist(),
            "key": key_name, "key_semitone": key_semitone, "scale": scale, "genre": "Unknown",
            "energy": energy, "valence": valence, "danceability": danceability,
            "segments": segments,
            "similarity_features": similarity_features.tolist(),
            "chroma_matrix": chroma_mat.tolist(),  # Full chroma matrix (large).
            "has_vocals": bool(has_vocals),
            "theme_vector": theme_vector.tolist()
        }
    
    except Exception as e:
        # Log error and return a fallback dictionary if analysis fails (e.g., corrupted file).
        print(f"Error analyzing '{title}' by '{artist}': {e}")
        return {
            "title": title, "artist": artist, "file": filename, 
            "bpm": 120, "key": "C", "key_semitone": 0, "scale": "major", 
            "energy": 0.5, "valence": 0.5, "danceability": 0.5,
            "beat_positions": [], "downbeats": [], "segments": [],
            "similarity_features": [0.5, 2000, 4000, 0.1], "chroma_matrix": None, "has_vocals": False, 
            "theme_vector": [0.5] * 5, "genre": "Analysis Failed"
        }


# --- Transition and Vibe Logic ---

def compute_vibe_label(energy, valence, danceability):
    """
    Computes a contextual vibe label for a track based on its energy, valence, and danceability.

    Labels:
    - Peak Energy: High energy and danceability.
    - Sunset Chill: Low energy, high valence (relaxed mood).
    - Dance Floor Filler: High danceability, positive valence.
    - Intense Build: Moderate energy, low valence (dramatic).
    - Balanced Vibe: Default for other cases.

    Args:
        energy (float): Track energy (0 to 1).
        valence (float): Track valence (mood, 0 to 1).
        danceability (float): Track danceability (0 to 1).

    Returns:
        str: Vibe label describing the track's mood or purpose.
    """
    if energy > 0.2 and danceability > 0.7:
        return "Peak Energy"
    elif energy < 0.1 and valence > 0.6:
        return "Sunset Chill"
    elif danceability > 0.8 and valence > 0.5:
        return "Dance Floor Filler"
    elif energy > 0.15 and valence < 0.4:
        return "Intense Build"
    return "Balanced Vibe"


def suggest_transition(prev_track_data, current_track_data):
    """
    Suggests a DJ transition type based on the BPM and energy differences between consecutive tracks.

    Transition types:
    - Seamless Beatmatch: BPM difference ≤ 2.
    - Pitch Bend Adjustment: BPM difference ≤ 5.
    - Energy Build (EQ Sweep): Current track has significantly higher energy.
    - Fade Out Transition: Current track has significantly lower energy.
    - Crossfade: Default for other cases.

    Args:
        prev_track_data (dict or None): Metadata of the previous track (bpm, energy).
        current_track_data (dict): Metadata of the current track (bpm, energy).

    Returns:
        str: Suggested transition type.
    """
    if not prev_track_data:
        return "Fade In"  # First track in the setlist uses a fade-in.
    
    # Extract BPM and energy, using defaults if missing.
    prev_bpm = prev_track_data.get('bpm', 120)
    curr_bpm = current_track_data.get('bpm', 120)
    prev_energy = prev_track_data.get('energy', 0.15)
    curr_energy = current_track_data.get('energy', 0.15)
    bpm_diff = abs(prev_bpm - curr_bpm)
    
    # Suggest transition based on BPM and energy differences.
    if bpm_diff <= 2:
        return "Seamless Beatmatch"
    elif bpm_diff <= 5:
        return "Pitch Bend Adjustment"
    elif curr_energy > prev_energy + 0.05:
        return "Energy Build (EQ Sweep)"
    elif curr_energy < prev_energy - 0.05:
        return "Fade Out Transition"
    return "Crossfade"


def generate_notes(vibe_label, energy, danceability, genre):
    """
    Generates descriptive notes for a track based on its vibe label, energy, danceability, and genre.

    Args:
        vibe_label (str): Computed vibe label (e.g., 'Peak Energy').
        energy (float): Track energy (0 to 1).
        danceability (float): Track danceability (0 to 1).
        genre (str): Track genre (may be 'Unknown').

    Returns:
        str: Descriptive notes for the track, useful for DJs or event planners.
    """
    notes = f"{vibe_label} track. "  # Start with the vibe label.
    if energy > 0.2:
        notes += "High energy suitable for peak moments. "  # Highlight high-energy tracks.
    if danceability > 0.8:
        notes += "Excellent for dancing. "  # Highlight danceable tracks.
    notes += f"Genre: {genre}."  # Append genre information.
    return notes


# --- Setlist Analysis and Output ---

def analyze_tracks_in_setlist(setlist_json):
    """
    Analyzes all tracks in a setlist, adds advanced features (e.g., vibe labels, transitions), and sorts tracks by BPM
    within each time segment for smoother transitions.

    Process:
    - Parses the input setlist JSON.
    - Analyzes each track in each time segment using analyze_track().
    - Computes vibe labels and transition suggestions.
    - Sorts tracks within segments by BPM.
    - Saves the analyzed setlist to 'analyzed_setlist.json'.

    Args:
        setlist_json (str): JSON string containing the setlist with time segments and tracks.

    Returns:
        dict: Analyzed setlist with enriched track metadata.

    Raises:
        Exception: If JSON parsing or analysis fails, logs the error and re-raises.
    """
    try:
        # Parse the input setlist JSON string into a Python dictionary.
        setlist_data = json.loads(setlist_json)
        analyzed_setlist = []
        
        # Track the previous track's metadata globally for transition suggestions across segments.
        global_prev_track_data = None
        
        # Iterate over each time segment in the setlist.
        for segment in setlist_data["setlist"]:
            time_range = segment["time"]  # Get the time range (e.g., '19:00–20:00').
            tracks = segment["tracks"]  # Get the list of tracks in the segment.
            temp_analyzed = []  # Temporary list to store analyzed tracks before sorting.
            
            # Analyze each track in the segment.
            for track in tracks:
                title = track["title"]
                artist = track["artist"]
                filename = track["file"]
                
                # Analyze the track to extract audio features.
                metadata = analyze_track(title, artist, filename)
                
                # Compute a vibe label based on track features.
                vibe_label = compute_vibe_label(
                    metadata.get("energy", 0.5),  # Use defaults to handle missing files.
                    metadata.get("valence", 0.5), 
                    metadata.get("danceability", 0.5)
                )
                # Generate descriptive notes for the track.
                notes = generate_notes(vibe_label, metadata.get("energy", 0.5), metadata.get("danceability", 0.5), metadata.get("genre", "Unknown"))
                
                # Create an analyzed track dictionary with metadata and notes.
                analyzed_track = {
                    **metadata,
                    "notes": notes,
                    "transition": None  # Transition will be set after sorting.
                }
                temp_analyzed.append(analyzed_track)
            
            # Sort tracks in the segment by BPM for smoother transitions.
            temp_analyzed.sort(key=lambda x: x['bpm'])
            
            # Assign transitions to sorted tracks.
            analyzed_tracks = []
            for analyzed_track in temp_analyzed:
                metadata_for_transition = {
                    "bpm": analyzed_track["bpm"],
                    "energy": analyzed_track["energy"]
                }
                
                # Suggest a transition based on the previous track (global or within segment).
                transition = suggest_transition(global_prev_track_data, metadata_for_transition)
                analyzed_track["transition"] = transition
                analyzed_tracks.append(analyzed_track)
                global_prev_track_data = metadata_for_transition  # Update global previous track.
            
            # Add the analyzed segment to the setlist.
            analyzed_setlist.append({
                "time": time_range,
                "analyzed_tracks": analyzed_tracks
            })
        
        # Create the final output dictionary.
        output = {"analyzed_setlist": analyzed_setlist}
        # Save the analyzed setlist to a JSON file.
        with open("analyzed_setlist.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Analyzed setlist saved to 'analyzed_setlist.json'")
        
        return output
    except Exception as e:
        # Log any errors during analysis and re-raise for caller handling.
        print(f"Error in Track Analysis Engine: {str(e)}")
        raise


if __name__ == "__main__":
    """
    Entry point for testing the track analysis engine with a sample setlist.

    Defines a sample setlist JSON string with two time segments and four tracks.
    Calls analyze_tracks_in_setlist() to process the setlist and save the output.
    """
    sample_setlist_json = '''
    {
        "setlist": [
            {
                "time": "19:00–20:00",
                "tracks": [
                    {"title": "Tum Hi Ho", "artist": "Arijit Singh", "file": "Arijit Singh - Tum Hi Ho.mp3"},
                    {"title": "No Scrubs", "artist": "TLC", "file": "TLC - No Scrubs.mp3"}
                ]
            },
            {
                "time": "20:00–21:00",
                "tracks": [
                    {"title": "Ye", "artist": "Burna Boy", "file": "Burna Boy - Ye.mp3"},
                    {"title": "Essence", "artist": "Wizkid ft. Tems", "file": "Wizkid ft. Tems - Essence.mp3"}
                ]
            }
        ]
    }
    '''
    # Process the sample setlist and generate the analyzed output.
    analyze_tracks_in_setlist(sample_setlist_json)