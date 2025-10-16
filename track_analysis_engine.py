import json
import os
import warnings
# Suppress Numba/Librosa warnings that often occur during initialization
warnings.filterwarnings("ignore", category=UserWarning, module="numba")

import librosa
from librosa.feature import rms, spectral_centroid, spectral_contrast, mfcc, chroma_stft
import numpy as np
# from scipy.spatial.distance import cosine # Removed unused import
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf # Placeholder for pre-trained models

# Suppress TensorFlow oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SONGS_DIR = "./songs"

# Placeholder for pre-trained models (Will likely be None in this environment)
downbeat_model = tf.keras.models.load_model('downbeat_model.h5', compile=False) if os.path.exists('downbeat_model.h5') else None

# --- Core Feature Extraction Functions ---

def estimate_key(chroma_mean):
    """Estimate key and scale using Krumhansl-Kessler key profiles."""
    # Krumhansl-Kessler profiles for major and minor keys
    major_profile = np.array([6.35, 2.26, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles and mean chroma vector
    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)
    chroma_mean_norm = chroma_mean / np.sum(chroma_mean)
    
    # Correlate normalized mean chroma against the major profile starting at C
    major_corr_c = np.corrcoef(chroma_mean_norm, major_profile)[0, 1]
    # Correlate normalized mean chroma against the minor profile starting at C
    minor_corr_c = np.corrcoef(chroma_mean_norm, minor_profile)[0, 1]
    
    if major_corr_c > minor_corr_c:
        # Find best major key match by rolling the chroma vector
        corrs = [np.corrcoef(np.roll(chroma_mean_norm, -i), major_profile)[0, 1] for i in range(12)]
        key_idx = np.argmax(corrs)
        scale = 'major'
    else:
        # Find best minor key match by rolling the chroma vector
        corrs = [np.corrcoef(np.roll(chroma_mean_norm, -i), minor_profile)[0, 1] for i in range(12)]
        key_idx = np.argmax(corrs)
        scale = 'minor'
    
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = keys[key_idx]
    
    return key, scale

def robust_tempo(y, sr):
    """Tempo estimation using librosa's robust beat tracking."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(np.mean(tempo)) if isinstance(tempo, np.ndarray) else float(tempo))
        return bpm
    except Exception as e:
        print(f"Tempo detection failed: {e}")
        return 120 # Fallback tempo

def _detect_downbeats(y, sr, beats):
    """Downbeat detection using ML model (if available) or 4-beat fallback."""
    if downbeat_model is None:
        # Fallback to 4-beat grouping (assuming 4/4 time signature)
        if len(beats) > 0:
            downbeats = beats[::4]
        else:
            downbeats = np.array([0])
        return librosa.frames_to_time(downbeats, sr=sr)
    
    # --- Placeholder ML logic ---
    # In a real implementation, you would extract features suitable for the model
    # and use downbeat_model.predict(features)
    print("Using placeholder logic for ML model.")
    if len(beats) > 0:
        downbeats = beats[::4]
    else:
        downbeats = np.array([0])
    return librosa.frames_to_time(downbeats, sr=sr)


def _structural_segmentation(y, sr):
    """
    Structural segmentation using self-similarity and novelty.
    Note: This uses beat tracking boundaries as a starting point, which is simpler 
    than full structural analysis but effective for many dance genres.
    """
    try:
        _, boundaries = librosa.beat.beat_track(y=y, sr=sr)
        segments = []
        duration = len(y) / sr
        
        # Convert boundaries from frames to time (seconds)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr).tolist()
        if not boundary_times or boundary_times[0] > 0.1:
            boundary_times.insert(0, 0.0)
        if boundary_times[-1] < duration - 0.1:
            boundary_times.append(duration)

        for i in range(1, len(boundary_times)):
            start_time = boundary_times[i-1]
            end_time = boundary_times[i]
            
            # Segment audio is needed to calculate energy labels later
            start_sample = int(start_time * sr)
            end_sample = int(min(end_time * sr, len(y)))
            seg_audio = y[start_sample:end_sample]
            
            segments.append({'start': round(start_time, 2), 'end': round(end_time, 2), 'audio': seg_audio})
        
        # Post-process segments to label high/low energy
        if segments:
            energies = [np.mean(rms(y=seg['audio'])) for seg in segments]
            if energies:
                med = np.median(energies)
                # Label based on whether segment energy is above or below the median
                labels = ['High' if e > med else 'Low' for e in energies]
            else:
                labels = ['Low'] * len(segments)
            
            cleaned_segments = []
            for i, seg in enumerate(segments):
                cleaned_segments.append({
                    'start': seg['start'], 
                    'end': seg['end'], 
                    'label': labels[i]
                })
            return cleaned_segments

        return [{'start': 0.0, 'end': duration, 'label': 'Full Track'}] # Fallback for no segments

    except Exception as e:
        print(f"Segmentation failed: {e}")
        duration = len(y) / sr
        return [{'start': 0.0, 'end': duration, 'label': 'Full Track'}]

def _extract_similarity_features(y, sr):
    """Extract general features for similarity matching (e.g., Euclidean distance)."""
    energy = np.mean(rms(y=y))
    spectral_centroid_mean = np.mean(spectral_centroid(y=y, sr=sr))
    spectral_rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
    return np.array([energy, spectral_centroid_mean, spectral_rolloff_mean, zero_crossing])

def _compute_theme_descriptor(y, sr):
    """Extract a compact theme vector from spectral and MFCC features (no PCA for single track)."""
    try:
        # 1. Extract core spectral features
        features = []
        features.append(np.mean(spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        features.append(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
        
        # 2. MFCC features
        mfccs_feat = mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs_feat, axis=1))  # 13 MFCC means

        # 3. Return as numpy array
        return np.array(features)  # Total length: 16
    except Exception as e:
        print(f"Theme descriptor failed: {e}")
        return np.zeros(16)


def _key_to_semitone(key, scale):
    """Map key to semitone index (0-11 major, 12-23 minor)."""
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = keys.index(key)
    if scale == 'minor':
        idx += 12
    return idx

def _detect_vocals(y, sr):
    """Simple heuristic for singing voice detection (presence of energy in human vocal range)."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    # Look between 200 Hz (low male) and 4000 Hz (high female/harmonics)
    mid_idx = np.where((freqs > 200) & (freqs < 4000))[0]
    mid_energy = np.mean(S[mid_idx, :])
    # Threshold is empirically set
    return bool(mid_energy > 0.01)

# --- Main Analysis Function ---

def analyze_track(title, artist, filename):
    """Analyze local MP3 file to extract comprehensive audio features using Librosa."""
    file_path = os.path.join(SONGS_DIR, filename)
    
    # CRITICAL FIX: Return error dictionary instead of raising, so setlist analysis continues
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
        y, sr = librosa.load(file_path, sr=None)
        
        # Core Timing and Tonal Features
        bpm = robust_tempo(y, sr)
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_positions = librosa.frames_to_time(beats, sr=sr)
        downbeats = _detect_downbeats(y, sr, beats)

        # Chromagram is calculated once and its mean is used for key estimation
        chroma_mat = chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma_mat, axis=1)
        key, scale = estimate_key(chroma_mean)
        key_semitone = _key_to_semitone(key, scale)
        key_name = f"{key}m" if scale == 'minor' else key

        # Energy and Vibe Heuristics
        energy_rms = np.mean(rms(y=y))
        energy = round(float(energy_rms), 2)
        centroid = np.mean(spectral_centroid(y=y, sr=sr))
        valence = round(float(min(centroid / 5000, 1)), 2) # Approximation
        contrast_mean = np.mean(spectral_contrast(y=y, sr=sr))
        danceability = round(float(min(contrast_mean / 40, 1)), 2) # Approximation

        # Advanced Structural & Similarity Features
        segments = _structural_segmentation(y, sr)
        similarity_features = _extract_similarity_features(y, sr)
        has_vocals = _detect_vocals(y, sr)
        theme_vector = _compute_theme_descriptor(y, sr)
        
        # Package result
        return {
            "title": title, "artist": artist, "file": filename, 
            "bpm": round(bpm), "beat_positions": beat_positions.tolist(), "downbeats": downbeats.tolist(),
            "key": key_name, "key_semitone": key_semitone, "scale": scale, "genre": "Unknown",
            "energy": energy, "valence": valence, "danceability": danceability,
            "segments": segments,
            "similarity_features": similarity_features.tolist(),
            "chroma_matrix": chroma_mat.tolist(), # Return full matrix (large)
            "has_vocals": bool(has_vocals),
            "theme_vector": theme_vector.tolist()
        }
    
    except Exception as e:
        # Fallback for analysis errors (e.g., corrupted file)
        print(f"Error analyzing '{title}' by '{artist}': {e}")
        return {
            "title": title, "artist": artist, "file": filename, 
            "bpm": 120, "key": "C", "key_semitone": 0, "scale": "major", 
            "energy": 0.5, "valence": 0.5, "danceability": 0.5,
            "beat_positions": [], "downbeats": [], "segments": [],
            "similarity_features": [0.5, 2000, 4000, 0.1], "chroma_matrix": None, "has_vocals": False, 
            "theme_vector": [0.5] * 5, "genre": "Analysis Failed"
        }


# --- Transition and Vibe Logic (Unchanged and Correct) ---

def compute_vibe_label(energy, valence, danceability):
    """Compute contextual vibe label based on metadata."""
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
    """Suggest transition type based on previous and current track metadata."""
    if not prev_track_data:
        return "Fade In"
    
    prev_bpm = prev_track_data.get('bpm', 120)
    curr_bpm = current_track_data.get('bpm', 120)
    prev_energy = prev_track_data.get('energy', 0.15)
    curr_energy = current_track_data.get('energy', 0.15)
    bpm_diff = abs(prev_bpm - curr_bpm)
    
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
    """Generate contextual notes for the track."""
    notes = f"{vibe_label} track. "
    if energy > 0.2:
        notes += "High energy suitable for peak moments. "
    if danceability > 0.8:
        notes += "Excellent for dancing. "
    notes += f"Genre: {genre}."
    return notes

# --- Setlist Analysis and Output ---

def analyze_tracks_in_setlist(setlist_json):
    """Analyze tracks from setlist using local MP3s, add advanced features, then sort by BPM for better transitions."""
    try:
        setlist_data = json.loads(setlist_json)
        analyzed_setlist = []
        
        # Use a global tracker for smooth transitions across setlist segments
        global_prev_track_data = None
        
        for segment in setlist_data["setlist"]:
            time_range = segment["time"]
            tracks = segment["tracks"]
            temp_analyzed = []
            
            for track in tracks:
                title = track["title"]
                artist = track["artist"]
                filename = track["file"]
                
                metadata = analyze_track(title, artist, filename)
                
                vibe_label = compute_vibe_label(
                    metadata.get("energy", 0.5), # Use get with default to prevent error on missing files
                    metadata.get("valence", 0.5), 
                    metadata.get("danceability", 0.5)
                )
                notes = generate_notes(vibe_label, metadata.get("energy", 0.5), metadata.get("danceability", 0.5), metadata.get("genre", "Unknown"))
                
                analyzed_track = {
                    **metadata,
                    "notes": notes,
                    "transition": None
                }
                temp_analyzed.append(analyzed_track)
            
            # Sort tracks in the segment by BPM
            temp_analyzed.sort(key=lambda x: x['bpm'])
            
            analyzed_tracks = []
            for analyzed_track in temp_analyzed:
                metadata_for_transition = {
                    "bpm": analyzed_track["bpm"],
                    "energy": analyzed_track["energy"]
                }
                
                transition = suggest_transition(global_prev_track_data, metadata_for_transition)
                analyzed_track["transition"] = transition
                analyzed_tracks.append(analyzed_track)
                global_prev_track_data = metadata_for_transition
            
            analyzed_setlist.append({
                "time": time_range,
                "analyzed_tracks": analyzed_tracks
            })
        
        output = {"analyzed_setlist": analyzed_setlist}
        with open("analyzed_setlist.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Analyzed setlist saved to 'analyzed_setlist.json'")
        
        return output
    except Exception as e:
        print(f"Error in Track Analysis Engine: {str(e)}")
        raise

if __name__ == "__main__":
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
    analyze_tracks_in_setlist(sample_setlist_json)
