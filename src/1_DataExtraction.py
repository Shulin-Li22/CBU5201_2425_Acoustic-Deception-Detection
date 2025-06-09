import numpy as np
import pandas as pd
import librosa
import os
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self, audio_dir, metadata_path, sample_rate=22050):
        self.audio_dir = audio_dir
        self.metadata_path = metadata_path
        self.sample_rate = sample_rate
        self.metadata = pd.read_csv(metadata_path)
        
    def get_pitch_features(self, audio):
        """Extract pitch (F0) related features"""
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        
        # Get the pitch with highest magnitude for each time
        pitch_values = []
        for time_idx in range(pitches.shape[1]):
            pitch_idx = magnitudes[:, time_idx].argmax()
            pitch_values.append(pitches[pitch_idx, time_idx])
            
        pitch_values = np.array(pitch_values)
        pitch_values = pitch_values[pitch_values > 0]  # Remove zero pitches
        
        if len(pitch_values) == 0:
            return {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_range': 0,
                'pitch_var': 0
            }
            
        return {
            'pitch_mean': np.mean(pitch_values),
            'pitch_std': np.std(pitch_values),
            'pitch_range': np.max(pitch_values) - np.min(pitch_values),
            'pitch_var': np.var(pitch_values)
        }
    
    def get_voice_quality_features(self, audio):
        """Extract voice quality features (similar to jitter/shimmer)"""
        # Zero crossing rate can indicate voice breaks
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Compute frame-to-frame energy differences (similar to shimmer)
        rmse = librosa.feature.rms(y=audio)[0]
        energy_diff = np.diff(rmse)
        
        # Compute spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        
        return {
            'voice_breaks_rate': np.mean(zcr),
            'energy_variability': np.std(energy_diff),
            'spectral_contrast_mean': np.mean(contrast),
            'spectral_contrast_std': np.std(contrast)
        }
    
    def get_rhythm_features(self, audio):
        """Extract enhanced rhythm features"""
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        
        # Calculate timing between beats
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            beat_interval_stats = {
                'beat_interval_mean': np.mean(beat_intervals),
                'beat_interval_std': np.std(beat_intervals),
                'rhythm_regularity': np.std(beat_intervals) / np.mean(beat_intervals)
            }
        else:
            beat_interval_stats = {
                'beat_interval_mean': 0,
                'beat_interval_std': 0,
                'rhythm_regularity': 0
            }
            
        return {
            'tempo': tempo,
            **beat_interval_stats
        }
    
    def get_harmonic_features(self, audio):
        """Extract harmonic features"""
        # Harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Harmonic-to-percussive ratio
        harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(percussive)) + 1e-10)
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        
        return {
            'harmonic_ratio': harmonic_ratio,
            'spectral_flatness_mean': np.mean(flatness),
            'spectral_flatness_std': np.std(flatness)
        }
    
    def get_segment_features(self, audio, n_segments=3):
        """Extract features from different segments of the audio"""
        segment_length = len(audio) // n_segments
        segment_features = {}
        
        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < n_segments - 1 else len(audio)
            segment = audio[start:end]
            
            # Extract basic features for each segment
            mfcc_segment = librosa.feature.mfcc(y=segment, sr=self.sample_rate, n_mfcc=13)
            rmse_segment = librosa.feature.rms(y=segment)[0]
            
            segment_features.update({
                f'segment{i+1}_mfcc_mean': np.mean(mfcc_segment),
                f'segment{i+1}_energy_mean': np.mean(rmse_segment)
            })
        
        return segment_features
    
    def extract_features(self, audio):
        """Extract all audio features"""
        features = {}
        
        # 1. Original MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        for idx, mfcc in enumerate(mfccs):
            features[f'mfcc{idx+1}_mean'] = np.mean(mfcc)
            features[f'mfcc{idx+1}_std'] = np.std(mfcc)
        
        # 2. Original spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features.update({
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff)
        })
        
        # 3. Original rhythm and energy features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        rmse = librosa.feature.rms(y=audio)[0]
        features.update({
            'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
            'zero_crossing_rate_std': np.std(zero_crossing_rate),
            'rmse_mean': np.mean(rmse),
            'rmse_std': np.std(rmse)
        })
        
        # 4. New pitch features
        features.update(self.get_pitch_features(audio))
        
        # 5. New voice quality features
        features.update(self.get_voice_quality_features(audio))
        
        # 6. New rhythm features
        features.update(self.get_rhythm_features(audio))
        
        # 7. New harmonic features
        features.update(self.get_harmonic_features(audio))
        
        # 8. Segment features
        features.update(self.get_segment_features(audio))
        
        return features
    
    def process_all_files(self):
        """Process all audio files and extract features"""
        all_features = []
        
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Processing audio files"):
            file_path = os.path.join(self.audio_dir, row['filename'])
            try:
                # Load audio
                audio, _ = librosa.load(file_path, sr=self.sample_rate)
                
                # Extract features
                features = self.extract_features(audio)
                
                # Add metadata
                features['filename'] = row['filename']
                features['Language'] = row['Language']
                features['Story_type'] = row['Story_type']
                
                all_features.append(features)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save features to CSV
        features_df.to_csv('audio_features.csv', index=False)
        print("Features saved to audio_features.csv")
        
        return features_df


extractor = AudioFeatureExtractor(
    audio_dir='CBU0521DD_stories',
    metadata_path='CBU0521DD_stories_attributes.csv'
)

# Extract features and save to CSV
features_df = extractor.process_all_files()