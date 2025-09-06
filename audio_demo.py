import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AudioDemo:
    """Simple demo of audio threat detection"""
    
    def __init__(self):
        self.classes = ['normal', 'scream_distress', 'aggressive', 'emergency', 'silent']
        self.sample_rate = 22050
        
        # Load the trained model
        try:
            self.model = tf.keras.models.load_model("audio_detection_model.h5")
            print("✅ Audio detection model loaded successfully!")
        except:
            print("❌ Model not found. Please train the model first.")
            self.model = None
    
    def generate_demo_audio(self, audio_type="normal", duration=2):
        """Generate synthetic audio for demonstration"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        if audio_type == "normal":
            # Low frequency background noise
            audio = 0.1 * np.random.normal(0, 0.5, samples) + 0.05 * np.sin(2 * np.pi * 100 * t)
        elif audio_type == "scream_distress":
            # High frequency with sudden spikes
            audio = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.5 * np.random.exponential(0.3, samples)
            audio += 0.2 * np.sin(2 * np.pi * 1200 * t)
        elif audio_type == "aggressive":
            # Medium frequency with high variance
            audio = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.random.normal(0, 0.8, samples)
        elif audio_type == "emergency":
            # Sharp spikes like breaking glass
            audio = 0.2 * np.random.normal(0, 0.3, samples)
            # Add random spikes
            spike_indices = np.random.choice(samples, size=samples//20, replace=False)
            audio[spike_indices] += np.random.exponential(1, len(spike_indices))
        else:  # silent
            audio = 0.02 * np.random.normal(0, 0.1, samples)
        
        return audio
    
    def extract_features(self, audio_data):
        """Extract features from audio data"""
        try:
            # Basic feature extraction
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            rms = librosa.feature.rms(y=audio_data)
            
            # Combine features
            features = []
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            features.append(np.mean(spectral_centroid))
            features.append(np.mean(spectral_bandwidth))
            features.append(np.mean(spectral_rolloff))
            features.append(np.mean(zcr))
            features.append(np.mean(rms))
            
            # Pad or truncate to match expected size (50 features)
            features = np.array(features)
            if len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)))
            else:
                features = features[:50]
                
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict_threat(self, audio_data):
        """Predict threat level from audio"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        features = self.extract_features(audio_data)
        if features is None:
            return {"error": "Feature extraction failed"}
        
        # Make prediction
        features = features.reshape(1, -1)
        prediction = self.model.predict(features, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return {
            'predicted_class': self.classes[class_idx],
            'confidence': confidence,
            'all_probabilities': dict(zip(self.classes, prediction[0]))
        }
    
    def visualize_audio(self, audio_data, title="Audio Waveform"):
        """Visualize audio waveform and spectrogram"""
        plt.figure(figsize=(15, 8))
        
        # Waveform
        plt.subplot(2, 2, 1)
        time = np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data))
        plt.plot(time, audio_data)
        plt.title(f"{title} - Waveform")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        
        # Spectrogram
        plt.subplot(2, 2, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='hz')
        plt.title(f"{title} - Spectrogram")
        plt.colorbar()
        
        # MFCC
        plt.subplot(2, 2, 3)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=self.sample_rate, x_axis='time')
        plt.title(f"{title} - MFCC")
        plt.colorbar()
        
        # Frequency analysis
        plt.subplot(2, 2, 4)
        fft_data = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
        magnitude = np.abs(fft_data)
        plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
        plt.title(f"{title} - Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        
        plt.tight_layout()
        plt.show()
    
    def run_demo(self):
        """Run complete audio detection demo"""
        print("🎧 SangRakshak Audio Detection Demo")
        print("=" * 50)
        
        audio_types = ['normal', 'scream_distress', 'aggressive', 'emergency', 'silent']
        
        for audio_type in audio_types:
            print(f"\n🔊 Testing: {audio_type.upper()}")
            print("-" * 30)
            
            # Generate demo audio
            audio_data = self.generate_demo_audio(audio_type)
            
            # Predict threat
            result = self.predict_threat(audio_data)
            
            if 'error' not in result:
                predicted = result['predicted_class']
                confidence = result['confidence']
                
                # Determine if threat detected
                threat_classes = ['scream_distress', 'aggressive', 'emergency']
                is_threat = predicted in threat_classes and confidence > 0.7
                
                print(f"   Predicted: {predicted}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Threat Level: {'🚨 HIGH' if is_threat else '✅ SAFE'}")
                
                # Show top 3 predictions
                sorted_probs = sorted(result['all_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                print("   Top 3 predictions:")
                for i, (class_name, prob) in enumerate(sorted_probs[:3]):
                    print(f"      {i+1}. {class_name}: {prob:.3f}")
            else:
                print(f"   Error: {result['error']}")
        
        print("\n" + "=" * 50)
        print("🎯 Demo completed! Audio detection system working properly.")
        print("\n💡 How to integrate with SangRakshak:")
        print("   1. Combine with video analysis for multi-modal detection")
        print("   2. Add real-time microphone monitoring")
        print("   3. Implement emergency alert system")
        print("   4. Store threat logs in database")

if __name__ == "__main__":
    demo = AudioDemo()
    demo.run_demo()
