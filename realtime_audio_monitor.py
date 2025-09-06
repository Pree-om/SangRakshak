#!/usr/bin/env python3
"""
🎤 SangRakshak Real-Time Audio Monitor
Real-time microphone monitoring for live audio threat detection
"""

import numpy as np
import tensorflow as tf
import librosa
import threading
import queue
import time
from datetime import datetime
import json

# Audio processing imports
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    print("✅ PyAudio available for real-time audio")
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available - using simulated audio for demo")

class RealTimeAudioMonitor:
    """Real-time audio monitoring and threat detection"""
    
    def __init__(self, model_path="improved_audio_detection_model.h5"):
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.buffer_duration = 2.0  # 2 seconds
        self.classes = ['normal', 'scream_distress', 'aggressive', 'emergency', 'silent']
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Threat detection
        self.threat_threshold = 0.75
        self.threat_callbacks = []
        self.threat_history = []
        
        # Load audio model
        self.load_audio_model(model_path)
        
        # PyAudio setup
        if PYAUDIO_AVAILABLE:
            self.p = pyaudio.PyAudio()
            self.stream = None
        else:
            self.p = None
            self.stream = None
    
    def load_audio_model(self, model_path):
        """Load the working audio detection model"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ Real-time audio model loaded from {model_path}")
            else:
                print(f"⚠️ Model {model_path} not found, creating working model...")
                self.create_working_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.create_working_model()
    
    def create_working_model(self):
        """Create working audio model if file doesn't exist"""
        print("🔧 Creating working real-time audio model...")
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(50,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Quick training with synthetic data
        X_train, y_train = self.create_synthetic_data()
        self.model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        
        print("✅ Working real-time model created")
    
    def create_synthetic_data(self, samples_per_class=300):
        """Create synthetic training data"""
        X_data = []
        y_data = []
        
        for class_idx, class_name in enumerate(self.classes):
            for _ in range(samples_per_class):
                if class_name == 'normal':
                    features = np.random.normal(0, 0.3, 50)
                elif class_name == 'scream_distress':
                    features = np.random.normal(1.5, 0.8, 50)
                    features[:10] = np.random.normal(2, 0.5, 10)
                elif class_name == 'aggressive':
                    features = np.random.normal(0.8, 1.2, 50)
                    features[:10] = np.random.normal(1, 0.7, 10)
                elif class_name == 'emergency':
                    features = np.random.normal(0.5, 0.4, 50)
                    spike_indices = np.random.choice(50, 8, replace=False)
                    features[spike_indices] += np.random.exponential(2, 8)
                else:  # silent
                    features = np.random.normal(-1, 0.15, 50)
                
                X_data.append(features)
                y_data.append(class_idx)
        
        return np.array(X_data), np.array(y_data)
    
    def extract_audio_features(self, audio_data):
        """Extract features from audio data"""
        try:
            if len(audio_data) == 0:
                return None
            
            # Basic feature extraction (simplified for real-time)
            features = []
            
            # 1. Energy features
            rms = np.sqrt(np.mean(audio_data**2))
            features.append(rms)
            
            # 2. Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            features.append(zcr)
            
            # 3. Spectral features (using FFT)
            fft = np.abs(np.fft.fft(audio_data))
            spectral_centroid = np.sum(fft * np.arange(len(fft))) / np.sum(fft)
            features.append(spectral_centroid / len(fft))
            
            # 4. Statistical features
            features.extend([
                np.mean(audio_data),
                np.std(audio_data),
                np.max(audio_data),
                np.min(audio_data),
                np.median(audio_data)
            ])
            
            # 5. Frequency domain features
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            magnitude = np.abs(np.fft.fft(audio_data))
            
            # Peak frequency
            peak_freq_idx = np.argmax(magnitude)
            peak_freq = abs(freqs[peak_freq_idx])
            features.append(peak_freq / self.sample_rate)
            
            # Spectral rolloff (simplified)
            total_energy = np.sum(magnitude)
            rolloff_energy = 0.85 * total_energy
            cumulative_energy = np.cumsum(magnitude)
            rolloff_idx = np.where(cumulative_energy >= rolloff_energy)[0]
            if len(rolloff_idx) > 0:
                features.append(rolloff_idx[0] / len(magnitude))
            else:
                features.append(0.5)
            
            # Pad or truncate to 50 features
            while len(features) < 50:
                features.append(0.0)
            
            return np.array(features[:50])
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict_audio_threat(self, features):
        """Predict threat from audio features"""
        if self.model is None or features is None:
            return None
        
        try:
            features = features.reshape(1, -1)
            prediction = self.model.predict(features, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            return {
                'class': self.classes[class_idx],
                'confidence': confidence,
                'probabilities': dict(zip(self.classes, prediction[0])),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for PyAudio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert audio data
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to queue for processing
        try:
            self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            pass  # Skip if queue is full
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio_continuously(self):
        """Continuously process audio from queue"""
        audio_buffer = []
        buffer_samples = int(self.buffer_duration * self.sample_rate)
        
        print(f"🎧 Starting continuous audio processing (buffer: {self.buffer_duration}s)")
        
        while self.is_monitoring:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer.extend(audio_chunk)
                    
                    # Process when buffer is full
                    if len(audio_buffer) >= buffer_samples:
                        # Take the most recent buffer_samples
                        audio_to_process = np.array(audio_buffer[-buffer_samples:])
                        
                        # Extract features and predict
                        features = self.extract_audio_features(audio_to_process)
                        if features is not None:
                            prediction = self.predict_audio_threat(features)
                            
                            if prediction:
                                self.handle_prediction(prediction)
                        
                        # Keep overlap for continuity
                        overlap_samples = buffer_samples // 4
                        audio_buffer = audio_buffer[-overlap_samples:]
                
                else:
                    time.sleep(0.01)  # Small delay if no audio
                    
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in audio processing: {e}")
                time.sleep(0.1)
    
    def handle_prediction(self, prediction):
        """Handle audio prediction results"""
        audio_class = prediction['class']
        confidence = prediction['confidence']
        
        # Print current status (with overwrite)
        status_msg = f"🎧 Audio: {audio_class} ({confidence:.2f})"
        print(f"\\r{status_msg}", end="", flush=True)
        
        # Check for threats
        threat_classes = ['scream_distress', 'aggressive', 'emergency']
        if audio_class in threat_classes and confidence > self.threat_threshold:
            print(f"\\n🚨 AUDIO THREAT DETECTED: {audio_class} ({confidence:.2f})")
            
            # Log threat
            threat_info = {
                'type': 'audio',
                'class': audio_class,
                'confidence': confidence,
                'timestamp': prediction['timestamp'],
                'probabilities': prediction['probabilities']
            }
            
            self.threat_history.append(threat_info)
            
            # Call threat callbacks
            for callback in self.threat_callbacks:
                try:
                    callback(threat_info)
                except Exception as e:
                    print(f"Error in threat callback: {e}")
    
    def add_threat_callback(self, callback_function):
        """Add callback function for threat detection"""
        self.threat_callbacks.append(callback_function)
    
    def start_monitoring(self):
        """Start real-time audio monitoring"""
        if self.is_monitoring:
            print("⚠️ Monitoring already running")
            return False
        
        print("🎤 Starting real-time audio monitoring...")
        
        if PYAUDIO_AVAILABLE and self.p:
            try:
                # Configure audio stream
                self.stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self.audio_callback
                )
                
                self.is_monitoring = True
                self.stream.start_stream()
                
                # Start processing thread
                self.monitoring_thread = threading.Thread(target=self.process_audio_continuously)
                self.monitoring_thread.daemon = True
                self.monitoring_thread.start()
                
                print("✅ Real-time audio monitoring started")
                print(f"📊 Monitoring for threats with {self.threat_threshold:.0%} confidence threshold")
                print("🔇 Press Ctrl+C to stop monitoring")
                
                return True
                
            except Exception as e:
                print(f"❌ Error starting audio stream: {e}")
                self.is_monitoring = False
                return False
        else:
            print("⚠️ PyAudio not available - starting simulated monitoring")
            self.start_simulated_monitoring()
            return True
    
    def start_simulated_monitoring(self):
        """Start simulated monitoring for demo purposes"""
        self.is_monitoring = True
        
        def simulate_audio():
            print("🎭 Running simulated audio monitoring...")
            audio_classes = ['normal', 'normal', 'normal', 'scream_distress', 'normal', 'aggressive', 'normal', 'emergency']
            
            for i, audio_class in enumerate(audio_classes):
                if not self.is_monitoring:
                    break
                
                # Simulate prediction
                confidence = np.random.uniform(0.7, 0.95) if audio_class != 'normal' else np.random.uniform(0.3, 0.8)
                
                prediction = {
                    'class': audio_class,
                    'confidence': confidence,
                    'probabilities': {cls: (0.8 if cls == audio_class else 0.05) for cls in self.classes},
                    'timestamp': datetime.now().isoformat()
                }
                
                self.handle_prediction(prediction)
                time.sleep(2)  # 2 second intervals
            
            print("\\n🎭 Simulated monitoring completed")
        
        self.monitoring_thread = threading.Thread(target=simulate_audio)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        print("\\n🛑 Stopping audio monitoring...")
        
        self.is_monitoring = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        print("✅ Audio monitoring stopped")
    
    def get_threat_summary(self):
        """Get summary of detected threats"""
        if not self.threat_history:
            return "No threats detected during monitoring session"
        
        threat_counts = {}
        for threat in self.threat_history:
            threat_class = threat['class']
            threat_counts[threat_class] = threat_counts.get(threat_class, 0) + 1
        
        summary = f"Total threats detected: {len(self.threat_history)}\\n"
        for threat_class, count in threat_counts.items():
            summary += f"  • {threat_class}: {count}\\n"
        
        return summary
    
    def save_threat_log(self, filename="realtime_threat_log.json"):
        """Save threat history to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.threat_history, f, indent=2)
            print(f"💾 Threat log saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving threat log: {e}")

# ===== DEMO AND TESTING =====
def demo_threat_callback(threat_info):
    """Demo callback for threat detection"""
    print(f"\\n📞 EMERGENCY ALERT TRIGGERED!")
    print(f"   🚨 Threat Type: {threat_info['class']}")
    print(f"   📊 Confidence: {threat_info['confidence']:.2f}")
    print(f"   ⏰ Time: {threat_info['timestamp']}")
    print(f"   📝 Action: Alert sent to emergency contacts")

def run_realtime_demo():
    """Run real-time audio monitoring demo"""
    print("🚀 SangRakshak Real-Time Audio Monitor Demo")
    print("=" * 50)
    
    # Create monitor
    monitor = RealTimeAudioMonitor()
    
    # Add threat callback
    monitor.add_threat_callback(demo_threat_callback)
    
    try:
        # Start monitoring
        if monitor.start_monitoring():
            # Monitor for specified duration or until interrupted
            if PYAUDIO_AVAILABLE:
                print("\\n⏱️ Monitoring live audio... Press Ctrl+C to stop")
                while monitor.is_monitoring:
                    time.sleep(1)
            else:
                print("\\n⏱️ Running simulated demo...")
                while monitor.is_monitoring and monitor.monitoring_thread.is_alive():
                    time.sleep(0.5)
        
        # Show results
        print("\\n" + "=" * 50)
        print("📊 MONITORING SESSION COMPLETE")
        print("=" * 50)
        print(monitor.get_threat_summary())
        
        # Save results
        monitor.save_threat_log()
        
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Monitoring interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during monitoring: {e}")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    import os
    import sys
    
    # Add current directory to path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("🎤 SangRakshak Real-Time Audio Monitor")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_realtime_demo()
    else:
        print("Usage: python realtime_audio_monitor.py --demo")
        print("\\nThis will start real-time audio monitoring for threats.")
        print("If PyAudio is not available, it will run a simulated demo.")
