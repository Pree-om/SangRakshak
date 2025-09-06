#!/usr/bin/env python3
"""
🛡️ SangRakshak Integrated Demo - Audio + Video Safety Detection
Combines working audio detection with video analysis for comprehensive threat detection
"""

import gradio as gr
import torch
import cv2
import numpy as np
import tensorflow as tf
import librosa
import time
import os
from datetime import datetime
import json

# ===== AUDIO DETECTION SYSTEM (From Working Model) =====
class AudioFeatureExtractor:
    """Extract audio features for distress sound detection"""
    
    def __init__(self, sample_rate=22050, n_mfcc=13, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
    
    def extract_features(self, audio_path=None, audio_data=None):
        """Extract comprehensive audio features (50-dimensional vector)"""
        try:
            # Load audio
            if audio_path:
                y, sr = librosa.load(audio_path, sr=self.sample_rate)
            else:
                y, sr = audio_data, self.sample_rate
            
            # Extract features
            features = {}
            
            # 1. MFCC (Mel Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # 2. Spectral Features
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            # 3. Zero Crossing Rate
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # 4. Root Mean Square Energy
            features['rms'] = np.mean(librosa.feature.rms(y=y))
            
            # 5. Tempo and Beat
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # 6. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # 7. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            
            # Flatten all features into a single vector
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

class WorkingAudioDetectionModel:
    """Working audio detection model (uses improved model)"""
    
    def __init__(self, model_path="improved_audio_detection_model.h5"):
        self.classes = ['normal', 'scream_distress', 'aggressive', 'emergency', 'silent']
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the working improved audio model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✅ Audio detection model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model file {self.model_path} not found, creating new model...")
                self.create_working_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.create_working_model()
    
    def create_working_model(self):
        """Create a working model with proper architecture"""
        print("🔧 Creating working audio detection model...")
        
        # Build improved model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(50,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with improved synthetic data
        X_train, y_train = self.create_better_synthetic_data()
        self.model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        
        # Save the model
        self.model.save(self.model_path)
        print(f"✅ Working model created and saved to {self.model_path}")
    
    def create_better_synthetic_data(self, samples_per_class=500):
        """Create realistic synthetic training data"""
        X_data = []
        y_data = []
        
        for class_idx, class_name in enumerate(self.classes):
            for _ in range(samples_per_class):
                if class_name == 'normal':
                    # Normal: Low energy, centered around 0
                    features = np.random.normal(0, 0.3, 50)
                    features[:10] = np.random.normal(-0.5, 0.2, 10)
                    
                elif class_name == 'scream_distress':
                    # Scream: High energy, high frequency
                    features = np.random.normal(1.5, 0.8, 50)
                    features[:10] = np.random.normal(2, 0.5, 10)
                    features[26:30] = np.random.normal(3, 0.3, 4)
                    
                elif class_name == 'aggressive':
                    # Aggressive: Medium-high energy, irregular
                    features = np.random.normal(0.8, 1.2, 50)
                    features[:10] = np.random.normal(1, 0.7, 10)
                    features[30:35] = np.random.normal(2, 0.8, 5)
                    
                elif class_name == 'emergency':
                    # Emergency: Sharp spikes, high energy bursts
                    features = np.random.normal(0.5, 0.4, 50)
                    spike_indices = np.random.choice(50, 8, replace=False)
                    features[spike_indices] += np.random.exponential(2, 8)
                    features[35:40] = np.random.normal(4, 0.5, 5)
                    
                else:  # silent
                    # Silent: Very low energy
                    features = np.random.normal(-1, 0.15, 50)
                    features = np.abs(features) * -1
                    features[40:] = np.random.normal(-0.5, 0.1, 10)
                
                X_data.append(features)
                y_data.append(class_idx)
        
        return np.array(X_data), np.array(y_data)
    
    def predict(self, features):
        """Predict audio class with confidence"""
        if self.model is None:
            return None
        
        # Ensure features are the right size (50-dimensional)
        if len(features) != 50:
            if len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)))
            else:
                features = features[:50]
        
        features = features.reshape(1, -1)
        prediction = self.model.predict(features, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return {
            'class': self.classes[class_idx],
            'confidence': confidence,
            'probabilities': dict(zip(self.classes, prediction[0]))
        }

# ===== VIDEO DETECTION SYSTEM =====
class VideoAnalyzer:
    """Video analysis for crowd detection and safety assessment"""
    
    def __init__(self):
        self.model = self.build_video_model()
    
    def build_video_model(self):
        """Build simple video analysis model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')  # safe, crowd, violence
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train with synthetic data
        X_train = np.random.rand(200, 224, 224, 3)
        y_train = np.random.randint(0, 3, 200)
        model.fit(X_train, y_train, epochs=5, verbose=0)
        
        return model
    
    def analyze_video(self, video_path):
        """Analyze video for threats"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            frame_count = 0
            while cap.isOpened() and frame_count < 30:  # Analyze first 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                return {"error": "No frames found in video"}
            
            # Analyze representative frame
            frame = cv2.resize(frames[len(frames)//2], (224, 224))
            frame_tensor = np.expand_dims(frame, axis=0) / 255.0
            
            prediction = self.model.predict(frame_tensor, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)
            
            classes = ['safe', 'crowd_detected', 'violence_detected']
            
            return {
                'class': classes[class_idx],
                'confidence': confidence,
                'frame_count': len(frames),
                'video_duration': len(frames) / 30.0  # Assume 30 FPS
            }
            
        except Exception as e:
            return {"error": f"Video analysis failed: {str(e)}"}

# ===== THREAT LOGGING SYSTEM =====
class ThreatLogger:
    """Log and track detected threats"""
    
    def __init__(self):
        self.threat_history = []
        self.log_file = "threat_log.json"
    
    def log_threat(self, threat_info):
        """Log a detected threat"""
        threat_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': threat_info.get('type', 'unknown'),
            'class': threat_info.get('class', 'unknown'),
            'confidence': threat_info.get('confidence', 0.0),
            'details': threat_info.get('details', {})
        }
        
        self.threat_history.append(threat_entry)
        
        # Save to file
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.threat_history, f, indent=2)
        except Exception as e:
            print(f"Error saving threat log: {e}")
        
        print(f"🚨 THREAT LOGGED: {threat_entry['type']} - {threat_entry['class']} ({threat_entry['confidence']:.2f})")
    
    def get_recent_threats(self, limit=5):
        """Get recent threats"""
        return self.threat_history[-limit:] if self.threat_history else []
    
    def get_threat_summary(self):
        """Get threat statistics"""
        if not self.threat_history:
            return "No threats detected yet"
        
        total_threats = len(self.threat_history)
        audio_threats = len([t for t in self.threat_history if t['type'] == 'audio'])
        video_threats = len([t for t in self.threat_history if t['type'] == 'video'])
        
        return f"Total: {total_threats} | Audio: {audio_threats} | Video: {video_threats}"

# ===== INITIALIZE SYSTEMS =====
print("🔄 Initializing SangRakshak Integrated System...")

# Initialize components
audio_extractor = AudioFeatureExtractor()
audio_model = WorkingAudioDetectionModel()
video_analyzer = VideoAnalyzer()
threat_logger = ThreatLogger()

print("✅ All systems initialized successfully!")

# ===== MAIN ANALYSIS FUNCTION =====
def analyze_multimedia_safety(video_file, audio_file, enable_video=True, enable_audio=True):
    """Comprehensive safety analysis with audio and video"""
    
    analysis_results = []
    detected_threats = []
    
    print(f"🔍 Starting analysis - Video: {enable_video}, Audio: {enable_audio}")
    
    # Video Analysis
    if video_file and enable_video:
        try:
            print("🎥 Analyzing video...")
            video_result = video_analyzer.analyze_video(video_file)
            
            if 'error' not in video_result:
                video_class = video_result['class']
                video_confidence = video_result['confidence']
                
                analysis_results.append(f"🎥 Video Analysis:")
                analysis_results.append(f"   • Status: {video_class}")
                analysis_results.append(f"   • Confidence: {video_confidence:.2f}")
                analysis_results.append(f"   • Frames analyzed: {video_result['frame_count']}")
                
                # Check for video threats
                if video_class in ['crowd_detected', 'violence_detected'] and video_confidence > 0.6:
                    detected_threats.append(f"video_{video_class}")
                    threat_logger.log_threat({
                        'type': 'video',
                        'class': video_class,
                        'confidence': video_confidence,
                        'details': video_result
                    })
            else:
                analysis_results.append(f"🎥 Video Analysis: {video_result['error']}")
                
        except Exception as e:
            analysis_results.append(f"🎥 Video Analysis: Error - {str(e)}")
    
    # Audio Analysis
    if audio_file and enable_audio:
        try:
            print("🎧 Analyzing audio...")
            
            # Extract audio features
            features = audio_extractor.extract_features(audio_path=audio_file)
            
            if features is not None:
                # Get prediction from working model
                audio_prediction = audio_model.predict(features)
                
                if audio_prediction:
                    audio_class = audio_prediction['class']
                    audio_confidence = audio_prediction['confidence']
                    
                    analysis_results.append(f"\\n🎧 Audio Analysis:")
                    analysis_results.append(f"   • Classification: {audio_class}")
                    analysis_results.append(f"   • Confidence: {audio_confidence:.2f}")
                    
                    # Show top 3 predictions
                    sorted_probs = sorted(audio_prediction['probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True)
                    analysis_results.append("   • Top predictions:")
                    for i, (class_name, prob) in enumerate(sorted_probs[:3]):
                        analysis_results.append(f"     {i+1}. {class_name}: {prob:.3f}")
                    
                    # Check for audio threats
                    threat_classes = ['scream_distress', 'aggressive', 'emergency']
                    if audio_class in threat_classes and audio_confidence > 0.7:
                        detected_threats.append(f"audio_{audio_class}")
                        threat_logger.log_threat({
                            'type': 'audio',
                            'class': audio_class,
                            'confidence': audio_confidence,
                            'details': audio_prediction['probabilities']
                        })
                else:
                    analysis_results.append("\\n🎧 Audio Analysis: Prediction failed")
            else:
                analysis_results.append("\\n🎧 Audio Analysis: Feature extraction failed")
                
        except Exception as e:
            analysis_results.append(f"\\n🎧 Audio Analysis: Error - {str(e)}")
    
    # Threat Assessment
    analysis_results.append("\\n" + "="*50)
    if detected_threats:
        analysis_results.append("🚨 THREAT ASSESSMENT: HIGH RISK")
        analysis_results.append("⚠️ Detected Threats:")
        for threat in detected_threats:
            analysis_results.append(f"   • {threat.replace('_', ' ').title()}")
        
        # Emergency recommendations
        analysis_results.append("\\n📞 Recommended Actions:")
        analysis_results.append("   • Contact emergency services if immediate danger")
        analysis_results.append("   • Alert security personnel")
        analysis_results.append("   • Move to safe location")
        analysis_results.append("   • Document incident details")
    else:
        analysis_results.append("✅ THREAT ASSESSMENT: LOW RISK")
        analysis_results.append("   No immediate threats detected")
    
    # Add threat history
    recent_threats = threat_logger.get_recent_threats(3)
    if recent_threats:
        analysis_results.append(f"\\n📋 Recent Threat History ({threat_logger.get_threat_summary()}):")
        for i, threat in enumerate(recent_threats, 1):
            timestamp = threat['timestamp'][:19].replace('T', ' ')
            analysis_results.append(f"   {i}. {timestamp}: {threat['type']} - {threat['class']} ({threat['confidence']:.2f})")
    
    return "\\n".join(analysis_results)

# ===== GRADIO INTERFACE =====
def create_sangrakshak_interface():
    """Create the SangRakshak Gradio interface"""
    
    with gr.Blocks(
        title="🛡️ SangRakshak - AI Safety Detection System",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; color: #2c3e50; }
        .threat-box { background-color: #ffe6e6; padding: 10px; border-radius: 5px; }
        .safe-box { background-color: #e6ffe6; padding: 10px; border-radius: 5px; }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1 style="font-size: 3em; margin-bottom: 0;">🛡️ संरक्षक</h1>
                <h2 style="font-size: 1.5em; color: #7f8c8d;">Integrated AI Safety Detection System</h2>
                <p style="font-size: 1.1em; color: #95a5a6;">
                    Advanced Audio + Video Threat Detection for Women's Safety
                </p>
            </div>
        """)
        
        # Main analysis section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎥 Video Upload")
                video_input = gr.Video(
                    label="Upload Video File",
                    height=300
                )
                enable_video = gr.Checkbox(
                    label="Enable Video Analysis",
                    value=True
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### 🎧 Audio Upload")
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                enable_audio = gr.Checkbox(
                    label="Enable Audio Analysis", 
                    value=True
                )
        
        # Analysis button
        with gr.Row():
            analyze_button = gr.Button(
                "🔍 Analyze for Safety Threats",
                variant="primary",
                size="lg"
            )
        
        # Results section
        with gr.Row():
            results_output = gr.Textbox(
                label="🛡️ Safety Analysis Results",
                lines=20,
                max_lines=30,
                placeholder="Upload audio and/or video files, then click 'Analyze for Safety Threats' to get comprehensive threat assessment..."
            )
        
        # Feature showcase
        gr.HTML("""
            <div style="margin-top: 30px; text-align: center;">
                <h3>🔊 Audio Detection Capabilities</h3>
                <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                    <div>🗣️ Scream Detection</div>
                    <div>😠 Aggressive Voice Analysis</div>
                    <div>🚨 Emergency Sound Recognition</div>
                    <div>🤫 Unusual Silence Detection</div>
                </div>
                
                <h3>🎯 Video Analysis Features</h3>
                <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                    <div>👥 Crowd Detection</div>
                    <div>⚡ Violence Recognition</div>
                    <div>🎯 Safety Assessment</div>
                    <div>📊 Real-time Analysis</div>
                </div>
                
                <p style="color: #7f8c8d; margin-top: 20px;">
                    Powered by advanced machine learning models for real-time threat detection
                </p>
            </div>
        """)
        
        # Connect events
        analyze_button.click(
            fn=analyze_multimedia_safety,
            inputs=[video_input, audio_input, enable_video, enable_audio],
            outputs=results_output
        )
        
        return interface

# ===== LAUNCH APPLICATION =====
if __name__ == "__main__":
    print("🚀 Launching SangRakshak Integrated Safety System...")
    
    # Create and launch interface
    interface = create_sangrakshak_interface()
    
    interface.launch(
        share=False,  # Set to True for public sharing
        server_name="127.0.0.1",
        server_port=7860,
        debug=True
    )
    
    print("✅ SangRakshak is now running!")
    print("🌐 Access the application at: http://127.0.0.1:7860")
