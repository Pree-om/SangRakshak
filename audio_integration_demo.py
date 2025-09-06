import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class SangRakshakAudioIntegration:
    """SangRakshak Audio + Video Integration Demo"""
    
    def __init__(self):
        self.classes = ['normal', 'scream_distress', 'aggressive', 'emergency', 'silent']
        self.sample_rate = 22050
        
        # Load audio model
        try:
            self.audio_model = tf.keras.models.load_model("audio_detection_model.h5")
            print("✅ Audio model loaded!")
        except:
            print("❌ Audio model not found")
            self.audio_model = None
            
        # Threat log
        self.threat_log = []
    
    def extract_audio_features(self, audio_data):
        """Extract features from audio"""
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            rms = librosa.feature.rms(y=audio_data)
            
            features = []
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            features.append(np.mean(spectral_centroid))
            features.append(np.mean(spectral_bandwidth))
            features.append(np.mean(spectral_rolloff))
            features.append(np.mean(zcr))
            features.append(np.mean(rms))
            
            features = np.array(features)
            if len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)))
            else:
                features = features[:50]
                
            return features
        except Exception as e:
            return None
    
    def analyze_audio(self, audio_file):
        """Analyze uploaded audio file"""
        if audio_file is None:
            return "No audio file provided", None
        
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            if len(audio_data) == 0:
                return "Empty audio file", None
            
            # Extract features
            features = self.extract_audio_features(audio_data)
            if features is None:
                return "Error extracting audio features", None
            
            # Make prediction
            if self.audio_model:
                features = features.reshape(1, -1)
                prediction = self.audio_model.predict(features, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_class = self.classes[class_idx]
                
                # Check if threat
                threat_classes = ['scream_distress', 'aggressive', 'emergency']
                is_threat = predicted_class in threat_classes and confidence > 0.7
                
                # Create analysis result
                result = f"""
🎧 AUDIO ANALYSIS RESULTS
========================

📊 Prediction: {predicted_class.upper()}
📈 Confidence: {confidence:.3f}
🚨 Threat Level: {'HIGH ALERT' if is_threat else 'SAFE'}

📋 All Predictions:
"""
                for i, (class_name, prob) in enumerate(zip(self.classes, prediction[0])):
                    result += f"   {class_name}: {prob:.3f}\\n"
                
                if is_threat:
                    result += f"""
🚨 EMERGENCY ALERT TRIGGERED!
Threat Type: {predicted_class}
Recommended Actions:
- Alert security personnel
- Contact emergency services
- Start video recording
- Send notifications to emergency contacts
"""
                    
                    # Log threat
                    threat_info = {
                        'type': 'audio',
                        'class': predicted_class,
                        'confidence': confidence,
                        'timestamp': 'Now'
                    }
                    self.threat_log.append(threat_info)
                
                # Create visualization
                visualization = self.create_audio_visualization(audio_data, predicted_class)
                
                return result, visualization
            else:
                return "Audio model not available", None
                
        except Exception as e:
            return f"Error analyzing audio: {str(e)}", None
    
    def create_audio_visualization(self, audio_data, prediction):
        """Create audio visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Audio Analysis - Predicted: {prediction.upper()}', fontsize=16)
            
            # Waveform
            time = np.linspace(0, len(audio_data)/self.sample_rate, len(audio_data))
            axes[0,0].plot(time, audio_data)
            axes[0,0].set_title('Waveform')
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Amplitude')
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
            img = librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='hz', ax=axes[0,1])
            axes[0,1].set_title('Spectrogram')
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            img2 = librosa.display.specshow(mfccs, sr=self.sample_rate, x_axis='time', ax=axes[1,0])
            axes[1,0].set_title('MFCC Features')
            
            # Frequency spectrum
            fft_data = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            magnitude = np.abs(fft_data)
            axes[1,1].plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
            axes[1,1].set_title('Frequency Spectrum')
            axes[1,1].set_xlabel('Frequency (Hz)')
            axes[1,1].set_ylabel('Magnitude')
            
            plt.tight_layout()
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close()
            
            return img
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None
    
    def get_threat_history(self):
        """Get threat history"""
        if not self.threat_log:
            return "No threats detected yet."
        
        history = "🚨 THREAT HISTORY\\n" + "=" * 30 + "\\n"
        for i, threat in enumerate(self.threat_log[-5:], 1):  # Last 5 threats
            history += f"{i}. {threat['type'].upper()}: {threat['class']} ({threat['confidence']:.3f}) - {threat['timestamp']}\\n"
        
        return history
    
    def simulate_integration_demo(self):
        """Simulate full integration demo"""
        demo_result = """
🛡️ SANGRAKSHAK FULL INTEGRATION DEMO
=====================================

✅ SYSTEMS ACTIVE:
📹 Video Analysis: READY
🎧 Audio Detection: READY  
🔐 Encryption: ACTIVE
📡 Real-time Monitoring: READY

🎯 DETECTION CAPABILITIES:
• Gender Classification
• Violence Detection  
• Crowd Monitoring
• Audio Threat Detection
• Emergency Sound Recognition

🚨 ALERT SYSTEMS:
• Real-time Threat Notifications
• Emergency Contact Alerts
• Security Personnel Notification
• Automatic Incident Logging

📊 CURRENT STATUS: ALL SYSTEMS OPERATIONAL
"""
        return demo_result

# Initialize the system
system = SangRakshakAudioIntegration()

# Create Gradio interface
with gr.Blocks(title="🛡️ SangRakshak Audio Integration Demo") as demo:
    gr.HTML("""
        <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🛡️ संरक्षक</h1>
            <h2 style="color: white; margin: 10px 0; font-size: 1.5em;">Audio Detection Integration Demo</h2>
            <p style="color: white; margin: 0;">Real-time Audio Threat Detection for Women's Safety</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🎧 Audio Threat Detection")
            audio_input = gr.Audio(label="Upload Audio File", type="filepath")
            analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary")
            
        with gr.Column():
            gr.Markdown("## 📊 System Status")
            status_btn = gr.Button("📋 Show System Status", variant="secondary")
            history_btn = gr.Button("📜 View Threat History", variant="secondary")
    
    with gr.Row():
        with gr.Column():
            analysis_output = gr.Textbox(
                label="🎯 Analysis Results",
                lines=15,
                placeholder="Upload an audio file and click 'Analyze Audio'"
            )
        
        with gr.Column():
            visualization_output = gr.Image(
                label="📈 Audio Visualization",
                type="pil"
            )
    
    with gr.Row():
        system_output = gr.Textbox(
            label="🖥️ System Information",
            lines=8,
            placeholder="Click buttons above to see system status"
        )
    
    # Event handlers
    analyze_btn.click(
        fn=system.analyze_audio,
        inputs=[audio_input],
        outputs=[analysis_output, visualization_output]
    )
    
    status_btn.click(
        fn=system.simulate_integration_demo,
        inputs=[],
        outputs=[system_output]
    )
    
    history_btn.click(
        fn=system.get_threat_history,
        inputs=[],
        outputs=[system_output]
    )
    
    gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
            <h3>🎯 Audio Detection Features:</h3>
            <p>🔊 Scream Detection • 🗣️ Aggressive Voice Recognition • 🚨 Emergency Sounds • 🔇 Unusual Silence Detection</p>
            <h3>🔗 Integration Capabilities:</h3>
            <p>📹 Video + Audio Analysis • 🚨 Real-time Alerts • 📊 Threat Logging • 🔐 Encrypted Processing</p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)
    print("🚀 SangRakshak Audio Integration Demo is running at http://127.0.0.1:7860")
