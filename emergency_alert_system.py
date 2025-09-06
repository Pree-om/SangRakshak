#!/usr/bin/env python3
"""
📞 SangRakshak Emergency Alert System
Emergency notification system with SMS/email alerts, threat logging, and escalation protocols
"""

import smtplib
import json
import os
import time
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import threading
import queue

# For SMS (using email-to-SMS gateways)
SMS_GATEWAYS = {
    'verizon': '@vtext.com',
    'tmobile': '@tmomail.net',
    'att': '@txt.att.net',
    'sprint': '@messaging.sprintpcs.com',
    'boost': '@myboostmobile.com',
    'cricket': '@sms.cricketwireless.net',
    'uscellular': '@email.uscc.net'
}

class EmergencyAlertSystem:
    """Comprehensive emergency alert and notification system"""
    
    def __init__(self, config_file="alert_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        
        # Alert queues and threading
        self.alert_queue = queue.Queue()
        self.is_running = False
        self.alert_thread = None
        
        # Threat tracking
        self.active_threats = []
        self.alert_history = []
        self.escalation_levels = {
            'low': {'delay': 300, 'recipients': 'contacts'},      # 5 min
            'medium': {'delay': 120, 'recipients': 'emergency'},  # 2 min
            'high': {'delay': 30, 'recipients': 'all'},           # 30 sec
            'critical': {'delay': 0, 'recipients': 'all'}         # immediate
        }
        
        # Email setup
        self.smtp_server = None
        
        print(f"📞 Emergency Alert System initialized")
        print(f"📧 Email configured: {self.config.get('email', {}).get('enabled', False)}")
        print(f"📱 SMS configured: {self.config.get('sms', {}).get('enabled', False)}")
    
    def load_config(self):
        """Load alert configuration"""
        default_config = {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your_email@gmail.com",
                "password": "your_app_password",
                "sender_name": "SangRakshak Alert System"
            },
            "sms": {
                "enabled": True,
                "default_carrier": "verizon"
            },
            "contacts": {
                "emergency": [
                    {
                        "name": "Emergency Contact 1",
                        "email": "emergency1@example.com",
                        "phone": "1234567890",
                        "carrier": "verizon",
                        "role": "primary"
                    }
                ],
                "security": [
                    {
                        "name": "Security Team",
                        "email": "security@example.com",
                        "phone": "0987654321",
                        "carrier": "tmobile",
                        "role": "security"
                    }
                ],
                "family": [
                    {
                        "name": "Family Member",
                        "email": "family@example.com",
                        "phone": "5555555555",
                        "carrier": "att",
                        "role": "family"
                    }
                ]
            },
            "escalation": {
                "auto_escalate": True,
                "escalation_delay": 180,  # 3 minutes
                "max_escalation_level": "critical"
            },
            "logging": {
                "log_file": "emergency_alerts.log",
                "save_attachments": True,
                "attachment_dir": "alert_attachments"
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                # Create default config file
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                print(f"📝 Created default config file: {self.config_file}")
                print("⚠️ Please update the configuration with your email and contact details")
                return default_config
                
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return default_config
    
    def setup_email(self):
        """Setup SMTP connection for email"""
        if not self.config.get('email', {}).get('enabled', False):
            return False
        
        try:
            email_config = self.config['email']
            self.smtp_server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            self.smtp_server.starttls()
            self.smtp_server.login(email_config['username'], email_config['password'])
            print("✅ Email SMTP connection established")
            return True
            
        except Exception as e:
            print(f"❌ Email setup failed: {e}")
            self.smtp_server = None
            return False
    
    def send_email_alert(self, recipients, subject, message, attachments=None):
        """Send email alert"""
        if not self.config.get('email', {}).get('enabled', False):
            print("📧 Email alerts disabled")
            return False
        
        if not self.smtp_server:
            if not self.setup_email():
                return False
        
        try:
            email_config = self.config['email']
            sender_email = email_config['username']
            sender_name = email_config.get('sender_name', 'SangRakshak Alert')
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = f"{sender_name} <{sender_email}>"
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MimeText(message, 'html'))
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, "rb") as attachment:
                                part = MimeBase('application', 'octet-stream')
                                part.set_payload(attachment.read())
                                encoders.encode_base64(part)
                                part.add_header(
                                    'Content-Disposition',
                                    f'attachment; filename= {os.path.basename(file_path)}'
                                )
                                msg.attach(part)
                        except Exception as e:
                            print(f"⚠️ Could not attach {file_path}: {e}")
            
            # Send to each recipient
            sent_count = 0
            for recipient in recipients:
                try:
                    msg['To'] = recipient['email']
                    self.smtp_server.send_message(msg)
                    sent_count += 1
                    print(f"📧 Email sent to {recipient['name']} ({recipient['email']})")
                except Exception as e:
                    print(f"❌ Failed to send email to {recipient['name']}: {e}")
                finally:
                    del msg['To']  # Remove for next recipient
            
            return sent_count > 0
            
        except Exception as e:
            print(f"❌ Email sending failed: {e}")
            return False
    
    def send_sms_alert(self, recipients, message):
        """Send SMS alert using email-to-SMS gateway"""
        if not self.config.get('sms', {}).get('enabled', False):
            print("📱 SMS alerts disabled")
            return False
        
        if not self.smtp_server:
            if not self.setup_email():
                return False
        
        try:
            email_config = self.config['email']
            sender_email = email_config['username']
            
            sent_count = 0
            for recipient in recipients:
                if 'phone' not in recipient or not recipient['phone']:
                    continue
                
                # Get carrier gateway
                carrier = recipient.get('carrier', self.config['sms'].get('default_carrier', 'verizon'))
                gateway = SMS_GATEWAYS.get(carrier.lower())
                
                if not gateway:
                    print(f"⚠️ Unknown carrier {carrier} for {recipient['name']}")
                    continue
                
                # Create SMS address
                sms_address = f"{recipient['phone']}{gateway}"
                
                try:
                    # Create SMS message
                    msg = MimeText(message)
                    msg['From'] = sender_email
                    msg['To'] = sms_address
                    msg['Subject'] = ""  # SMS doesn't need subject
                    
                    self.smtp_server.send_message(msg)
                    sent_count += 1
                    print(f"📱 SMS sent to {recipient['name']} ({recipient['phone']})")
                    
                except Exception as e:
                    print(f"❌ Failed to send SMS to {recipient['name']}: {e}")
            
            return sent_count > 0
            
        except Exception as e:
            print(f"❌ SMS sending failed: {e}")
            return False
    
    def get_recipients(self, alert_level, recipient_type='all'):
        """Get recipients based on alert level and type"""
        recipients = []
        
        contacts = self.config.get('contacts', {})
        
        if recipient_type in ['all', 'emergency']:
            recipients.extend(contacts.get('emergency', []))
        
        if recipient_type in ['all', 'security']:
            recipients.extend(contacts.get('security', []))
        
        if recipient_type in ['all', 'family'] and alert_level in ['high', 'critical']:
            recipients.extend(contacts.get('family', []))
        
        return recipients
    
    def create_alert_message(self, threat_info):
        """Create formatted alert message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        threat_type = threat_info.get('type', 'unknown')
        threat_class = threat_info.get('class', 'unknown')
        confidence = threat_info.get('confidence', 0.0)
        location = threat_info.get('location', 'Unknown location')
        
        # Email version (HTML)
        email_message = f"""
        <html>
        <head>
            <style>
                .alert-header {{ background-color: #d32f2f; color: white; padding: 15px; text-align: center; }}
                .alert-body {{ padding: 20px; font-family: Arial, sans-serif; }}
                .threat-details {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .action-items {{ background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .footer {{ font-size: 12px; color: #666; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h1>🚨 EMERGENCY ALERT - SangRakshak</h1>
                <h2>THREAT DETECTED</h2>
            </div>
            
            <div class="alert-body">
                <p><strong>Time:</strong> {timestamp}</p>
                
                <div class="threat-details">
                    <h3>🎯 Threat Details:</h3>
                    <ul>
                        <li><strong>Type:</strong> {threat_type.title()}</li>
                        <li><strong>Classification:</strong> {threat_class.replace('_', ' ').title()}</li>
                        <li><strong>Confidence:</strong> {confidence:.1%}</li>
                        <li><strong>Location:</strong> {location}</li>
                    </ul>
                </div>
                
                <div class="action-items">
                    <h3>📞 Immediate Actions Required:</h3>
                    <ul>
                        <li>🚔 Contact local emergency services if immediate danger</li>
                        <li>🏃‍♀️ Ensure personal safety - move to safe location</li>
                        <li>📱 Verify situation and respond to this alert</li>
                        <li>🔍 Check SangRakshak dashboard for more details</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>This alert was generated by SangRakshak AI Safety Detection System</p>
                    <p>If this is a false alarm, please update the system settings</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # SMS version (plain text, short)
        sms_message = f"""
🚨 SANGRAKSHAK ALERT 🚨
{timestamp}

THREAT: {threat_class.replace('_', ' ').upper()}
CONFIDENCE: {confidence:.0%}
LOCATION: {location}

⚠️ TAKE IMMEDIATE ACTION
📞 Call emergency if needed
🏃 Move to safety

Reply SAFE when secure
        """.strip()
        
        return email_message, sms_message
    
    def process_threat_alert(self, threat_info):
        """Process and send threat alert"""
        try:
            # Determine alert level
            confidence = threat_info.get('confidence', 0.0)
            threat_class = threat_info.get('class', '')
            
            if confidence >= 0.9 or threat_class in ['emergency', 'violence_detected']:
                alert_level = 'critical'
            elif confidence >= 0.8 or threat_class in ['aggressive', 'scream_distress']:
                alert_level = 'high'
            elif confidence >= 0.7:
                alert_level = 'medium'
            else:
                alert_level = 'low'
            
            # Get escalation info
            escalation = self.escalation_levels[alert_level]
            recipients = self.get_recipients(alert_level, escalation['recipients'])
            
            if not recipients:
                print("⚠️ No recipients configured for alerts")
                return False
            
            # Create messages
            email_message, sms_message = self.create_alert_message(threat_info)
            
            # Send alerts
            email_sent = self.send_email_alert(
                recipients,
                f"🚨 EMERGENCY ALERT - {threat_class.replace('_', ' ').title()} Detected",
                email_message
            )
            
            sms_sent = self.send_sms_alert(recipients, sms_message)
            
            # Log alert
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'threat_info': threat_info,
                'alert_level': alert_level,
                'recipients_count': len(recipients),
                'email_sent': email_sent,
                'sms_sent': sms_sent
            }
            
            self.alert_history.append(alert_record)
            self.save_alert_log(alert_record)
            
            print(f"✅ Alert sent - Level: {alert_level}, Recipients: {len(recipients)}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing threat alert: {e}")
            return False
    
    def save_alert_log(self, alert_record):
        """Save alert to log file"""
        try:
            log_config = self.config.get('logging', {})
            log_file = log_config.get('log_file', 'emergency_alerts.log')
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(alert_record) + "\\n")
                
        except Exception as e:
            print(f"❌ Error saving alert log: {e}")
    
    def start_alert_service(self):
        """Start the alert processing service"""
        if self.is_running:
            print("⚠️ Alert service already running")
            return
        
        print("📞 Starting emergency alert service...")
        self.is_running = True
        
        def process_alerts():
            while self.is_running:
                try:
                    # Get alert from queue (blocking with timeout)
                    threat_info = self.alert_queue.get(timeout=1)
                    self.process_threat_alert(threat_info)
                    self.alert_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ Error in alert processing: {e}")
        
        self.alert_thread = threading.Thread(target=process_alerts)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        print("✅ Emergency alert service started")
    
    def stop_alert_service(self):
        """Stop the alert processing service"""
        print("🛑 Stopping emergency alert service...")
        self.is_running = False
        
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=2)
        
        if self.smtp_server:
            try:
                self.smtp_server.quit()
            except:
                pass
            self.smtp_server = None
        
        print("✅ Emergency alert service stopped")
    
    def queue_threat_alert(self, threat_info):
        """Queue a threat for alert processing"""
        try:
            self.alert_queue.put(threat_info, block=False)
            print(f"📥 Threat queued for alert: {threat_info.get('class', 'unknown')}")
        except queue.Full:
            print("⚠️ Alert queue full - dropping alert")
    
    def send_test_alert(self):
        """Send test alert to verify configuration"""
        print("🧪 Sending test alert...")
        
        test_threat = {
            'type': 'test',
            'class': 'system_test',
            'confidence': 0.95,
            'location': 'Test Location',
            'timestamp': datetime.now().isoformat(),
            'details': 'This is a test alert to verify system configuration'
        }
        
        return self.process_threat_alert(test_threat)
    
    def get_alert_statistics(self):
        """Get alert system statistics"""
        if not self.alert_history:
            return "No alerts sent yet"
        
        total_alerts = len(self.alert_history)
        email_success = sum(1 for alert in self.alert_history if alert.get('email_sent', False))
        sms_success = sum(1 for alert in self.alert_history if alert.get('sms_sent', False))
        
        recent_alerts = [alert for alert in self.alert_history 
                        if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(hours=24)]
        
        stats = f"""
📊 Alert System Statistics:
  • Total alerts sent: {total_alerts}
  • Email success rate: {email_success}/{total_alerts} ({email_success/total_alerts*100:.1f}%)
  • SMS success rate: {sms_success}/{total_alerts} ({sms_success/total_alerts*100:.1f}%)
  • Alerts in last 24h: {len(recent_alerts)}
  • Service status: {'Running' if self.is_running else 'Stopped'}
        """.strip()
        
        return stats

# ===== DEMO FUNCTIONS =====
def demo_alert_system():
    """Demo the emergency alert system"""
    print("📞 SangRakshak Emergency Alert System Demo")
    print("=" * 50)
    
    # Create alert system
    alert_system = EmergencyAlertSystem()
    
    # Start service
    alert_system.start_alert_service()
    
    try:
        # Send test alert
        print("\\n🧪 Testing alert system configuration...")
        if alert_system.send_test_alert():
            print("✅ Test alert sent successfully!")
        else:
            print("❌ Test alert failed - check configuration")
        
        # Simulate some threats
        print("\\n🎭 Simulating threat detection...")
        
        test_threats = [
            {
                'type': 'audio',
                'class': 'scream_distress',
                'confidence': 0.85,
                'location': 'Main Entrance',
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'video',
                'class': 'violence_detected', 
                'confidence': 0.92,
                'location': 'Parking Area',
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'audio',
                'class': 'emergency',
                'confidence': 0.95,
                'location': 'Building Lobby',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        for i, threat in enumerate(test_threats, 1):
            print(f"\\n📥 Queuing threat {i}: {threat['class']}")
            alert_system.queue_threat_alert(threat)
            time.sleep(2)  # Wait between threats
        
        # Wait for processing
        print("\\n⏳ Processing alerts...")
        time.sleep(5)
        
        # Show statistics
        print("\\n" + alert_system.get_alert_statistics())
        
    except KeyboardInterrupt:
        print("\\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo error: {e}")
    finally:
        alert_system.stop_alert_service()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_alert_system()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        alert_system = EmergencyAlertSystem()
        alert_system.start_alert_service()
        alert_system.send_test_alert()
        alert_system.stop_alert_service()
    else:
        print("📞 SangRakshak Emergency Alert System")
        print("=" * 40)
        print("Usage:")
        print("  python emergency_alert_system.py --demo    # Run full demo")
        print("  python emergency_alert_system.py --test    # Send test alert")
        print("\\nConfiguration file: alert_config.json")
        print("Please update with your email and contact details before use.")
