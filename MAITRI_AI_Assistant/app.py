import time
import threading
import yaml
import json
import os
import html
from flask import Flask, render_template, jsonify, request

# Import Core Modules
from core.offline_utils import OfflineUtils
from core.input_manager import InputManager
from core.fusion_engine import MultimodalFusionEngine

# Import AI Modules
from modules.emotion_detection.facial_emotion import FacialEmotionDetector
from modules.emotion_detection.speech_emotion import SpeechEmotionDetector
from modules.conversation.dialogue_manager import DialogueManager
from modules.conversation.tts_offline import TTSOffline
from modules.health_monitoring.fatigue_detector import FatigueDetector
from modules.health_monitoring.stress_analyzer import StressAnalyzer
from modules.health_monitoring.recommendations import Recommendations
from modules.reporting.alert_manager import AlertManager
from modules.reporting.report_generator import ReportGenerator
from modules.reporting.ground_sync import GroundSync

from core.constants import EMOTION_LABELS

app = Flask(__name__, template_folder='ui/templates', static_folder='ui/static')

# --- Global Initialization ---
CONFIG_PATH = 'config/system_config.yaml'
THRESHOLDS_PATH = 'config/thresholds.yaml'

try:
    with open(CONFIG_PATH, 'r') as f:
        SYSTEM_CONFIG = yaml.safe_load(f)
    with open(THRESHOLDS_PATH, 'r') as f:
        THRESHOLDS = yaml.safe_load(f)
except FileNotFoundError as e:
    print(f"FATAL ERROR: Configuration file not found: {e}")
    exit(1)

# Initialize Components
offline_utils = OfflineUtils()
input_manager = InputManager(SYSTEM_CONFIG)
fusion_engine = MultimodalFusionEngine(THRESHOLDS)

facial_detector = FacialEmotionDetector(offline_utils)
speech_detector = SpeechEmotionDetector(offline_utils)

# Add safe defaults for configuration paths
dialogue_manager = DialogueManager(
    SYSTEM_CONFIG.get('response_library_path', 'modules/conversation/response_library.json')
)
tts_engine = TTSOffline()

# Updated to pass thresholds
fatigue_detector = FatigueDetector(THRESHOLDS) 
stress_analyzer = StressAnalyzer(THRESHOLDS)
recommendations = Recommendations()

alert_manager = AlertManager(THRESHOLDS)
report_generator = ReportGenerator(
    SYSTEM_CONFIG.get('log_directory', 'data/logs/'), 
    SYSTEM_CONFIG.get('report_format', 'json')        
)
ground_sync = GroundSync(SYSTEM_CONFIG.get('log_directory', 'data/logs/')) 

# Global State for UI
current_state = {
    'fused_emotion': 'Neutral',
    'fused_score': 0.5,
    'stress_index': 0.0,
    'is_fatigued': False,
    'alert': None,
    'conversation_history': []
}

# Add lock for thread safety
state_lock = threading.Lock()
# Variable for better time-based sync control
last_sync_time = 0

# --- Background Processing Thread ---
def process_loop():
    global last_sync_time 
    """Continuously captures data, processes it, and updates global state."""
    while True:
        try:
            # 1. Capture Inputs
            frame = input_manager.capture_frame()
            audio = input_manager.capture_audio_chunk(duration_s=1)

            # 2. Emotion Detection
            facial_result = facial_detector.detect(frame)
            speech_result = speech_detector.detect(audio)

            # 3. Multimodal Fusion
            fused_result = fusion_engine.fuse(facial_result, speech_result)

            # 4. Health Monitoring
            fatigue_data = fatigue_detector.check_fatigue_cues(frame, audio)
            stress_data = stress_analyzer.analyze(fused_result, fatigue_data['score'])

            # 5. Alert Check
            alert_status = alert_manager.check_critical_state(
                stress_data['is_critically_stressed'], fused_result
            )

            # 6. Update Global State
            with state_lock:
                current_state.update({
                    'fused_emotion': fused_result['fused_emotion'],
                    'fused_score': fused_result['score'],
                    'stress_index': stress_data['current_index'],
                    'is_fatigued': fatigue_data['is_fatigued'],
                    'alert': alert_status if alert_status['alert_triggered'] else None
                })

                # This ensures the list never exceeds the target size + 1 (21 items) during the cycle.
                if len(current_state['conversation_history']) > 20:
                    current_state['conversation_history'] = current_state['conversation_history'][-20:]
            
            # 7. Logging (for reporting)
            report_generator.log_session_event('STATE_UPDATE', {
                'emotion': fused_result['fused_emotion'],
                'confidence': fused_result['score'],
                'stress_index': stress_data['current_index'],
                'is_fatigued': fatigue_data['is_fatigued'],
            })

            # 8. Ground Sync (runs periodically, not every cycle)
            current_time = time.time()
            if current_time - last_sync_time >= 60:
                ground_sync.sync_reports()
                last_sync_time = current_time

            time.sleep(1) # Run analysis every 1 second

        except Exception as e:
            print(f"Error in processing loop: {e}")
            time.sleep(5)

# Start the background processing thread
processing_thread = threading.Thread(target=process_loop, daemon=True)
processing_thread.start()

# --- Flask Routes ---

@app.route('/')
def index():
    """Default route renders the main dashboard."""
    return render_template('dashboard.html', config=SYSTEM_CONFIG)

@app.route('/api/state', methods=['GET'])
def get_current_state():
    """API endpoint to fetch the current emotional and health state."""
    # We use dict() here to safely return a copy of the dictionary
    return jsonify(dict(current_state)) 

@app.route('/api/conversation', methods=['POST'])
def handle_conversation():
    """Handles user input and generates an AI response."""
    data = request.json
    user_message = data.get('message', '').strip()

    # Sanitize HTML to prevent XSS
    user_message = html.escape(user_message)
    
    # Basic input validation and max length check
    MAX_LENGTH = 300
    if not user_message:
        return jsonify({'response': "Please type something to start the conversation."})

    if len(user_message) > MAX_LENGTH:
        return jsonify({'response': f"Your message is too long (max {MAX_LENGTH} characters)."})

    # 1. Get current emotional context
    emotion = current_state['fused_emotion']
    is_stressed = current_state['stress_index'] > THRESHOLDS.get('stress_level_high', 0.65) 


    # 2. Generate MAITRI's response
    maitri_response = dialogue_manager.generate_response(emotion, is_stressed)
    
    # 3. Simulate TTS output (prints to console)
    tts_engine.speak(maitri_response)

    # 4. Generate recommendation if necessary
    tip = recommendations.get_tip('stress' if is_stressed else emotion)

    # 5. Update history
    with state_lock:
        current_state['conversation_history'].append({'speaker': 'user', 'text': user_message})
        current_state['conversation_history'].append({'speaker': 'maitri', 'text': maitri_response})
    
        # Truncate history to prevent excessive growth
        if len(current_state['conversation_history']) > 20:
            current_state['conversation_history'] = current_state['conversation_history'][-20:]

    return jsonify({
        'response': maitri_response,
        'tip': tip,
        'history': current_state['conversation_history']
    })

@app.route('/monitoring')
def monitoring_page():
    """Renders the monitoring and historical log page."""
    return render_template('monitoring.html')

@app.route('/alert_screen')
def alert_screen():
    """Renders the critical alert screen."""
    return render_template('critical_alert.html')

if __name__ == '__main__':
    print(f"Starting {SYSTEM_CONFIG['app_name']} v{SYSTEM_CONFIG['version']}...")
    
    if SYSTEM_CONFIG.get('sync_on_startup', False):
        print("Initial ground sync initiated...")
        ground_sync.sync_reports()
    
    app.run(debug=SYSTEM_CONFIG['debug_mode'], port=5000)
