# MAITRI AI Assistant: System Design Architecture
## 1. Modularity Principle
The system follows a strict modular design, separating input/output (UI, InputManager) from core logic (Core, Modules), and configuration. This allows for:
- Offline Reliability: Core AI modules rely solely on local models and configurations.
- Easy Maintenance: Updates to the facial detection model do not affect the speech detection module or the Dialogue Manager.
- Scalability: New health monitoring modules (e.g., drowsiness detection) can be easily plugged in.

## 2. Data Flow (Sense-Process-Act Cycle)
1. Sense (Input Layer): InputManager captures raw video frames and audio chunks.
2. Process (Feature & AI Layer):
- Raw data feeds into FeatureExtractor.
- Features feed into FacialEmotionDetector and SpeechEmotionDetector.
- Results are sent to the MultimodalFusionEngine for a unified emotional state.

3. Analyze (Health Layer):
- Fused state and raw data feed into FatigueDetector and StressAnalyzer.
- AlertManager checks for critical thresholds.

4. Act (Output & Storage Layer):
- State updates and alerts feed the UI (dashboard.html).
- State updates guide the DialogueManager response, which is output via TTSOffline.
- All state changes are recorded by ReportGenerator.
- Reports are periodically synchronized by GroundSync when connectivity is available.

## 3. Technology Stack 
- Backend Core: Python (3.8+)
- Web Framework: Flask
- AI/ML: PyTorch (for model loading), NumPy, scikit-learn (for feature processing)
- Configuration: YAML
- Data Serialization: JSON