```
MAITRI_AI_Assistant/
├── app.py                        # Main entry: integrates UI + AI models
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── .env                          # Environment vars (if needed)

├── config/
│   ├── thresholds.yaml            # Emotion detection sensitivity
│   └── system_config.yaml         # Offline runtime settings

├── data/
│   ├── models/                   # Trained AI models
│   │   ├── facial_emotion_model.pth
│   │   ├── speech_emotion_model.pth
│   │   └── multimodal_fusion.pth
│   ├── datasets/                 # Sample datasets (offline testing)
│   └── logs/                     # Emotion/health logs + reports

├── core/                         # Core system modules
│   ├── __init__.py
│   ├── input_manager.py          # Handle audio + video capture
│   ├── fusion_engine.py          # Combine audio-visual emotion results
│   └── offline_utils.py          # Ensure models run standalone

├── modules/                      # Functional AI modules
│   ├── emotion_detection/
│   │   ├── facial_emotion.py     # Detect facial expressions
│   │   ├── speech_emotion.py     # Detect voice tone/stress
│   │   └── feature_extraction.py # Feature engineering
│   ├── conversation/
│   │   ├── dialogue_manager.py   # Adaptive counseling
│   │   ├── response_library.json # Predefined supportive responses
│   │   └── tts_offline.py        # Text-to-speech (offline)
│   ├── health_monitoring/
│   │   ├── fatigue_detector.py   # Detect tiredness from audio/video cues
│   │   ├── stress_analyzer.py    # Stress/fatigue scoring
│   │   └── recommendations.py    # Tips (hydrate, sleep, stretch)
│   └── reporting/
│       ├── alert_manager.py      # Trigger critical alerts
│       ├── report_generator.py   # Create JSON/PDF logs
│       └── ground_sync.py        # Send reports when comms available

├── ui/                           # User interface
│   ├── templates/
│   │   ├── dashboard.html        # Emotional state + alerts
│   │   ├── conversation.html     # Chat interaction
│   │   ├── monitoring.html       # Stress/fatigue logs
│   │   └── critical_alert.html   # Emergency alert screen
│   ├── static/
│   │   ├── css/                  # Styling
│   │   └── js/                   # UI logic
│   └── wireframes/               # Wireframe images

├── tests/                        # Unit + integration testing
│   ├── test_facial_emotion.py
│   ├── test_speech_emotion.py
│   ├── test_dialogue_manager.py
│   ├── test_alerts.py
│   └── test_offline_system.py

└── docs/
    ├── problem_statement.pdf      # SIH official doc
    ├── design_architecture.md     # System design
    ├── user_manual.md             # Usage guide
    └── model_training_report.md   # AI model details
```