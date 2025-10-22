import unittest
import time
from modules.reporting.alert_manager import AlertManager

class TestAlertManager(unittest.TestCase):
    def setUp(self):
        # Mock thresholds for testing
        self.thresholds = {
            'min_confidence': 0.70,
            'emotion_specific': {
                'anger': 0.90,
                'fear': 0.90
            }
        }
        self.manager = AlertManager(self.thresholds)
        self.manager.alert_cooldown_s = 5 # Shorten cooldown for testing

    def test_critical_stress_alert(self):
        """Test if an alert is triggered by critical stress."""
        result = self.manager.check_critical_state(
            is_critically_stressed=True, 
            fused_emotion={'fused_emotion': 'neutral', 'score': 0.5}
        )
        self.assertTrue(result['alert_triggered'])
        self.assertIn('CRITICAL STRESS ALERT', result['message'])

    def test_acute_emotion_alert(self):
        """Test if an alert is triggered by high-confidence acute emotion."""
        result = self.manager.check_critical_state(
            is_critically_stressed=False, 
            fused_emotion={'fused_emotion': 'anger', 'score': 0.95}
        )
        self.assertTrue(result['alert_triggered'])
        self.assertIn('ACUTE ANGER DETECTED', result['message'])

    def test_cooldown_mechanism(self):
        """Test if the cooldown prevents immediate re-alerting."""
        # 1. Trigger first alert
        first_alert = self.manager.check_critical_state(
            is_critically_stressed=True, 
            fused_emotion={'fused_emotion': 'neutral', 'score': 0.5}
        )
        self.assertTrue(first_alert['alert_triggered'])

        # 2. Immediately try to trigger again (should fail due to cooldown=5s)
        second_alert = self.manager.check_critical_state(
            is_critically_stressed=True, 
            fused_emotion={'fused_emotion': 'neutral', 'score': 0.5}
        )
        self.assertFalse(second_alert['alert_triggered'])

        # 3. Simulate passing the cooldown period
        time.sleep(self.manager.alert_cooldown_s + 0.1) 
        
        third_alert = self.manager.check_critical_state(
            is_critically_stressed=True, 
            fused_emotion={'fused_emotion': 'neutral', 'score': 0.5}
        )
        self.assertTrue(third_alert['alert_triggered'])

if __name__ == '__main__':
    unittest.main()
