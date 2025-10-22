import torch
import os
import yaml

class OfflineUtils:
    """
    Handles loading of local AI models and configuration files.
    Ensures the application remains functional without external dependencies.
    """
    def __init__(self, config_path='./config/system_config.yaml'):
        # Load core configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_dir = self.config.get('model_directory', './data/models/')
        print(f"OfflineUtils initialized. Looking for models in: {self.model_dir}")

    def load_model(self, model_name):
        """
        Loads a PyTorch model from the specified directory.
        """
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found at {model_path}. Returning mock model.")
            # In a real scenario, this would raise an error. Here we return a placeholder.
            return None

        try:
            # Placeholder: Use map_location to ensure compatibility
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            print(f"SUCCESS: Model '{model_name}' loaded successfully.")
            return model
        except Exception as e:
            print(f"ERROR: Failed to load model {model_name}. Details: {e}")
            return None

    def get_config_value(self, key):
        """Retrieves a configuration value from the loaded system config."""
        return self.config.get(key)