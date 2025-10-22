import time
import os

class GroundSync:
    """
    Manages synchronization of reports/logs when communication links become available.
    Designed for offline-first deployments.
    """
    def __init__(self, log_dir='./data/logs/'):
        self.log_dir = log_dir
        self.is_connected = False # Simulate connection status
        self.last_sync_time = 0

    def check_connection(self):
        """Simulates checking for an available internet connection."""
        # This logic would be replaced by actual network status checks (e.g., ping)
        if time.time() % 10 < 5: # Toggle connection every 10 seconds for simulation
            self.is_connected = True
        else:
            self.is_connected = False
        return self.is_connected

    def sync_reports(self):
        """Sends unsynced reports to a remote server when connected."""
        if not self.check_connection():
            # print("Ground Sync: Connection offline. Reports queued.")
            return False

        files_to_sync = [f for f in os.listdir(self.log_dir) if f.endswith('.json')]
        
        if not files_to_sync:
            # print("Ground Sync: Connected, but no reports to sync.")
            return True # Successfully connected, nothing to do

        synced_count = 0
        for filename in files_to_sync:
            filepath = os.path.join(self.log_dir, filename)
            # Simulate secure file upload
            # print(f"Ground Sync: Uploading {filename}...")
            time.sleep(0.01) # Simulate network delay
            
            # On successful upload, delete or move the file (simulating deletion here)
            # os.remove(filepath)
            synced_count += 1
            
        self.last_sync_time = time.time()
        print(f"Ground Sync: SUCCESS! Synced {synced_count} files.")
        return True
