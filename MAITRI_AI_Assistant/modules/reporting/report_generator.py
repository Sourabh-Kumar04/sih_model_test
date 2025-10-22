import json
import time
import os

class ReportGenerator:
    """
    Generates chronological logs and summary reports (JSON/PDF).
    """
    def __init__(self, log_dir='./data/logs/', report_format='json'):
        self.log_dir = log_dir
        self.report_format = report_format
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"ReportGenerator initialized. Logs saved to: {self.log_dir}")

    def log_session_event(self, event_type, data):
        """Logs a single event data point to a chronological JSON file."""
        log_entry = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'type': event_type,
            'data': data
        }
        
        # Use a daily log file
        log_file = os.path.join(self.log_dir, f"{time.strftime('%Y-%m-%d')}_session_log.json")

        try:
            if os.path.exists(log_file):
                with open(log_file, 'r+') as f:
                    file_data = json.load(f)
                    file_data.append(log_entry)
                    f.seek(0)
                    json.dump(file_data, f, indent=2)
            else:
                with open(log_file, 'w') as f:
                    json.dump([log_entry], f, indent=2)
            # print(f"Logged event: {event_type}")
        except Exception as e:
            print(f"ERROR: Failed to write log entry: {e}")


    def generate_summary_report(self, log_data):
        """
        Generates a summary report based on all gathered log data.
        """
        report_data = {
            "report_id": f"REPORT_{int(time.time())}",
            "date_generated": time.strftime("%Y-%m-%d"),
            "total_events": len(log_data),
            "max_stress_index": max(d['data'].get('stress_index', 0) for d in log_data if d['type'] == 'STATE_UPDATE'),
            # ... more complex analysis
            "recommendations": "Based on the data, focus on improving sleep hygiene and taking micro-breaks every hour."
        }
        
        # For simplicity, we only generate JSON here. PDF generation requires external libs.
        if self.report_format == 'json':
            report_name = os.path.join(self.log_dir, f"Summary_Report_{report_data['report_id']}.json")
            with open(report_name, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"Summary Report generated: {report_name}")
            return report_data
        
        # Placeholder for PDF generation
        # elif self.report_format == 'pdf':
        #     return "PDF generation simulated."

        return {}
