#!/usr/bin/env python
import os
import csv
import time
from datetime import datetime

class Logger:
    def __init__(self, base_folder="."):
        # Get the current date and time
        self.start_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create a folder called "log" plus start_datetime in the predefined folder
        self.log_folder = os.path.join(base_folder, f"log_{self.start_datetime}")
        os.makedirs(self.log_folder, exist_ok=True)

    def write_to_file(self, file_name, content):
        """
        Writes or appends content with a timestamp to a file in the previously created folder.

        Parameters:
        - file_name (str): Name of the file to write to.
        - content (str): Content to write or append in the file.
        """
        timestamp = datetime.now().isoformat()
        file_path = os.path.join(self.log_folder, file_name)
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode) as f:
            f.write(f"{timestamp},{content}\n")
    def log_event(self, file_name, event_name, status, message=""):
        """
        Logs an event with a timestamp to a CSV file.

        Parameters:
        - file_name (str): Name of the CSV file to write to.
        - event_name (str): Name of the event.
        - status (str): "START","END", "ERROR"
        - message (str): Optional message
        """
        timestamp = datetime.now().isoformat()
        file_path = os.path.join(self.log_folder, file_name)

        # Check if file exists to decide header writing
        write_header = not os.path.exists(file_path)

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["EventName", "Timestamp", "Status","Message"])
            writer.writerow([event_name, timestamp, status, message])


# Usage example:
#logger = Logger("/home/toya/rkop_experiments/")
#logger.log_event("crampylog.csv", "experiment","START")
#logger.log_event("crampylog.csv", "human_detection","START")
#logger.log_event("crampylog.csv", "human_detection","END")
