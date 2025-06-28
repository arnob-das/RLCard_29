"""
File: file_header.py
Author: Arnob Das
Date: 2025-06-28
"""
import os
from datetime import datetime

# Configuration
author_name = "Arnob Das"
folder_path = "../RLCard_29"
extensions = [".py"]  # Add extensions as needed

def add_header_to_file(file_path):
    file_name = os.path.basename(file_path)
    today = datetime.now().strftime("%Y-%m-%d")
    header = f"""/*
    File: {file_name}
    Author: {author_name}
    Date: {today}
    */
    """
    with open(file_path, 'r+') as file:
        content = file.read()
        file.seek(0, 0)
        file.write(header + '\n' + content)

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if any(file.endswith(ext) for ext in extensions):
            add_header_to_file(os.path.join(root, file))
