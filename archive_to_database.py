import os
import shutil
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = r"D:\Business\Freelancer\Webscraper+Datawrangling"
DATABASE_DIR = os.path.join(DATA_DIR, "database")

def copy_and_archive_files():
    # 1. Ensure the 'database' folder exists
    os.makedirs(DATABASE_DIR, exist_ok=True)
    
    # 2. Get today's date in _DDMMYYYY format (TT = Day, MM = Month, YYYY = Year)
    date_str = datetime.now().strftime("%d%m%Y")
    
    copied_count = 0
    
    # 3. Scan the main directory for JSON and CSV files
    print(f"Scanning directory: {DATA_DIR}")
    for filename in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, filename)
        
        # Skip subdirectories (like the 'database' folder itself or '.venv')
        if os.path.isdir(full_path):
            continue
            
        # Check for .json or .csv extensions
        if filename.lower().endswith(('.json', '.csv')):
            # Separate filename and extension (e.g., 'combined_output' and '.csv')
            name, ext = os.path.splitext(filename)
            
            # Construct new filename with the date suffix
            new_filename = f"{name}_{date_str}{ext}"
            target_path = os.path.join(DATABASE_DIR, new_filename)
            
            # Copy the file (shutil.copy2 preserves metadata like modification timestamps)
            shutil.copy2(full_path, target_path)
            print(f"  📋 Copied & Renamed: {filename}  ==>  database/{new_filename}")
            copied_count += 1
            
    print(f"\n✅ Done! Successfully copied and archived {copied_count} files into the 'database' folder.")

if __name__ == "__main__":
    copy_and_archive_files()