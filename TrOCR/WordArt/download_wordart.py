#!/usr/bin/env python3
"""
Script to download WordArt dataset from Google Drive and extract it to the current WordArt folder.
Run this script from within the WordArt directory.
"""

import os
import zipfile
import shutil
import subprocess
import sys
from pathlib import Path

# Google Drive file ID extracted from the URL
FILE_ID = "1SanxRwTxd2q7UrQxlbC3BmP3nhFXwZ3g"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# Get the current directory (WordArt folder)
CURRENT_DIR = Path(__file__).resolve().parent

def check_gdown():
    """Check if gdown is installed, install if not."""
    try:
        import gdown
        return True
    except ImportError:
        print("gdown not found. Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        return True

def download_file(url, output_path):
    """Download file from Google Drive using gdown."""
    import gdown
    print(f"Downloading from Google Drive...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Download completed: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file to target directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction completed to {extract_to}")
        return True
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def copy_contents(source_dir, target_dir):
    """Copy all contents from source directory to target directory."""
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return False
    
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Copying contents from {source_dir} to {target_dir}...")
    
    try:
        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            target_path = os.path.join(target_dir, item)
            
            if os.path.isdir(source_path):
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
        
        print(f"Copy completed to {target_dir}")
        return True
    except Exception as e:
        print(f"Error copying contents: {e}")
        return False

def main():
    """Main function to download and extract WordArt dataset."""
    print("=" * 60)
    print("WordArt Dataset Download Script")
    print("=" * 60)
    print(f"Target directory: {CURRENT_DIR}")
    print("=" * 60)
    
    # Check and install gdown if needed
    check_gdown()
    
    # Create a temporary directory for download
    temp_dir = CURRENT_DIR / "temp_download"
    os.makedirs(temp_dir, exist_ok=True)
    
    zip_path = temp_dir / "WordArt.zip"
    
    # Download the file
    if not download_file(DOWNLOAD_URL, str(zip_path)):
        print("Failed to download file. Exiting.")
        return
    
    if not zip_path.exists():
        print(f"Downloaded file not found: {zip_path}")
        return
    
    # Extract to temporary directory first
    temp_extract_dir = temp_dir / "extracted"
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    if not extract_zip(str(zip_path), str(temp_extract_dir)):
        print("Failed to extract zip file. Exiting.")
        return
    
    # Find the actual WordArt content (might be in a subdirectory)
    extracted_content = temp_extract_dir
    if (temp_extract_dir / "WordArt").exists():
        extracted_content = temp_extract_dir / "WordArt"
    
    # Copy to current directory (WordArt folder)
    if not copy_contents(str(extracted_content), str(CURRENT_DIR)):
        print(f"Warning: Failed to copy to {CURRENT_DIR}")
    
    # Clean up temporary files
    print("\nCleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir)
        print("Cleanup completed.")
    except Exception as e:
        print(f"Warning: Could not clean up temp directory: {e}")
    
    print("\n" + "=" * 60)
    print("WordArt dataset download and extraction completed!")
    print("=" * 60)
    print(f"\nFiles are available in: {CURRENT_DIR}")

if __name__ == "__main__":
    main()
