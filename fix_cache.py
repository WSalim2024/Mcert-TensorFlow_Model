import os
from pathlib import Path
import shutil

# The emnist library typically stores data in the user's home directory under .cache
cache_dir = Path.home() / ".cache" / "emnist"
zip_file = cache_dir / "emnist.zip"

print(f"Inspecting cache directory: {cache_dir}")

if zip_file.exists():
    size_kb = zip_file.stat().st_size / 1024
    print(f"Found cached file. Size: {size_kb:.2f} KB")

    if size_kb < 1000:  # If it's less than 1MB, it's definitely broken
        print("File is clearly corrupted (too small). Deleting now...")
        try:
            os.remove(zip_file)
            print("SUCCESS: Corrupted file deleted.")
            print("You can now run main.py again.")
        except PermissionError:
            print("ERROR: Could not delete the file. Please close any running python processes and try again.")
    else:
        print("The file size looks significant, but if it's failing, we should force delete it anyway.")
        os.remove(zip_file)
        print("Deleted existing cache to force re-download.")
else:
    print("No cached 'emnist.zip' found in the default location.")
    print("Please check if your internet connection is blocking the download.")