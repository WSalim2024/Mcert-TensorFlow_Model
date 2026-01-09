import os
from pathlib import Path

# 1. Define the hidden path where the library looks for data
cache_dir = Path.home() / ".cache" / "emnist"

# 2. Create the folder if it doesn't exist
cache_dir.mkdir(parents=True, exist_ok=True)

# 3. Open the folder in Windows Explorer
print(f"Opening folder: {cache_dir}")
os.startfile(cache_dir)