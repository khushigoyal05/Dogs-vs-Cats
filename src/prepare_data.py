import os
import shutil

SOURCE_FOLDER = "../data/raw/train/train"
DEST_FOLDER = "../data/train"

cats_folder = os.path.join(DEST_FOLDER, "cats")
dogs_folder = os.path.join(DEST_FOLDER, "dogs")

os.makedirs(cats_folder, exist_ok=True)
os.makedirs(dogs_folder, exist_ok=True)

for filename in os.listdir(SOURCE_FOLDER):
    if filename.startswith("cat"):
        shutil.copy(
            os.path.join(SOURCE_FOLDER, filename),
            os.path.join(cats_folder, filename)
        )
    elif filename.startswith("dog"):
        shutil.copy(
            os.path.join(SOURCE_FOLDER, filename),
            os.path.join(dogs_folder, filename)
        )

print("Dataset organized successfully!")

