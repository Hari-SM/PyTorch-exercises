import os
import shutil

src_dir = "train"
cat_dir = "data/cat"
dog_dir = "data/dog"

# Make directories
os.makedirs(cat_dir, exist_ok=True)
os.makedirs(dog_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.startswith("cat"):
        new_filename = filename.replace("cat.", "")
        shutil.move(os.path.join(src_dir, filename), os.path.join(cat_dir, new_filename))
    if filename.startswith("dog"):
        new_filename = filename.replace("dog.", "")
        shutil.move(os.path.join(src_dir, filename), os.path.join(dog_dir, new_filename))
print("All files are moved")