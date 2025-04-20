import os
import shutil

def filter_images_by_keywords(source_dir, dest_dir, keywords):
    """
    Copy images from source_dir to dest_dir if their filenames contain any of the keywords.
    
    :param source_dir: Path to the folder containing original images
    :param dest_dir: Path to the folder where filtered images will be stored
    :param keywords: List of strings to match in filenames
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    count = 0
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            if any(keyword.lower() in filename.lower() for keyword in keywords):
                shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
                count += 1

    print(f"[✔] Copied {count} matching images to '{dest_dir}'")

# === CONFIGURATION ===
source_folder = r"train\Hair Loss Photos Alopecia and other Hair Diseases"  # Path to the folder containing original images
destination_folder = "new_traning_data/Hair"  # Path to the folder where filtered images will be stored
keywords_to_match = [
    "telogen-effluvium",
    "anagen-effluvium",
    "trichorrhexis-nodosa",
    "hirsutism"
]
  # Modify with your required filters

filter_images_by_keywords(source_folder, destination_folder, keywords_to_match)
