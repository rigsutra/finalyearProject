# import os

# def count_images_by_label(base_dir):
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
#     class_counts = {}

#     for root, dirs, files in os.walk(base_dir):
#         for dir_name in dirs:
#             class_path = os.path.join(root, dir_name)
#             image_files = [
#                 file for file in os.listdir(class_path)
#                 if file.lower().endswith(image_extensions)
#             ]
#             class_counts[dir_name] = len(image_files)

#     return class_counts

# # === CONFIG ===
# base_folder = r"E:\college final year project\new_traning_data"  # Replace with your actual dataset path

# # === RUN ===
# counts = count_images_by_label(base_folder)

# print("\n[📊 Image Count Per Class]")
# for class_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
#     print(f"{class_name}: {count}")


import os
from collections import Counter

def count_images_by_keyword(folder_path, keywords):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    counts = Counter()

    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            for keyword in keywords:
                if keyword.lower() in file.lower():
                    counts[keyword] += 1
                    break  # Don't double count if multiple keywords match

    return counts

# === CONFIG ===
folder = r"E:\college final year project\new_traning_data\Hair"  # Change this to your actual path
keywords = [
    "vitiligo",
    "melasma",
    "telogen-effluvium",
    "anagen-effluvium",
    "trichorrhexis-nodosa",
    "hirsutism",
    "alopecia-areata"
]

# === RUN ===
counts = count_images_by_keyword(folder, keywords)

print("\n=== Image Count Based on Filename Keywords ===")

for keyword, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{keyword}: {count}")
