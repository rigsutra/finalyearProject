# import os
# import re
# from collections import Counter

# def extract_keywords_from_filenames(folder_path, min_length=3):
#     all_words = []

#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
#             name_part = os.path.splitext(filename)[0]
#             words = re.split(r'[^a-zA-Z0-9]+', name_part)
#             filtered_words = [word.lower() for word in words if len(word) >= min_length]
#             all_words.extend(filtered_words)

#     keyword_counter = Counter(all_words)
#     return keyword_counter

# # === CONFIG ===
# image_folder_path = r"train\Light Diseases and Disorders of Pigmentation"  # OR use forward slashes

# keyword_counts = extract_keywords_from_filenames(image_folder_path)

# print("\n[Most Common Words in Filenames]")
# for word, count in keyword_counts.most_common(20):
#     print(f"{word}: {count}")





import os
import re
from collections import Counter

def clean_filename(name):
    # Remove trailing numbers and extra symbols
    name = re.sub(r'[\d_]+$', '', name)  # Remove trailing digits and underscores
    return name.lower().strip()

def extract_fullnames(folder_path):
    name_counter = Counter()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            base_name = os.path.splitext(filename)[0]
            clean_name = clean_filename(base_name)
            name_counter[clean_name] += 1

    return name_counter

# === CONFIG ===
image_folder_path = r"E:\college final year project\new_traning_data\Rosacea"  # Use raw string or forward slashes

# === RUN ===
counts = extract_fullnames(image_folder_path)

print("\n[Most Common Base Names in Filenames]")
for name, count in counts.most_common(20):
    print(f"{name}: {count}")
