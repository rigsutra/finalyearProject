# # import os
# # import shutil
# # from collections import defaultdict

# # # Set your paths here
# # SOURCE_ROOT = r"E:\college final year project\new_traning_data"
# # DEST_ROOT = r"E:\college final year project\dataset_cleaned"

# # # Define vitamin-deficiency related conditions
# # VITAMIN_LABELS = [
# #     # Pigmentation
# #     "vitiligo",
# #     "melasma",
# #     "idiopathic-guttate-hypomelanosis",
# #     "lentigo-adults",
# #     "erythema-ab-igne",
# #     "poikiloderma-civatte",
# #     "erythromelanosis-follicularis-faciei-colli",
# #     "albinism",

# #     # Hair/Scalp
# #     "telogen-effluvium",
# #     "anagen-effluvium",
# #     "trichorrhexis-nodosa",
# #     "hirsutism",
# #     "alopecia-areata",

# #     # Acne
# #     "acne-cystic",
# #     "acne-scar",
# #     "acne-pustular",
# #     "acne-nodular",

# #     # Eczema
# #     "eczema-fingertips",
# #     "eczema-hand",
# #     "eczema-nummular",
# #     "eczema-face",
# #     "eczema-acute",
# #     "eczema-subacute",
# #     "eczema-asteatotic",

# #     # Psoriasis
# #     "psoriasis-palms-soles",
# #     "psoriasis-chronic-plaque",
# #     "psoriasis-scalp",
# #     "psoriasis-erythrodermic",

# #     # Rosacea (Optional group, comment if needed)
# #     "rosacea",
# #     "rosacea-nose",
# #     "rosacea-steroid"
# # ]


# # IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# # def ensure_folder(path):
# #     if not os.path.exists(path):
# #         os.makedirs(path)

# # def clean_dataset():
# #     label_counts = defaultdict(int)
# #     discarded_count = 0

# #     ensure_folder(DEST_ROOT)
# #     discarded_path = os.path.join(DEST_ROOT, "_discarded")
# #     ensure_folder(discarded_path)

# #     for folder in os.listdir(SOURCE_ROOT):
# #         folder_path = os.path.join(SOURCE_ROOT, folder)
# #         if not os.path.isdir(folder_path):
# #             continue

# #         for file in os.listdir(folder_path):
# #             if not file.lower().endswith(IMAGE_EXTENSIONS):
# #                 continue

# #             matched = False
# #             for label in VITAMIN_LABELS:
# #                 if label in file.lower():
# #                     target_folder = os.path.join(DEST_ROOT, label)
# #                     ensure_folder(target_folder)
# #                     shutil.copy(os.path.join(folder_path, file), os.path.join(target_folder, file))
# #                     label_counts[label] += 1
# #                     matched = True
# #                     break

# #             if not matched:
# #                 shutil.copy(os.path.join(folder_path, file), os.path.join(discarded_path, file))
# #                 discarded_count += 1

# #     print("\n✅ CLEANUP COMPLETE\n")
# #     print("=== Final Class Distribution ===")
# #     for label, count in label_counts.items():
# #         print(f"{label}: {count}")
# #     print(f"\n🗑️ Discarded images: {discarded_count}")

# # if __name__ == "__main__":
# #     clean_dataset()


# THis is to make the diffreent folders into one under same decisis name 


# import os
# import shutil
# from collections import defaultdict

# # Set your cleaned dataset path
# CLEANED_ROOT = r"E:\college final year project\dataset_cleaned"
# FINAL_DEST   = r"E:\college final year project\train"  # This will have 5 folders

# # Mapping from detailed condition folder → category
# CATEGORY_MAP = {
#     # Acne
#     "acne-cystic": "acne",
#     "acne-pustular": "acne",
#     "acne-scar": "acne",
#     "acne-open-comedo": "acne",
#     "acne-closed-comedo": "acne",
#     "acne-excoriated": "acne",
#     "acne-infantile": "acne",
#     "acne-primary-lesion": "acne",
#     "acne-mechanica": "acne",
    
#     # Eczema
#     "eczema-fingertips": "eczema",
#     "eczema-hand": "eczema",
#     "eczema-nummular": "eczema",
#     "eczema-subacute": "eczema",
#     "eczema-acute": "eczema",
#     "eczema-asteatotic": "eczema",
#     "eczema-chronic": "eczema",
#     "eczema-face": "eczema",
    
#     # Psoriasis
#     "psoriasis-palms-soles": "psoriasis",
#     "psoriasis-chronic-plaque": "psoriasis",
#     "psoriasis-scalp": "psoriasis",
#     "psoriasis-erythrodermic": "psoriasis",
    
#     # Pigmentation Disorders
#     "vitiligo": "pigmentation",
#     "melasma": "pigmentation",
#     "idiopathic-guttate-hypomelanosis": "pigmentation",
#     "lentigo-adults": "pigmentation",
#     "erythema-ab-igne": "pigmentation",
#     "poikiloderma-civatte": "pigmentation",
#     "erythromelanosis-follicularis-faciei-colli": "pigmentation",
#     "albinism": "pigmentation",
    
#     # Rosacea
#     "rosacea": "rosacea",
#     "rosacea-nose": "rosacea",
#     "rosacea-steroid": "rosacea"
# }

# def ensure_folder(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def consolidate_folders():
#     moved_counts = defaultdict(int)

#     for folder_name in os.listdir(CLEANED_ROOT):
#         src_path = os.path.join(CLEANED_ROOT, folder_name)
#         if not os.path.isdir(src_path) or folder_name == "_discarded":
#             continue

#         # Get target category
#         category = CATEGORY_MAP.get(folder_name)
#         if category:
#             dest_folder = os.path.join(FINAL_DEST, category)
#             ensure_folder(dest_folder)

#             for img in os.listdir(src_path):
#                 if img.lower().endswith(('.jpg', '.jpeg', '.png')):
#                     shutil.move(os.path.join(src_path, img), os.path.join(dest_folder, img))
#                     moved_counts[category] += 1

#             os.rmdir(src_path)  # Delete empty subfolder
#         else:
#             print(f"[SKIPPED] {folder_name} not mapped to any category.")

#     print("\n✅ Consolidation Complete!\n=== Final Category Counts ===")
#     for cat, count in moved_counts.items():
#         print(f"{cat}: {count} images")

# if __name__ == "__main__":
#     consolidate_folders()








import os
import shutil
import random
from collections import defaultdict

SOURCE_ROOT = r"E:\college final year project\train_cleaned dataser"
DEST_ROOT = r"E:\college final year project\train_sample"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_all_images(category_path):
    images = []
    for root, _, files in os.walk(category_path):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, category_path)
                images.append((full_path, rel_path))
    return images

def main():
    category_image_map = defaultdict(list)

    for category in os.listdir(SOURCE_ROOT):
        category_path = os.path.join(SOURCE_ROOT, category)
        if os.path.isdir(category_path):
            images = get_all_images(category_path)
            if images:
                category_image_map[category] = images
                print(f"[OK] {category}: {len(images)} images")
            else:
                print(f"[!] {category} has no images.")

    if not category_image_map:
        print(" No valid categories found.")
        return

    min_images = min(len(imgs) for imgs in category_image_map.values())
    print(f" Using {min_images} images per class for balanced sampling.")

    for category, image_list in category_image_map.items():
        selected = random.sample(image_list, min_images)
        for src, rel_path in selected:
            dest_path = os.path.join(DEST_ROOT, category, rel_path)
            ensure_folder(os.path.dirname(dest_path))
            shutil.copy(src, dest_path)
        print(f"[OK] Copied {len(selected)} images to '{category}'")

    print("\n Sample dataset created successfully, balanced across all categories.")

if __name__ == "__main__":
    main()
