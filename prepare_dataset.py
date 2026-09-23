import os
import random
import shutil

# Directory definitions matching your architecture
RAW_IMAGES_DIR = "data/dataset_raw/images"
RAW_LABELS_DIR = "data/dataset_raw/labels"

PROCESSED_BASE = "data/dataset_processed/splits"
TRAIN_IMG = os.path.join(PROCESSED_BASE, "train/images")
TRAIN_LBL = os.path.join(PROCESSED_BASE, "train/labels")
VAL_IMG = os.path.join(PROCESSED_BASE, "val/images")
VAL_LBL = os.path.join(PROCESSED_BASE, "val/labels")

for p in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    os.makedirs(p, exist_ok=True)

def process_and_split(train_ratio=0.8):
    image_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) == 0:
        print(f"[!] No raw images found in {RAW_IMAGES_DIR}. Capture images first!")
        return

    random.seed(42)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)

    train_set = image_files[:split_idx]
    val_set = image_files[split_idx:]

    print(f"[*] Processing {len(image_files)} Raw Images -> {len(train_set)} Train | {len(val_set)} Validation")

    def copy_pair(files, img_dest, lbl_dest):
        for img_name in files:
            src_img = os.path.join(RAW_IMAGES_DIR, img_name)
            dst_img = os.path.join(img_dest, img_name)
            shutil.copy(src_img, dst_img)

            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(RAW_LABELS_DIR, label_name)
            dst_lbl = os.path.join(lbl_dest, label_name)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                open(dst_lbl, 'a').close()

    copy_pair(train_set, TRAIN_IMG, TRAIN_LBL)
    copy_pair(val_set, VAL_IMG, VAL_LBL)
    print(f"[✓] Processed splits successfully generated in {PROCESSED_BASE}!")

if __name__ == "__main__":
    process_and_split()