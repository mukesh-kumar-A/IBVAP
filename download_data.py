import os
import shutil
import kagglehub

print("[*] Downloading Border Threat Dataset...")
download_path = kagglehub.dataset_download("shihabsarar/ai-based-threat-detection-dataset-for-border")
print(f"[✓] Downloaded to cache: {download_path}")

PROCESSED_BASE = "data/dataset_processed/splits"
os.makedirs(f"{PROCESSED_BASE}/train/images", exist_ok=True)
os.makedirs(f"{PROCESSED_BASE}/train/labels", exist_ok=True)
os.makedirs(f"{PROCESSED_BASE}/val/images", exist_ok=True)
os.makedirs(f"{PROCESSED_BASE}/val/labels", exist_ok=True)

for root, dirs, files in os.walk(download_path):
    for file in files:
        src_file = os.path.join(root, file)
        
        # Train images & labels copy
        if "train" in root.lower() and file.endswith(('.jpg', '.png', '.jpeg')):
            shutil.copy(src_file, f"{PROCESSED_BASE}/train/images/{file}")
        elif "train" in root.lower() and file.endswith('.txt'):
            shutil.copy(src_file, f"{PROCESSED_BASE}/train/labels/{file}")
            
        # Val images & labels copy
        elif ("val" in root.lower() or "test" in root.lower()) and file.endswith(('.jpg', '.png', '.jpeg')):
            shutil.copy(src_file, f"{PROCESSED_BASE}/val/images/{file}")
        elif ("val" in root.lower() or "test" in root.lower()) and file.endswith('.txt'):
            shutil.copy(src_file, f"{PROCESSED_BASE}/val/labels/{file}")

print(f"[✓] All dataset files copied to {PROCESSED_BASE}!")