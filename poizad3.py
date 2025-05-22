import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Konfiguracja
TEXTURE_CATEGORIES = ["gres", "tynk", "laminat"]
RAW_DIR = "raw_images"
PATCH_DIR = "patches"
CSV_FILE = "texture_features.csv"

# 2. Wycinanie próbek
def extract_patches(input_dir, output_dir, patch_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    for category in TEXTURE_CATEGORIES:
        in_path = os.path.join(input_dir, category)
        out_path = os.path.join(output_dir, category)
        os.makedirs(out_path, exist_ok=True)

        for filename in os.listdir(in_path):
            img_path = os.path.join(in_path, filename)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            for y in range(0, h - patch_size[1] + 1, patch_size[1]):
                for x in range(0, w - patch_size[0] + 1, patch_size[0]):
                    patch = img[y:y+patch_size[1], x:x+patch_size[0]]
                    out_name = f"{os.path.splitext(filename)[0]}_{x}_{y}.png"
                    cv2.imwrite(os.path.join(out_path, out_name), patch)

# 3. Ekstrakcja cech GLCM
def extract_glcm_features(patches_dir, bit_depth=5, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    features = []
    levels = 2 ** bit_depth
    props = ["dissimilarity", "correlation", "contrast", "energy", "homogeneity", "ASM"]

    for category in TEXTURE_CATEGORIES:
        cat_path = os.path.join(patches_dir, category)
        for filename in os.listdir(cat_path):
            path = os.path.join(cat_path, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # Kwantyzacja do 5 bitów
            img = np.floor(img / (256 / levels)).astype(np.uint8)

            glcm = greycomatrix(img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
            feature_vector = {prop: greycoprops(glcm, prop).mean() for prop in props}
            feature_vector["category"] = category
            features.append(feature_vector)

    return pd.DataFrame(features)

# 4. Zapis do CSV
def save_features_to_csv(df, csv_path):
    df.to_csv(csv_path, index=False)

# 5. Klasyfikacja
def classify(csv_file):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=["category"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = SVC(kernel='linear')  # Możesz zmienić na 'rbf' albo użyć KNN
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność klasyfikatora: {acc:.4f}")

# 6. Git — zostawiamy Tobie do zrobienia w terminalu
# git init, git add ., git commit -m "Initial commit", itp.

# =========================
# Wykonanie wszystkich kroków
# =========================
if __name__ == "__main__":
    extract_patches(RAW_DIR, PATCH_DIR)
    df = extract_glcm_features(PATCH_DIR)
    save_features_to_csv(df, CSV_FILE)
    classify(CSV_FILE)