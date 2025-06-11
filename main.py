import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Set style for better visualizations
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set Kaggle paths
HOME = "Sheep Classification Images"
# Define paths for train and test data
train_dir = Path(f'{HOME}/train')
test_dir = Path(f'{HOME}/test')
labels_file = Path(f'{HOME}/train_labels.csv')

print("📁 Setup Complete!")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Fungsi bantu untuk memuat dan mengubah ukuran gambar
def load_images(image_paths, image_size=(64, 64)):
    images = []
    for path in tqdm(image_paths, desc="🔄 Loading images"):
        img = Image.open(path).convert("RGB").resize(image_size)
        images.append(np.array(img).flatten())  # Flatten to 1D
    return np.array(images)

# 1. Load labels
df_labels = pd.read_csv(labels_file)
df_labels['file_path'] = df_labels['filename'].apply(lambda x: train_dir / x)

# Encode labels
le = LabelEncoder()
df_labels['encoded_label'] = le.fit_transform(df_labels['label'])

# 2. Load and process training images
X = load_images(df_labels['file_path'])
y = df_labels['encoded_label']

# 3. Split and train simple model (logistic regression)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print(f"✅ Validation Accuracy: {accuracy_score(y_val, y_pred):.2f}")

# 4. Load test images
test_image_paths = sorted(list(test_dir.glob("*.jpg")))  # Ganti ekstensi jika bukan jpg
X_test = load_images(test_image_paths)

# 5. Predict
test_preds = model.predict(X_test)
test_labels = le.inverse_transform(test_preds)

# 6. Simpan hasil prediksi
submission_df = pd.DataFrame({
    "filename": [p.name for p in test_image_paths],
    "label": test_labels
})
submission_df.to_csv("test_predictions.csv", index=False)
print("📄 Prediksi disimpan ke test_predictions.csv")
