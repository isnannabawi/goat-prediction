import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import warnings

# PyTorch dan torchvision
import torch
from torchvision import models, transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import xgboost as xgb

# Style dan warning
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

# Path setup
HOME = Path("Sheep Classification Images")
train_dir = HOME / "train"
test_dir = HOME / "test"
labels_file = HOME / "train_labels.csv"

print("📁 Setup Complete!")

# Pastikan file dan folder ada
assert train_dir.exists(), f"Folder train tidak ditemukan: {train_dir}"
assert test_dir.exists(), f"Folder test tidak ditemukan: {test_dir}"
assert labels_file.exists(), f"File label tidak ditemukan: {labels_file}"

# Load label dan encode
df_labels = pd.read_csv(labels_file)
df_labels['file_path'] = df_labels['filename'].apply(lambda x: train_dir / x)
le = LabelEncoder()
df_labels['encoded_label'] = le.fit_transform(df_labels['label'])

# Transformasi gambar dengan augmentasi ringan
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset PyTorch
class ImageDataset(Dataset):
    def __init__(self, df, transform):
        self.file_paths = df['file_path'].values
        self.labels = df['encoded_label'].values
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

train_dataset = ImageDataset(df_labels, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Load EfficientNetB0 dan hapus classifier layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
effnet = efficientnet_b0(pretrained=True)
effnet.classifier = torch.nn.Identity()  # remove classifier layer
effnet.to(device)
effnet.eval()

# Ekstrak fitur
def extract_features(dataloader, model):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="🔍 Extracting features"):
            imgs = imgs.to(device)
            feats = model(imgs).squeeze()
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)

X, y = extract_features(train_loader, effnet)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost classifier
clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

f1 = f1_score(y_val, y_pred, average='macro')
print(f"✅ Macro F1 Score: {f1:.5f}")

# ----------- Prediksi gambar test -------------
class TestDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# Load test data
test_image_paths = sorted(list(test_dir.glob("*.jpg")))
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_dataset = TestDataset(test_image_paths, test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Ekstrak fitur test
def extract_test_features(dataloader, model):
    feats = []
    with torch.no_grad():
        for imgs in tqdm(dataloader, desc="📸 Extracting test features"):
            imgs = imgs.to(device)
            f = model(imgs).squeeze()
            feats.append(f.cpu().numpy())
    return np.vstack(feats)

X_test = extract_test_features(test_loader, effnet)
test_preds = clf.predict(X_test)
test_labels = le.inverse_transform(test_preds)

submission_df = pd.DataFrame({
    "filename": [p.name for p in test_image_paths],
    "predicted_label": test_labels
})
submission_df.to_csv("test_predictions.csv", index=False)
print("📄 Prediksi test disimpan ke test_predictions.csv")