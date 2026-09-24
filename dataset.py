"""
Dataset handling for FANet Industrial Anomaly Detection using real datasets.
Automatically discovers categories from the datasets/ folder.
"""
import os
import logging
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)

# ─────────────────────────── Image Transforms ────────────────────

# Advanced Data Augmentation as requested
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────── Helpers ─────────────────────────────

def find_category_path(category: str, dataset_dir: str) -> str:
    """Find the dataset path for a category, supporting '<cat>1' or '<cat>'."""
    root = Path(dataset_dir)
    # Try <cat>1 first
    p1 = root / f"{category}1"
    if p1.exists() and (p1 / 'train' / 'good').exists():
        return str(p1)
    # Fallback to <cat>
    p2 = root / category
    if p2.exists() and (p2 / 'train' / 'good').exists():
        return str(p2)
    return None

def list_available_categories(dataset_dir: str) -> list:
    """Dynamically scan datasets/ folder for valid categories."""
    root = Path(dataset_dir)
    if not root.exists():
        return []
    cats = []
    for d in root.iterdir():
        if d.is_dir() and (d / 'train' / 'good').exists():
            # Strip '1' from end for display/mapping if it's there
            name = d.name
            if name.endswith('1'):
                name = name[:-1]
            if name not in cats:
                cats.append(name)
    return sorted(cats)

# ─────────────────────────── PyTorch Datasets ─────────────────────

class NormalImageDataset(Dataset):
    """Normal (defect-free) images for training."""
    EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

    def __init__(self, root: str, category: str, transform=None, limit=150):
        self.transform = transform or TRAIN_TRANSFORM
        self.images = []
        
        cat_path = find_category_path(category, root)
        if not cat_path:
            raise FileNotFoundError(f"Dataset for '{category}' not found in {root}")
            
        good_dir = Path(cat_path) / 'train' / 'good'
        for ext in self.EXTENSIONS:
            self.images.extend(sorted(good_dir.glob(ext)))
            
        # Limit dataset size for performance
        if limit and len(self.images) > limit:
            self.images = self.images[:limit]
            
        if not self.images:
            raise FileNotFoundError(f"No training images in {good_dir}")
        logger.info(f"Loaded {len(self.images)} images for {category}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(str(self.images[idx])).convert('RGB')
        return self.transform(img)

class TestImageDataset(Dataset):
    """Test images (normal + defective) for evaluation."""
    EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

    def __init__(self, root: str, category: str, transform=None):
        self.transform = transform or INFER_TRANSFORM
        self.images = []
        self.labels = []
        
        cat_path = find_category_path(category, root)
        if not cat_path:
            return

        test_dir = Path(cat_path) / 'test'
        if test_dir.exists():
            for subdir in sorted(test_dir.iterdir()):
                if subdir.is_dir():
                    is_defect = (subdir.name != 'good')
                    for ext in self.EXTENSIONS:
                        for p in sorted(subdir.glob(ext)):
                            self.images.append(p)
                            self.labels.append(1 if is_defect else 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(str(self.images[idx])).convert('RGB')
        return self.transform(img), self.labels[idx]

# ─────────────────────────── DataLoader factories ─────────────────

def get_train_loader(category: str, dataset_dir: str, batch_size: int = 8) -> DataLoader:
    ds = NormalImageDataset(dataset_dir, category, limit=150)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

def get_test_loader(category: str, dataset_dir: str, batch_size: int = 8) -> DataLoader:
    ds = TestImageDataset(dataset_dir, category)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_supervised_loader(category: str, dataset_dir: str, batch_size: int = 8) -> DataLoader:
    path = find_category_path(category, dataset_dir)
    ds = SupervisedIndustrialDataset(path, transform=TRAIN_TRANSFORM)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
class SupervisedIndustrialDataset(Dataset):
    """
    Dataset for supervised binary classification (Normal vs Defective).
    Loads 'train/good' as class 0.
    Loads 'test/defective_types' as class 1 to provide defective examples for training.
    """
    def __init__(self, root_dir, transform=None, max_images=300):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # 1. Normal samples (Class 0)
        normal_dir = self.root_dir / 'train' / 'good'
        if normal_dir.exists():
            files = list(normal_dir.glob('*'))[:max_images//2]
            self.images.extend([str(f) for f in files])
            self.labels.extend([0] * len(files))

        # 2. Defective samples (Class 1)
        # In MVTec, defects are in 'test/' subfolders other than 'good'
        test_dir = self.root_dir / 'test'
        if test_dir.exists():
            defective_folders = [d for d in test_dir.iterdir() if d.is_dir() and d.name != 'good']
            # Distribute remaining quota among all defect types
            per_folder = max(1, (max_images // 2) // len(defective_folders)) if defective_folders else 0
            for d in defective_folders:
                files = list(d.glob('*'))[:per_folder]
                self.images.extend([str(f) for f in files])
                self.labels.extend([1] * len(files))

        logger.info(f"Supervised Dataset: {len(self.images)} images "
                    f"({self.labels.count(0)} normal, {self.labels.count(1)} defective)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            # Return a blank tensor if image fails
            return torch.zeros(3, 224, 224), label
