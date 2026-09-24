# FANet Industrial Anomaly Detection System

A production-grade industrial AI inspection platform using a **Feature Aggregation Network (FANet)** to detect manufacturing defects through memory-based anomaly detection.

---

## Features

- **FANet Architecture** — Multi-scale feature extraction (EfficientNetB0 / ResNet18) with CBAM attention modules and aggregation head
- **Synthetic Dataset Generator** — Built-in generator creates 300+ realistic normal + 50 defective images per category with NO downloads required
- **Real Dataset Support** — Drop in your own images or download MVTec AD (15 categories)
- **Memory Bank** — CoreSet subsampling stores compact, representative normal-image embeddings
- **Anomaly Heatmaps** — Feature-difference maps highlight defective regions
- **Role-Based Access** — Admin oversight + Operator console with separate dashboards
- **Live Training Monitor** — Real-time epoch logs, loss & accuracy tracking stored in MySQL
- **Dual Theme** — Soft teal (#b2dcdc) and light blue (#d7eef5) switchable themes

---

## Requirements

- Python 3.10+
- MySQL 8.0+
- pip / virtualenv

---

## Quick Setup (Automated)

```bash
cd fanet_project
chmod +x setup.sh
./setup.sh
```

The setup script does everything automatically:
1. Creates virtual environment
2. Installs all dependencies
3. Configures MySQL database
4. Runs migrations
5. Creates default users
6. **Generates synthetic datasets** (no internet required)
7. **Pre-trains FANet models** for all 12 categories

---

## Manual Setup (Step by Step)

### 1. Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. MySQL database
```sql
CREATE DATABASE fanet_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 3. Environment file
```bash
cp .env.example .env
# Edit .env with your MySQL credentials
```

`.env`:
```
SECRET_KEY=your-secret-key-here
DEBUG=True
DB_NAME=fanet_db
DB_USER=root
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=3306
```

### 4. Migrations
```bash
python manage.py makemigrations accounts inspection
python manage.py migrate
```

### 5. Create default users
```bash
python manage.py create_default_users
```

### 6. Collect static files
```bash
python manage.py collectstatic --noinput
```

### 7. Generate datasets (choose one option)

**Option A — Synthetic (no downloads, always works):**
```bash
python manage.py generate_datasets
```
Generates 300 normal training images + 50 defective test images per category
using the built-in procedural texture engine (PIL/NumPy only).

**Option B — Your own images:**
```
datasets/<category>/train/good/     ← put your normal images here
datasets/<category>/test/defective/ ← put defective images here
```
Then train normally.

**Option C — MVTec AD (requires registration at mvtec.com):**
Download the `.tar.xz` files from https://www.mvtec.com/company/research/datasets/mvtec-ad
and extract into `datasets/<category>/`.

### 8. Train models
```bash
# Train all categories (recommended for first setup)
python manage.py pretrain_models --epochs 10

# Train a specific category
python manage.py train_model --category grid --epochs 15 --backbone efficientnet_b0
```

### 9. Start the server
```bash
python manage.py runserver
```

Access: **http://127.0.0.1:8000**

---

## Default Credentials

| Role     | Username  | Password     |
|----------|-----------|--------------|
| Admin    | admin     | admin        |
| Operator | operator  | Operator@123 |

---

## Supported Categories

| Category    | Texture Type        | Defect Types Generated          |
|-------------|---------------------|---------------------------------|
| bottle      | Circular container  | Cracks, contamination, scratches|
| cable       | Multi-strand cable  | Scratches, missing strand       |
| capsule     | Oval capsule        | Cracks, contamination           |
| carpet      | Woven carpet        | Color spots, holes, threads     |
| grid        | Regular grid        | Broken lines, contamination     |
| leather     | Leather surface     | Cuts, color spots, holes        |
| metal_nut   | Hexagonal nut       | Scratches, bent, color          |
| tile        | Tile pattern        | Cracks, chips, contamination    |
| transistor  | Electronic comp.    | Damage, contamination           |
| wood        | Wood grain          | Scratches, knots                |
| zipper      | Zipper fabric       | Broken teeth, fabric defects    |
| screw       | Threaded screw      | Thread damage, scratches        |

---

## Training Your Own Images (Custom Dataset)

```
datasets/
  my_category/
    train/
      good/         ← 100+ normal images (PNG/JPG)
    test/
      good/         ← optional: normal test images
      defective/    ← optional: defective test images
```

Then train:
```bash
python manage.py train_model --category my_category --epochs 20
```

No code changes needed — the system discovers any folder structure automatically.

---

## Management Commands

| Command | Description |
|---------|-------------|
| `generate_datasets` | Generate synthetic datasets for all/specific categories |
| `pretrain_models` | Generate + train models for all/specific categories |
| `train_model --category <cat>` | Train a single category |
| `create_default_users` | Create admin/operator accounts |

---

## Architecture

```
Input Image
    ↓
Preprocessing (resize 256→224, normalize)
    ↓
EfficientNetB0 Backbone (pretrained ImageNet)
    ↓
Multi-Scale Features:
  layer1(24ch) → layer2(40ch) → layer3(80ch) → layer4(1280ch)
    ↓
CBAM Attention (Channel + Spatial) at each scale
    ↓
Feature Aggregation Head (4×256 → 512-d embedding)
    ↓
L2 Normalization
    ↓
Memory Bank (CoreSet-subsampled normal embeddings)
    ↓
k-NN Euclidean Distance
    ↓
Anomaly Score → Classification + Heatmap
```

---

## Dataset Structures Supported

```
MVTec AD layout (official):
  datasets/<cat>/train/good/*.png
  datasets/<cat>/test/good/*.png
  datasets/<cat>/test/<defect_type>/*.png
  datasets/<cat>/ground_truth/<defect_type>/*_mask.png

Flat layout:
  datasets/<cat>/*.png        ← treated as all-normal training
```

---

## Project Structure

```
fanet_project/
├── manage.py
├── requirements.txt
├── .env.example
├── setup.sh
├── fanet_inspection/         # Django project config
├── accounts/                 # Auth + user management
├── inspection/               # Core inspection app
│   └── management/commands/
│       ├── generate_datasets.py   ← NEW: synthetic data generator
│       ├── pretrain_models.py     ← NEW: batch train all categories
│       └── train_model.py
├── ai_engine/
│   ├── model.py              # FANet + CBAM
│   ├── memory_bank.py        # CoreSet memory bank
│   ├── training.py           # Training pipeline
│   ├── inference.py          # Inference + heatmaps
│   ├── dataset.py            # Dataset loading + preparation
│   ├── synthetic_data.py     ← NEW: procedural image generator
│   └── utils.py
├── templates/
├── static/
└── datasets/                 ← auto-populated by generate_datasets
```

---

## Database Schema

| Table | Description |
|-------|-------------|
| `accounts_custom_user` | Extended user with role, department, employee_id |
| `inspection_uploaded_image` | Image metadata, path, category, status |
| `inspection_detection_result` | Anomaly score, classification, heatmap, confidence |
| `inspection_training_session` | Training config, status, progress, model paths |
| `inspection_training_log` | Per-epoch loss, accuracy, messages |
| `inspection_system_metric` | Performance metrics per category |

---

## Color Theme

| Token | Value | Usage |
|-------|-------|-------|
| Primary | `#d7eef5` | Page background |
| Secondary | `#b2dcdc` | Cards, sections |
| Accent teal | `#4a9a9a` | Buttons, highlights |
| Accent dark | `#2d6b6b` | Sidebar, headings |
