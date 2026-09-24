"""
Synthetic Industrial Image Generator for FANet Training.

Generates realistic industrial surface images (normal and defective) using
NumPy + PIL without any external downloads. Supports 8 industrial categories
each with category-specific texture and defect patterns.

Normal images: clean, regular patterns → used for training memory bank
Defective images: same textures with injected defects → used for test/eval
"""
import os
import random
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

logger = logging.getLogger(__name__)

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet',
    'grid', 'leather', 'metal_nut', 'tile',
    'transistor', 'wood', 'zipper', 'screw',
]

IMAGE_SIZE = (256, 256)
NORMAL_COUNT = 300    # normal images for training
TEST_NORMAL_COUNT = 50
TEST_DEFECT_COUNT = 50

SEED = 42
rng = np.random.default_rng(SEED)


# ─────────────────────────── helpers ────────────────────────────

def _noise(shape, scale=15):
    return (rng.random(shape) * scale).astype(np.uint8)


def _perlin_like(h, w, freq=8, octaves=4):
    """Approximate Perlin noise via summed sinusoids."""
    result = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    for k in range(octaves):
        f = freq * (2 ** k)
        xs = np.linspace(0, f * math.pi, w)
        ys = np.linspace(0, f * math.pi, h)
        X, Y = np.meshgrid(xs, ys)
        phase_x = rng.uniform(0, 2 * math.pi)
        phase_y = rng.uniform(0, 2 * math.pi)
        result += amp * (np.sin(X + phase_x) * np.cos(Y + phase_y))
        amp *= 0.5
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def _to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode='L').convert('RGB')
    return Image.fromarray(arr, mode='RGB')


def _save(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# ─────────────────────────── texture generators ─────────────────

def _tex_grid(h, w, cell=16, color=(200, 200, 200), line_color=(80, 80, 80)):
    arr = np.ones((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    n = _noise((h, w), 6)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + n, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + n, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + n, 0, 255)
    for i in range(0, h, cell):
        arr[i:i+2, :] = line_color
    for j in range(0, w, cell):
        arr[:, j:j+2] = line_color
    return arr


def _tex_leather(h, w):
    base = _perlin_like(h, w, freq=4, octaves=5)
    r = (base * 80 + 100).astype(np.uint8)
    g = (base * 40 + 55).astype(np.uint8)
    b = (base * 20 + 25).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _tex_metal(h, w):
    """Brushed metal look."""
    arr = np.zeros((h, w), dtype=np.float32)
    for _ in range(60):
        y = rng.integers(0, h)
        arr[max(0, y-1):y+2, :] += rng.uniform(0.3, 1.0)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    n = rng.random((h, w)).astype(np.float32) * 0.1
    arr = np.clip(arr + n, 0, 1)
    val = (arr * 60 + 170).astype(np.uint8)
    return np.stack([val, val, val], axis=-1)


def _tex_wood(h, w):
    rings = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            rings[y, x] = math.sin(d * 0.4 + rng.uniform(0, 0.5))
    rings = (rings - rings.min()) / (rings.max() - rings.min() + 1e-8)
    grain = _perlin_like(h, w, freq=2, octaves=3) * 0.3
    rings = np.clip(rings + grain, 0, 1)
    r = (rings * 60 + 130).astype(np.uint8)
    g = (rings * 30 + 80).astype(np.uint8)
    b = (rings * 10 + 30).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _tex_carpet(h, w):
    base = _perlin_like(h, w, freq=12, octaves=4)
    r = (base * 60 + 120).astype(np.uint8)
    g = (base * 40 + 70).astype(np.uint8)
    b = (base * 80 + 130).astype(np.uint8)
    for i in range(0, h, 4):
        stripe = rng.integers(0, 3)
        if stripe == 0:
            r[i:i+2, :] = np.clip(r[i:i+2, :] + 20, 0, 255)
    return np.stack([r, g, b], axis=-1)


def _tex_bottle(h, w):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 210
    cx, cy = w // 2, h // 2
    r_outer, r_inner = min(h, w) // 2 - 10, min(h, w) // 2 - 40
    for y in range(h):
        for x in range(w):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if r_inner <= d <= r_outer:
                arr[y, x] = [100, 150, 200]
            elif d < r_inner:
                arr[y, x] = [230, 230, 240]
    n = _noise((h, w, 3), 8)
    arr = np.clip(arr.astype(int) + n.astype(int) - 4, 0, 255).astype(np.uint8)
    return arr


def _tex_cable(h, w):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 30
    stripe_w = w // 8
    colors = [
        [200, 30, 30], [30, 200, 30], [30, 30, 200],
        [200, 200, 30], [200, 30, 200], [30, 200, 200],
        [200, 200, 200], [60, 60, 60],
    ]
    for i, col in enumerate(colors):
        x1 = i * stripe_w
        x2 = x1 + stripe_w - 3
        arr[:, x1:x2] = col
    n = _noise((h, w, 3), 10)
    arr = np.clip(arr.astype(int) + n.astype(int) - 5, 0, 255).astype(np.uint8)
    return arr


def _tex_capsule(h, w):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 220
    cx, cy = w // 2, h // 2
    cap_r, cap_h = w // 4, h // 3
    for y in range(h):
        for x in range(w):
            in_body = (cx - cap_r <= x <= cx + cap_r) and (cy - cap_h <= y <= cy + cap_h)
            in_top = math.sqrt((x - cx) ** 2 + (y - (cy - cap_h)) ** 2) < cap_r
            in_bot = math.sqrt((x - cx) ** 2 + (y - (cy + cap_h)) ** 2) < cap_r
            if in_body or in_top or in_bot:
                arr[y, x] = [160, 180, 200]
    n = _noise((h, w, 3), 6)
    arr = np.clip(arr.astype(int) + n.astype(int) - 3, 0, 255).astype(np.uint8)
    return arr


def _tex_tile(h, w, cell=32):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 200
    n = _noise((h, w, 3), 12)
    arr = np.clip(arr.astype(int) + n.astype(int) - 6, 0, 255).astype(np.uint8)
    for i in range(0, h, cell):
        arr[i:i+3, :] = [80, 80, 80]
    for j in range(0, w, cell):
        arr[:, j:j+3] = [80, 80, 80]
    return arr


def _tex_transistor(h, w):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 60
    pin_positions = [w // 5, 2 * w // 5, 3 * w // 5, 4 * w // 5]
    body_y1, body_y2 = h // 3, 2 * h // 3
    body_x1, body_x2 = w // 6, 5 * w // 6
    arr[body_y1:body_y2, body_x1:body_x2] = [30, 30, 50]
    for px in pin_positions:
        arr[body_y2:body_y2 + h // 5, px - 4:px + 4] = [180, 180, 180]
    n = _noise((h, w, 3), 8)
    arr = np.clip(arr.astype(int) + n.astype(int) - 4, 0, 255).astype(np.uint8)
    return arr


def _tex_metal_nut(h, w):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 190
    cx, cy = w // 2, h // 2
    outer_r, inner_r, hex_r = w // 2 - 5, w // 7, w // 2 - 5
    for y in range(h):
        for x in range(w):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if d <= outer_r:
                arr[y, x] = [170, 170, 175]
            if d <= inner_r:
                arr[y, x] = [220, 220, 220]
    n = _noise((h, w, 3), 8)
    arr = np.clip(arr.astype(int) + n.astype(int) - 4, 0, 255).astype(np.uint8)
    return arr


def _tex_zipper(h, w):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 150
    teeth_w = w // 12
    for i in range(0, w, teeth_w * 2):
        arr[:, i:i + teeth_w] = [100, 100, 110]
        for j in range(0, h, 8):
            arr[j:j+4, i:i + teeth_w] = [130, 130, 140]
    n = _noise((h, w, 3), 6)
    arr = np.clip(arr.astype(int) + n.astype(int) - 3, 0, 255).astype(np.uint8)
    return arr


def _tex_screw(h, w):
    arr = np.ones((h, w, 3), dtype=np.uint8) * 180
    cx = w // 2
    screw_w = w // 6
    arr[:, cx - screw_w:cx + screw_w] = [150, 150, 155]
    for i in range(0, h, 10):
        arr[i:i+3, cx - screw_w:cx + screw_w] = [100, 100, 105]
    arr[:h // 6, cx - w // 4:cx + w // 4] = [160, 160, 165]
    n = _noise((h, w, 3), 8)
    arr = np.clip(arr.astype(int) + n.astype(int) - 4, 0, 255).astype(np.uint8)
    return arr


TEXTURE_FN = {
    'bottle':     _tex_bottle,
    'cable':      _tex_cable,
    'capsule':    _tex_capsule,
    'carpet':     _tex_carpet,
    'grid':       _tex_grid,
    'leather':    _tex_leather,
    'metal_nut':  _tex_metal_nut,
    'tile':       _tex_tile,
    'transistor': _tex_transistor,
    'wood':       _tex_wood,
    'zipper':     _tex_zipper,
    'screw':      _tex_screw,
}


def _get_texture(category, h, w):
    fn = TEXTURE_FN.get(category, _tex_metal)
    if category == 'grid':
        return fn(h, w)
    return fn(h, w)


# ─────────────────────────── defect injectors ──────────────────

def _defect_scratch(arr):
    h, w = arr.shape[:2]
    for _ in range(rng.integers(1, 4)):
        x1, y1 = rng.integers(10, w - 10), rng.integers(10, h - 10)
        angle = rng.uniform(0, math.pi)
        length = rng.integers(30, 100)
        x2 = int(x1 + math.cos(angle) * length)
        y2 = int(y1 + math.sin(angle) * length)
        img = _to_pil(arr)
        draw = ImageDraw.Draw(img)
        scratch_col = tuple(int(c * 0.3) for c in img.getpixel((x1, y1))[:3])
        draw.line([(x1, y1), (x2, y2)], fill=scratch_col, width=rng.integers(1, 4))
        arr = np.array(img)
    return arr


def _defect_hole(arr):
    h, w = arr.shape[:2]
    for _ in range(rng.integers(1, 3)):
        cx = rng.integers(30, w - 30)
        cy = rng.integers(30, h - 30)
        r = rng.integers(5, 20)
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        arr[mask] = [20, 20, 20]
    return arr


def _defect_contamination(arr):
    h, w = arr.shape[:2]
    cx = rng.integers(20, w - 20)
    cy = rng.integers(20, h - 20)
    r = rng.integers(15, 40)
    blob = _perlin_like(h, w, freq=6, octaves=3)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = (dist < r) & (blob > 0.4)
    stain_color = np.array([rng.integers(80, 140),
                             rng.integers(60, 100),
                             rng.integers(20, 60)], dtype=np.uint8)
    arr[mask] = stain_color
    return arr


def _defect_crack(arr):
    h, w = arr.shape[:2]
    img = _to_pil(arr)
    draw = ImageDraw.Draw(img)
    x, y = rng.integers(20, w - 20), rng.integers(20, h - 20)
    pts = [(x, y)]
    for _ in range(rng.integers(5, 12)):
        dx = rng.integers(-20, 20)
        dy = rng.integers(-20, 20)
        x = max(0, min(w - 1, x + dx))
        y = max(0, min(h - 1, y + dy))
        pts.append((x, y))
    draw.line(pts, fill=(10, 10, 10), width=2)
    return np.array(img)


def _defect_color_spot(arr):
    h, w = arr.shape[:2]
    cx = rng.integers(20, w - 20)
    cy = rng.integers(20, h - 20)
    r = rng.integers(10, 30)
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    spot = np.array([rng.integers(180, 255),
                     rng.integers(50, 120),
                     rng.integers(50, 120)], dtype=np.uint8)
    arr[mask] = spot
    return arr


DEFECT_FNS = [_defect_scratch, _defect_hole, _defect_contamination,
              _defect_crack, _defect_color_spot]


def _apply_random_defect(arr):
    fn = rng.choice(DEFECT_FNS)
    return fn(arr)


# ─────────────────────────── image generation ──────────────────

def _make_normal(category, h, w):
    arr = _get_texture(category, h, w)
    img = _to_pil(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    if rng.random() > 0.5:
        img = img.rotate(rng.integers(0, 360))
    return img


def _make_defective(category, h, w):
    arr = _get_texture(category, h, w)
    n_defects = rng.integers(1, 4)
    for _ in range(n_defects):
        arr = _apply_random_defect(arr)
    return _to_pil(arr)


# ─────────────────────────── public API ────────────────────────

def generate_category(category: str, root: str,
                       n_train: int = NORMAL_COUNT,
                       n_test_normal: int = TEST_NORMAL_COUNT,
                       n_test_defect: int = TEST_DEFECT_COUNT) -> str:
    """
    Generate a full MVTec-style dataset directory for one category.

    Layout produced:
        root/<category>/train/good/          ← n_train normal images
        root/<category>/test/good/           ← n_test_normal normal images
        root/<category>/test/defective/      ← n_test_defect defective images
        root/<category>/ground_truth/defective/  ← placeholder masks

    Returns the category root path.
    """
    h, w = IMAGE_SIZE
    cat_root = Path(root) / category

    # Skip if already generated
    train_good = cat_root / 'train' / 'good'
    if train_good.exists():
        existing = len(list(train_good.glob('*.png')))
        if existing >= n_train:
            logger.info(f"Dataset '{category}' already exists ({existing} train images). Skipping.")
            return str(cat_root)

    logger.info(f"Generating synthetic dataset for '{category}' …")

    # Training normals
    train_good.mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        img = _make_normal(category, h, w)
        img.save(str(train_good / f'{i:04d}.png'))

    # Test normals
    test_good = cat_root / 'test' / 'good'
    test_good.mkdir(parents=True, exist_ok=True)
    for i in range(n_test_normal):
        img = _make_normal(category, h, w)
        img.save(str(test_good / f'{i:04d}.png'))

    # Test defectives
    test_defect = cat_root / 'test' / 'defective'
    test_defect.mkdir(parents=True, exist_ok=True)
    gt_dir = cat_root / 'ground_truth' / 'defective'
    gt_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_test_defect):
        img = _make_defective(category, h, w)
        img.save(str(test_defect / f'{i:04d}.png'))
        # Blank ground-truth mask (placeholder)
        mask = Image.new('L', (h, w), 0)
        mask.save(str(gt_dir / f'{i:04d}_mask.png'))

    logger.info(
        f"Generated '{category}': {n_train} train, "
        f"{n_test_normal} test-normal, {n_test_defect} test-defective."
    )
    return str(cat_root)


def generate_all(root: str, categories: list = None,
                 n_train: int = NORMAL_COUNT,
                 n_test_normal: int = TEST_NORMAL_COUNT,
                 n_test_defect: int = TEST_DEFECT_COUNT) -> dict:
    """Generate synthetic datasets for all (or specified) categories."""
    cats = categories or CATEGORIES
    paths = {}
    for cat in cats:
        try:
            path = generate_category(cat, root, n_train, n_test_normal, n_test_defect)
            paths[cat] = path
        except Exception as exc:
            logger.error(f"Failed to generate '{cat}': {exc}")
    return paths


def dataset_exists(category: str, root: str) -> bool:
    """Check if a dataset already exists and has enough training images."""
    train_dir = Path(root) / category / 'train' / 'good'
    if not train_dir.exists():
        return False
    imgs = list(train_dir.glob('*.png')) + list(train_dir.glob('*.jpg'))
    return len(imgs) >= 50
