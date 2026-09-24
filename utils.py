"""Utility helpers for the AI engine."""
import os
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)


def get_model_path(category: str) -> str:
    return os.path.join(settings.AI_MODEL_DIR, f'fanet_{category}.pth')


def get_bank_path(category: str) -> str:
    return os.path.join(settings.MEMORY_BANK_DIR, f'memory_{category}.npy')


def model_exists(category: str) -> bool:
    return os.path.exists(get_model_path(category))


def bank_exists(category: str) -> bool:
    return os.path.exists(get_bank_path(category))


def is_trained(category: str) -> bool:
    return model_exists(category) and bank_exists(category)


def list_trained_categories() -> list[str]:
    model_dir = Path(settings.AI_MODEL_DIR)
    if not model_dir.exists():
        return []
    return [p.stem.replace('fanet_', '') for p in model_dir.glob('fanet_*.pth')]


def get_heatmap_save_path(image_id: int, category: str) -> str:
    heatmap_dir = os.path.join(settings.MEDIA_ROOT, 'heatmaps', category)
    os.makedirs(heatmap_dir, exist_ok=True)
    return os.path.join(heatmap_dir, f'heatmap_{image_id}.png')


def get_heatmap_media_path(image_id: int, category: str) -> str:
    return f'heatmaps/{category}/heatmap_{image_id}.png'
