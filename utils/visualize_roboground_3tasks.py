"""Sanity-check visualization across the 3 completed RoboGround PnP
sources (PnPSinkToCounter, PnPCounterToCab, PnPCounterToMicrowave) --
English expressions shown since Cab/Microwave aren't translated yet;
this is purely a data/diversity check, not a training-quality check.
"""
import json
import os
import random

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboGround_PnP"
OUT_DIR = "/home/seongyeol/VATEX/results/roboground_3task_check"
N_PER_TASK = 6

SOURCES = [
    ("PnPSinkToCounter (기존)", os.path.join(ROOT, "instances_roboground_pnp.json")),
    ("PnPCounterToCab", os.path.join(ROOT, "instances_roboground_PnPCounterToCab.json")),
    ("PnPCounterToMicrowave", os.path.join(ROOT, "instances_roboground_PnPCounterToMicrowave.json")),
]

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(11)

sample_i = 0
manifest = []
for task_label, json_path in SOURCES:
    with open(json_path) as f:
        d = json.load(f)
    anns_by_image = {}
    for a in d["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    images_by_id = {im["id"]: im for im in d["images"]}

    sample_ids = random.sample(list(anns_by_image.keys()), min(N_PER_TASK, len(anns_by_image)))
    for img_id in sample_ids:
        im_info = images_by_id[img_id]
        img = np.array(Image.open(os.path.join(ROOT, im_info["file_name"])))
        anns = anns_by_image[img_id]
        pick = next(a for a in anns if a["category_id"] == 1)
        place = next(a for a in anns if a["category_id"] == 2)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title(task_label, fontsize=11)
        axes[0].axis("off")

        for ax, ann, color, label in [
            (axes[1], pick, [1, 0, 0], "pick"),
            (axes[2], place, [0, 1, 0], "place"),
        ]:
            m = mask_util.decode(ann["segmentation"]).astype(bool)
            ax.imshow(img)
            overlay = np.zeros((*m.shape, 4))
            overlay[m] = color + [0.55]
            ax.imshow(overlay)
            ax.set_title(f'{label}: "{ann["expressions"][0]}"', fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"sample_{sample_i:03d}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        manifest.append((task_label, out_path))
        sample_i += 1

print(f"generated {sample_i} samples")
for label, path in manifest:
    print(label, path)
