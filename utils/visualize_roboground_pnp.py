"""Sanity-check visualization for the extracted RoboGround PnP dataset --
no model involved, just decode the pick/place masks that were baked into
the extraction and show them against the Korean expressions, so the data
itself can be reviewed before it's used for training.
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
import matplotlib.font_manager as fm

_nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_nanum_path):
    fm.fontManager.addfont(_nanum_path)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

JSON_PATH = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboGround_PnP/instances_roboground_pnp_train_ko.json"
IMG_ROOT = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboGround_PnP"
OUT_DIR = "/home/seongyeol/VATEX/results/roboground_pnp_data_check"
N_SAMPLES = 16

with open(JSON_PATH) as f:
    d = json.load(f)

anns_by_image = {}
for a in d["annotations"]:
    anns_by_image.setdefault(a["image_id"], []).append(a)
images_by_id = {im["id"]: im for im in d["images"]}

random.seed(7)
sample_ids = random.sample(list(anns_by_image.keys()), N_SAMPLES)

os.makedirs(OUT_DIR, exist_ok=True)
for n, img_id in enumerate(sample_ids):
    im_info = images_by_id[img_id]
    img = np.array(Image.open(os.path.join(IMG_ROOT, im_info["file_name"])))
    anns = sorted(anns_by_image[img_id], key=lambda a: a["category_id"])
    pick = next(a for a in anns if a["category_id"] == 1)
    place = next(a for a in anns if a["category_id"] == 2)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    axes[0].imshow(img)
    axes[0].set_title("원본 이미지")
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
        ax.set_title(f'{label}: "{ann["expressions_ko"][0]}"')
        ax.axis("off")

    fig.suptitle(f'전체 지시문: "{pick["expressions_ko"][1]}"', fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"sample_{n:02d}.png")
    plt.savefig(out_path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
