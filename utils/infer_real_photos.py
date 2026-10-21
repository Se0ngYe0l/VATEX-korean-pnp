"""Run the trained Korean pick+place model on arbitrary real (non-simulated)
photos, given manually-written Korean pick/place expressions per image (no
COCO-format dataset needed -- builds the minimal dataset_dict by hand,
mirroring what YTVISDatasetMapper produces at inference time).
"""
import os
import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import cv2
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_nanum_path):
    fm.fontManager.addfont(_nanum_path)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

import vatex  # noqa: registers datasets/model via package side effects
from vatex import add_vatex_config
from vatex.data.augmentation import build_augmentation
from vatex.utils.split_pnp_ko import split_pnp_instruction_ko
from detectron2.modeling import build_model

CONFIG = "configs/roborefit/swin/swin_base_ko_pnp.yaml"
WEIGHTS = "results/roborefit_pnp_finetune_ko/model_final.pth"
OUT_DIR = "results/real_photo_visualizations"

# (image_path, one compound Korean instruction) -- split_pnp_instruction_ko
# parses this into (pick_phrase, place_phrase) before inference; VATEX is
# single-target, so the compound sentence itself is never fed to the model.
SAMPLES = [
    (
        "/mnt/c/Users/SeongYeol/Pictures/testt.png",
        "스팸을 김스낵 옆에 놔줘",
    ),
    (
        "/mnt/c/Users/SeongYeol/Pictures/testtt.png",
        "캔을 노란 바구니에 넣어줘",
    ),
]


def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_vatex_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.freeze()
    return cfg


def build_input(file_name, expression, augmentations, image_format):
    image = utils.read_image(file_name, format=image_format)
    orig_h, orig_w = image.shape[:2]
    aug_input = T.AugInput(image)
    augmentations(aug_input)
    image = aug_input.image
    image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    return {
        "height": orig_h,
        "width": orig_w,
        "length": 1,
        "video_name": os.path.basename(file_name),
        "video_id": "0",
        "anno_id": 0,
        "exp_id": 0,
        "expression": expression,
        "image": [image_tensor],
        "instances": [],
        "file_names": [file_name],
    }


def run_one(model, augmentations, image_format, file_name, expression):
    batched_inputs = [build_input(file_name, expression, T.AugmentationList(augmentations), image_format)]
    with torch.no_grad():
        out = model(batched_inputs)
    pred_masks = out["pred_masks"]
    if len(pred_masks) == 0:
        return None
    return np.squeeze(pred_masks[0].numpy())


def main():
    cfg = setup_cfg()
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    augmentations = build_augmentation(cfg, is_train=False)

    os.makedirs(OUT_DIR, exist_ok=True)

    for i, (file_name, instruction) in enumerate(SAMPLES):
        pick_expr, place_expr = split_pnp_instruction_ko(instruction)
        print(f'"{instruction}" -> pick="{pick_expr}" place="{place_expr}"')

        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pick_mask = run_one(model, augmentations, cfg.INPUT.FORMAT, file_name, pick_expr)
        place_mask = run_one(model, augmentations, cfg.INPUT.FORMAT, file_name, place_expr)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        axes[0].imshow(image)
        axes[0].set_title("원본 이미지")
        axes[0].axis("off")

        axes[1].imshow(image)
        if pick_mask is not None:
            ov = np.zeros((*pick_mask.shape, 4)); ov[pick_mask.astype(bool)] = [1, 0, 0, 0.5]; axes[1].imshow(ov)
        if place_mask is not None:
            ov = np.zeros((*place_mask.shape, 4)); ov[place_mask.astype(bool)] = [0, 1, 0, 0.5]; axes[1].imshow(ov)
        axes[1].set_title("모델 예측: 빨강=pick, 초록=place")
        axes[1].axis("off")

        fig.suptitle(
            f'지시문: "{instruction}"\npick: "{pick_expr}"   |   place: "{place_expr}"',
            fontsize=13,
        )
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"real_{i:02d}.png")
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
