"""Run inference with the newly trained pick+place model on random
RoboGround PnP test samples and save (image / GT pick+place / predicted
pick+place) visualizations with the Korean pick/place expressions.
"""
import os
import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import cv2
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

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

import vatex  # noqa: registers datasets/model via package side effects
from vatex import add_vatex_config
from vatex.data import build_detection_test_loader as vatex_build_test_loader
from vatex.data.dataset_mapper import YTVISDatasetMapper
from detectron2.modeling import build_model

import json

N_IMAGES = 16
OUT_DIR = "results/pnp_sample_visualizations"
CONFIG = "configs/roborefit/swin/swin_base_ko_pnp.yaml"
WEIGHTS = "results/roborefit_pnp_finetune_ko/model_final.pth"
TEST_JSON = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboGround_PnP/instances_roboground_pnp_merged_test_ko.json"


def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_vatex_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.freeze()
    return cfg


def main():
    with open(TEST_JSON) as f:
        gt_data = json.load(f)
    gt_by_image = {}
    for a in gt_data["annotations"]:
        gt_by_image.setdefault(a["image_id"], {})[a["category_id"]] = a

    cfg = setup_cfg()
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    mapper = YTVISDatasetMapper(cfg, is_train=False)
    loader = vatex_build_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)

    os.makedirs(OUT_DIR, exist_ok=True)
    preds_by_image = {}  # image_id -> {1: mask, 2: mask}
    exprs_by_image = {}  # image_id -> {1: expr, 2: expr}
    file_by_image = {}
    n_images_done = 0

    with torch.no_grad():
        for batched_inputs in loader:
            if n_images_done >= N_IMAGES:
                break
            if batched_inputs[0]["exp_id"] != 0:
                continue  # only the short phrase (index 0), not the shared full-sentence variant (index 1)
            anno_id = batched_inputs[0]["anno_id"]
            cat_id = anno_id + 1  # pick=anno_id0->cat1, place=anno_id1->cat2 (extraction order)
            expr = batched_inputs[0]["expression"]
            file_name = batched_inputs[0]["file_names"][0]
            image_id = int(batched_inputs[0]["video_id"])

            out = model(batched_inputs)
            pred_masks = out["pred_masks"]
            pred_mask = np.squeeze(pred_masks[0].numpy()) if len(pred_masks) > 0 else None

            preds_by_image.setdefault(image_id, {})[cat_id] = pred_mask
            exprs_by_image.setdefault(image_id, {})[cat_id] = expr
            file_by_image[image_id] = file_name

            if len(preds_by_image[image_id]) == 2:
                n_images_done += 1

    count = 0
    for image_id, preds in preds_by_image.items():
        if 1 not in preds or 2 not in preds:
            continue
        image = cv2.imread(file_by_image[image_id])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_pick = mask_util.decode(gt_by_image[image_id][1]["segmentation"]).astype(bool)
        gt_place = mask_util.decode(gt_by_image[image_id][2]["segmentation"]).astype(bool)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("원본 이미지")
        axes[0].axis("off")

        axes[1].imshow(image)
        ov = np.zeros((*gt_pick.shape, 4)); ov[gt_pick] = [1, 0, 0, 0.5]; axes[1].imshow(ov)
        ov = np.zeros((*gt_place.shape, 4)); ov[gt_place] = [0, 1, 0, 0.5]; axes[1].imshow(ov)
        axes[1].set_title("정답(GT): 빨강=pick, 초록=place")
        axes[1].axis("off")

        axes[2].imshow(image)
        if preds[1] is not None:
            ov = np.zeros((*preds[1].shape, 4)); ov[preds[1]] = [1, 0, 0, 0.5]; axes[2].imshow(ov)
        if preds[2] is not None:
            ov = np.zeros((*preds[2].shape, 4)); ov[preds[2]] = [0, 1, 0, 0.5]; axes[2].imshow(ov)
        axes[2].set_title("모델 예측: 빨강=pick, 초록=place")
        axes[2].axis("off")

        fig.suptitle(f'pick: "{exprs_by_image[image_id][1]}"   |   place: "{exprs_by_image[image_id][2]}"', fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"sample_{count:02d}.png")
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")
        count += 1


if __name__ == "__main__":
    main()
