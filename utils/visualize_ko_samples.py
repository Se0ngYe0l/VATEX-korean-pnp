"""Run inference with the trained Korean RoboRefit model on a random sample
of real test examples and save side-by-side (image / GT mask / predicted
mask) visualizations with the Korean expression as a caption.

Ground truth is decoded independently from the test JSON via pycocotools
(same approach vatex/data/ref_coco_eval.py uses for scoring) rather than
relying on the model's `batched_inputs["instances"]`, which the test-mode
dataset mapper leaves empty (test mode is inference-only; GT comparison is
the evaluator's job, done separately against the raw annotation file).
"""
import os
import sys
import random

sys.path.append(".")
sys.path.append("..")

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pycocotools.coco import COCO

_nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_nanum_path):
    fm.fontManager.addfont(_nanum_path)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model

import vatex  # noqa: registers datasets/model via package side effects
from vatex import add_vatex_config
from vatex.data import build_detection_test_loader as vatex_build_test_loader
from vatex.data.dataset_mapper import YTVISDatasetMapper

N_SAMPLES = 20
OUT_DIR = "results/ko_sample_visualizations"
CONFIG = "configs/roborefit/swin/swin_base_ko.yaml"
WEIGHTS = "results/roborefit_finetune_ko/model_final.pth"
GT_JSON = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboRefit/testA/instances_roborefit_testA_ko.json"


def setup_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_vatex_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.freeze()
    return cfg


def main():
    cfg = setup_cfg()
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    gt_api = COCO(GT_JSON)

    all_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    random.shuffle(all_dicts)
    sampled = all_dicts[:N_SAMPLES]

    mapper = YTVISDatasetMapper(cfg, is_train=False)
    loader = vatex_build_test_loader(sampled, mapper=mapper)

    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        for count, batched_inputs in enumerate(loader):
            expr = batched_inputs[0]["expression"]
            file_name = batched_inputs[0]["file_names"][0]
            image_id = int(batched_inputs[0]["video_id"])
            anno_id = batched_inputs[0]["anno_id"]

            out = model(batched_inputs)

            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pred_masks = out["pred_masks"]
            pred_mask = pred_masks[0].numpy() if len(pred_masks) > 0 else np.zeros(image.shape[:2], dtype=bool)
            pred_mask = np.squeeze(pred_mask)

            ann = gt_api.imgToAnns[image_id][anno_id]
            gt_mask = gt_api.annToMask(ann).astype(bool)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title("원본 이미지")
            axes[0].axis("off")

            axes[1].imshow(image)
            overlay = np.zeros((*gt_mask.shape, 4))
            overlay[gt_mask] = [0, 1, 0, 0.5]
            axes[1].imshow(overlay)
            axes[1].set_title("정답(GT) 마스크")
            axes[1].axis("off")

            axes[2].imshow(image)
            overlay = np.zeros((*pred_mask.shape, 4))
            overlay[pred_mask] = [1, 0, 0, 0.5]
            axes[2].imshow(overlay)
            axes[2].set_title("모델 예측 마스크")
            axes[2].axis("off")

            fig.suptitle(f'"{expr}"', fontsize=14)
            plt.tight_layout()
            out_path = os.path.join(OUT_DIR, f"sample_{count:02d}.png")
            plt.savefig(out_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {out_path}  expr={expr}  gt_px={gt_mask.sum()}  pred_px={pred_mask.sum()}")


if __name__ == "__main__":
    main()
