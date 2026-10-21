"""Extract pick+place dual-mask training data from RoboGround task hdf5
files (RoboCasa simulator demonstrations) into a VATEX/COCO-style
referring segmentation dataset.

Each demo's first frame already has a ground-truth segmentation mask
(pixel value 1 = pick target, 2 = place target) baked into the .hdf5
alongside a ready-made English phrase for each (`target_obj_phrase`,
`target_place_phrase`) -- no simulator re-rendering or GPT instruction
generation needed.

Produces two annotations per usable demo (category "pick_target" /
"place_target"), each with its own English `expressions` list, in the
same COCO+expressions format used for RoboRefit/OCID-Ref so it slots into
the existing VATEX data pipeline (and the existing NLLB translation
script) unchanged.

Usage: python extract_roboground_pnp.py <task_name> [n_demos]
  task_name matches the RoboGround_Data task, e.g. PnPCounterToCab.
  Expects <H5_ROOT>/<task_name>.hdf5 to exist (extracted from the
  downloaded zip/z01/z02/...).
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

H5_ROOT = "/mnt/e/Task_Infer_VLM/VLM_dataset/roboground_data"
OUT_ROOT = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboGround_PnP"
CAMERA = "robot0_agentview_left"
CATEGORIES = [
    {"id": 1, "name": "pick_target"},
    {"id": 2, "name": "place_target"},
]
MIN_MASK_PIXELS = 30  # skip degenerate/empty masks


def rle_from_binary_mask(bmask):
    rle = mask_util.encode(np.asfortranarray(bmask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def bbox_from_binary_mask(bmask):
    ys, xs = np.where(bmask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def main(task_name, n_demos=None):
    h5_path = os.path.join(H5_ROOT, f"{task_name}.hdf5")
    img_dir = os.path.join(OUT_ROOT, "images", task_name)
    os.makedirs(img_dir, exist_ok=True)

    f = h5py.File(h5_path, "r")
    demo_keys = list(f["data"].keys())
    if n_demos is not None:
        demo_keys = demo_keys[:n_demos]

    images, annotations = [], []
    ann_id = 0
    n_skipped = 0
    for img_id, demo_key in enumerate(demo_keys):
        d = f["data"][demo_key]
        meta = json.loads(d.attrs["ep_meta"])
        obj_phrase = meta.get("target_obj_phrase")
        place_phrase = meta.get("target_place_phrase")
        full_lang = meta.get("lang")
        if not obj_phrase or not place_phrase:
            n_skipped += 1
            continue

        if f"{CAMERA}_image" not in d["obs"] or f"{CAMERA}_mask" not in d["obs"]:
            n_skipped += 1
            continue

        img = d["obs"][f"{CAMERA}_image"][0]  # (H,W,3) uint8, first frame
        seg = d["obs"][f"{CAMERA}_mask"][0, :, :, 0]  # (H,W) uint8, values 0/1/2

        pick_mask = seg == 1
        place_mask = seg == 2
        if pick_mask.sum() < MIN_MASK_PIXELS or place_mask.sum() < MIN_MASK_PIXELS:
            n_skipped += 1
            continue

        file_name = f"{demo_key}.png"
        Image.fromarray(img).save(os.path.join(img_dir, file_name))

        images.append({
            "id": img_id,
            "file_name": f"images/{task_name}/{file_name}",
            "height": img.shape[0],
            "width": img.shape[1],
        })

        for cat_id, bmask, phrase in [
            (1, pick_mask, obj_phrase),
            (2, place_mask, place_phrase),
        ]:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": rle_from_binary_mask(bmask),
                "bbox": bbox_from_binary_mask(bmask),
                "area": int(bmask.sum()),
                "iscrowd": 0,
                "expressions": [phrase, full_lang],
                "original_id": f"roboground_pnp/{task_name}/{demo_key}/{cat_id}",
            })
            ann_id += 1

    coco = {
        "info": {"description": f"RoboGround {task_name} pick+place referring segmentation"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }
    out_path = os.path.join(OUT_ROOT, f"instances_roboground_{task_name}.json")
    with open(out_path, "w") as fo:
        json.dump(coco, fo)
    print(f"wrote {out_path}: {len(images)} images, {len(annotations)} annotations, {n_skipped} demos skipped")


if __name__ == "__main__":
    task = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(task, n)
