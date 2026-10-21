"""
Convert RoboRefit and OCID-Ref (OCID_VLG) raw annotations into the same
COCO-style referring-expression json format used by VATEX for RefCOCO(+/g)
(see utils/convert_refexp_to_coco.py), i.e.:

{
  "info": "", "licenses": "", "categories": [{"id", "name", "supercategory"}],
  "images": [{"id", "file_name", "height", "width", "original_id", "dataset_name"}],
  "annotations": [{
      "id", "image_id", "category_id", "bbox" (xywh), "area", "iscrowd",
      "segmentation": RLE dict, "original_id", "expressions": [str, ...]
  }]
}

Segmentation masks are stored as COCO RLE (bitmask) rather than polygons,
since both source datasets ship raw binary mask PNGs (not polygons) and
detectron2's `annotations_to_instances(..., mask_format="bitmask")` (used by
VATEX's dataset_mapper) accepts RLE directly.

Usage:
    python utils/convert_robot_datasets_to_coco.py --dataset roborefit
    python utils/convert_robot_datasets_to_coco.py --dataset ocidref --ocid_split multiple
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
import pycocotools.mask as mask_util


def rle_from_binary_mask(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def bbox_xywh_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]


def build_category_list(class_names):
    names = sorted(set(class_names))
    return {name: i + 1 for i, name in enumerate(names)}, [
        {"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(names)
    ]


def convert_roborefit(data_root, output_root):
    splits = ["train", "testA", "testB"]
    for split in splits:
        json_path = os.path.join(data_root, "RoboRefit", split, f"roborefit_{split}.json")
        if not os.path.exists(json_path):
            print(f"[roborefit] skip missing split {split} ({json_path})")
            continue
        with open(json_path, "r") as f:
            rows = json.load(f)

        cat_map, categories = build_category_list([r["class"] for r in rows])

        images = {}  # rgb_path -> image record
        annotations = []
        skipped = 0
        for row in rows:
            rgb_path = row["rgb_path"].replace("\\", "/").replace("final_dataset", "RoboRefit")
            mask_path = row["mask_path"].replace("\\", "/").replace("final_dataset", "RoboRefit")
            abs_rgb = os.path.join(data_root, rgb_path)
            abs_mask = os.path.join(data_root, mask_path)

            if not os.path.exists(abs_mask):
                skipped += 1
                continue

            if rgb_path not in images:
                with Image.open(abs_rgb) as im:
                    w, h = im.size
                images[rgb_path] = {
                    "file_name": rgb_path,
                    "height": h,
                    "width": w,
                    "original_id": rgb_path,
                    "id": len(images),
                    "dataset_name": "roborefit",
                }

            mask_arr = np.array(Image.open(abs_mask).convert("RGB"))
            binary_mask = mask_arr.sum(axis=-1) > 0

            annotations.append(
                {
                    "segmentation": rle_from_binary_mask(binary_mask),
                    "area": int(binary_mask.sum()),
                    "iscrowd": 0,
                    "bbox": bbox_xywh_from_mask(binary_mask),
                    "image_id": images[rgb_path]["id"],
                    "category_id": cat_map[row["class"]],
                    "id": len(annotations),
                    "original_id": mask_path,
                    "expressions": [row["text"]],
                }
            )

        coco_ann = {
            "info": "",
            "licenses": "",
            "images": list(images.values()),
            "annotations": annotations,
            "categories": categories,
        }
        out_dir = os.path.join(output_root, "RoboRefit", split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"instances_roborefit_{split}.json")
        with open(out_path, "w") as f:
            json.dump(coco_ann, f)
        print(
            f"[roborefit/{split}] images={len(images)} annotations={len(annotations)} "
            f"expressions={sum(len(a['expressions']) for a in annotations)} skipped={skipped} -> {out_path}"
        )


def convert_ocidref(data_root, output_root, ocid_split="multiple"):
    splits = ["train", "val", "test"]
    for split in splits:
        json_path = os.path.join(data_root, "OCID_VLG", "refer", ocid_split, f"{split}_expressions.json")
        if not os.path.exists(json_path):
            print(f"[ocidref/{ocid_split}] skip missing split {split} ({json_path})")
            continue
        with open(json_path, "r") as f:
            payload = json.load(f)
        rows = payload["data"]

        class_names = [r["target"].rsplit("_", 1)[0] for r in rows]
        cat_map, categories = build_category_list(class_names)

        images = {}  # image_filename -> image record
        # group expressions referring to the same object instance in the same image
        obj_key_to_ann_idx = {}
        annotations = []
        skipped = 0
        for row in rows:
            seq_path, im_name = row["image_filename"].split(",")
            rel_rgb = os.path.join("OCID_VLG", seq_path, "rgb", im_name)
            rel_mask = os.path.join("OCID_VLG", seq_path, "seg_mask_instances_combi", im_name)
            abs_rgb = os.path.join(data_root, rel_rgb)
            abs_mask = os.path.join(data_root, rel_mask)

            if not os.path.exists(abs_mask):
                skipped += 1
                continue

            if row["image_filename"] not in images:
                with Image.open(abs_rgb) as im:
                    w, h = im.size
                images[row["image_filename"]] = {
                    "file_name": rel_rgb,
                    "height": h,
                    "width": w,
                    "original_id": row["image_filename"],
                    "id": len(images),
                    "dataset_name": "ocidref",
                }
            image_id = images[row["image_filename"]]["id"]

            obj_key = (row["image_filename"], row["answer"])
            class_name = row["target"].rsplit("_", 1)[0]

            if obj_key not in obj_key_to_ann_idx:
                mask_arr = np.array(Image.open(abs_mask))
                binary_mask = mask_arr == row["answer"]
                if binary_mask.sum() == 0:
                    skipped += 1
                    continue
                obj_key_to_ann_idx[obj_key] = len(annotations)
                annotations.append(
                    {
                        "segmentation": rle_from_binary_mask(binary_mask),
                        "area": int(binary_mask.sum()),
                        "iscrowd": 0,
                        "bbox": bbox_xywh_from_mask(binary_mask),
                        "image_id": image_id,
                        "category_id": cat_map[class_name],
                        "id": len(annotations),
                        "original_id": f"{row['image_filename']}::{row['answer']}",
                        "expressions": [],
                    }
                )
            annotations[obj_key_to_ann_idx[obj_key]]["expressions"].append(row["question"])

        coco_ann = {
            "info": "",
            "licenses": "",
            "images": list(images.values()),
            "annotations": annotations,
            "categories": categories,
        }
        out_dir = os.path.join(output_root, "OCID_VLG", ocid_split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"instances_ocidref_{split}.json")
        with open(out_path, "w") as f:
            json.dump(coco_ann, f)
        print(
            f"[ocidref/{ocid_split}/{split}] images={len(images)} annotations={len(annotations)} "
            f"expressions={sum(len(a['expressions']) for a in annotations)} skipped={skipped} -> {out_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["roborefit", "ocidref", "all"], default="all")
    parser.add_argument("--data_root", default="/mnt/e/Task_Infer_VLM/VLM_dataset")
    parser.add_argument("--output_root", default="/mnt/e/Task_Infer_VLM/VLM_dataset")
    parser.add_argument("--ocid_split", default="multiple", choices=["multiple", "unique", "novel-classes", "novel-instances"])
    args = parser.parse_args()

    if args.dataset in ("roborefit", "all"):
        convert_roborefit(args.data_root, args.output_root)
    if args.dataset in ("ocidref", "all"):
        convert_ocidref(args.data_root, args.output_root, args.ocid_split)
