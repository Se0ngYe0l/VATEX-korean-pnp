# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os


from .ref_ytvos import (
    register_ref_ytvos_instances,
    _get_ref_ytvos_2021_instances_meta,
)

from .coco import register_coco_instances

# ==== Predefined splits for RefYTVIS 2021 ===========
_PREDEFINED_SPLITS_REF_YTVOS_2021 = {
    "ref_ytvos_2021_train": ("data/ref-youtube-vos/train/JPEGImages",
                         "data/train.json"),
    "ref_ytvos_2021_val_fake": ("data/ref-youtube-vos/valid/JPEGImages",
                         "data/valid_fake.json"),
    "ref_ytvos_2021_val": ("data/ref-youtube-vos/valid/JPEGImages",
                       "data/valid.json"),
    "ref_davis_val": ("data/ref-davis/valid/JPEGImages/",
                       "data/ref-davis/valid.json"),
    
    # "ref_ytvos_2021_test": ("ref_ytvos_2021/test/JPEGImages",
    #                     "ref_ytvos_2021/test.json"),
}

# ==== Predefined splits for RefCoco 2014 ===========
# Reuses already-downloaded COCO train2014 images + REFER annotations at
# _REF_SEG_ROOT instead of re-downloading via scripts/prepare_datasets.sh.
_REF_SEG_ROOT = "/mnt/e/Task_Infer_VLM/VLM_dataset/refer_seg"

_PREDEFINED_SPLITS_REF_COCO_2014 = {
    "refcoco_train": (f"{_REF_SEG_ROOT}/train2014",
                         f"{_REF_SEG_ROOT}/refcoco/instances_refcoco_train.json"),
    "refcoco_val": (f"{_REF_SEG_ROOT}/train2014",
                         f"{_REF_SEG_ROOT}/refcoco/instances_refcoco_val.json"),
    "refcoco_testA": (f"{_REF_SEG_ROOT}/train2014",
                         f"{_REF_SEG_ROOT}/refcoco/instances_refcoco_testA.json"),
    "refcoco_testB": (f"{_REF_SEG_ROOT}/train2014",
                         f"{_REF_SEG_ROOT}/refcoco/instances_refcoco_testB.json"),
    "refcoco+_train": (f"{_REF_SEG_ROOT}/train2014",
                        f"{_REF_SEG_ROOT}/refcoco+/instances_refcoco+_train.json"),
    "refcoco+_val": (f"{_REF_SEG_ROOT}/train2014",
                        f"{_REF_SEG_ROOT}/refcoco+/instances_refcoco+_val.json"),
    "refcoco+_testA": (f"{_REF_SEG_ROOT}/train2014",
                        f"{_REF_SEG_ROOT}/refcoco+/instances_refcoco+_testA.json"),
    "refcoco+_testB": (f"{_REF_SEG_ROOT}/train2014",
                        f"{_REF_SEG_ROOT}/refcoco+/instances_refcoco+_testB.json"),
    "refcocog_train": (f"{_REF_SEG_ROOT}/train2014",
                         f"{_REF_SEG_ROOT}/refcocog/instances_refcocog_train.json"),
    "refcocog_val": (f"{_REF_SEG_ROOT}/train2014",
                         f"{_REF_SEG_ROOT}/refcocog/instances_refcocog_val.json"),
    "refcocog_test": (f"{_REF_SEG_ROOT}/train2014",
                         f"{_REF_SEG_ROOT}/refcocog/instances_refcocog_test.json"),
}

# ==== Predefined splits for robot-domain referring segmentation datasets ===========
_ROBOT_DATA_ROOT = "/mnt/e/Task_Infer_VLM/VLM_dataset"

_PREDEFINED_SPLITS_ROBOREFIT = {
    "roborefit_train": (_ROBOT_DATA_ROOT,
                         f"{_ROBOT_DATA_ROOT}/RoboRefit/train/instances_roborefit_train.json"),
    "roborefit_testA": (_ROBOT_DATA_ROOT,
                         f"{_ROBOT_DATA_ROOT}/RoboRefit/testA/instances_roborefit_testA.json"),
    "roborefit_testB": (_ROBOT_DATA_ROOT,
                         f"{_ROBOT_DATA_ROOT}/RoboRefit/testB/instances_roborefit_testB.json"),
    # same images/masks, `expressions_ko` field added by
    # utils/translate_expressions_ko.py (offline, one-time MT of the
    # training data only -- inference-time commands are never translated)
    "roborefit_train_ko": (_ROBOT_DATA_ROOT,
                            f"{_ROBOT_DATA_ROOT}/RoboRefit/train/instances_roborefit_train_ko.json"),
    "roborefit_testA_ko": (_ROBOT_DATA_ROOT,
                            f"{_ROBOT_DATA_ROOT}/RoboRefit/testA/instances_roborefit_testA_ko.json"),
    "roborefit_testB_ko": (_ROBOT_DATA_ROOT,
                            f"{_ROBOT_DATA_ROOT}/RoboRefit/testB/instances_roborefit_testB_ko.json"),
}

_ROBOGROUND_PNP_ROOT = f"{_ROBOT_DATA_ROOT}/RoboGround_PnP"

_PREDEFINED_SPLITS_ROBOGROUND_PNP = {
    # Merged across 4 RoboCasa PnP task variants (SinkToCounter/CounterToCab/
    # CounterToMicrowave/CounterToStove) for place-target diversity
    # (plate/cabinet/microwave/pan) -- see utils/extract_roboground_pnp.py
    # and utils/translate_roboground_pnp.py. image_root must be
    # RoboGround_PnP itself since each annotation's file_name is already
    # "images/<task>/<demo>.png" relative to that directory.
    "roboground_pnp_train_ko": (_ROBOGROUND_PNP_ROOT,
                                 f"{_ROBOGROUND_PNP_ROOT}/instances_roboground_pnp_merged_train_ko.json"),
    "roboground_pnp_test_ko": (_ROBOGROUND_PNP_ROOT,
                                f"{_ROBOGROUND_PNP_ROOT}/instances_roboground_pnp_merged_test_ko.json"),
}

# OCID-Ref uses the "multiple" split variant of OCID_VLG's referring expressions
# (matches the grouping used by the LISA OCID-VLG loader). Change the middle
# path component to "unique" / "novel-classes" / "novel-instances" to switch.
_PREDEFINED_SPLITS_OCIDREF = {
    "ocidref_train": (_ROBOT_DATA_ROOT,
                       f"{_ROBOT_DATA_ROOT}/OCID_VLG/multiple/instances_ocidref_train.json"),
    "ocidref_val": (_ROBOT_DATA_ROOT,
                     f"{_ROBOT_DATA_ROOT}/OCID_VLG/multiple/instances_ocidref_val.json"),
    "ocidref_test": (_ROBOT_DATA_ROOT,
                      f"{_ROBOT_DATA_ROOT}/OCID_VLG/multiple/instances_ocidref_test.json"),
    "ocidref_train_ko": (_ROBOT_DATA_ROOT,
                          f"{_ROBOT_DATA_ROOT}/OCID_VLG/multiple/instances_ocidref_train_ko.json"),
    "ocidref_val_ko": (_ROBOT_DATA_ROOT,
                        f"{_ROBOT_DATA_ROOT}/OCID_VLG/multiple/instances_ocidref_val_ko.json"),
    "ocidref_test_ko": (_ROBOT_DATA_ROOT,
                         f"{_ROBOT_DATA_ROOT}/OCID_VLG/multiple/instances_ocidref_test_ko.json"),
}


def register_all_ref_ytvos_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REF_YTVOS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ref_ytvos_instances(
            key,
            _get_ref_ytvos_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ref_coco_2014(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REF_COCO_2014.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            {},
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_roborefit(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ROBOREFIT.items():
        register_coco_instances(key, {}, json_file, image_root)


def register_all_ocidref(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OCIDREF.items():
        register_coco_instances(key, {}, json_file, image_root)


def register_all_roboground_pnp(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ROBOGROUND_PNP.items():
        register_coco_instances(key, {}, json_file, image_root)


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = './'
    register_all_ref_ytvos_2021(_root)
    register_all_ref_coco_2014(_root)
    register_all_roborefit(_root)
    register_all_ocidref(_root)
    register_all_roboground_pnp(_root)