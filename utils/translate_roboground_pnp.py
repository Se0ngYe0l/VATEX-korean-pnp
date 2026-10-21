"""Translate the merged RoboGround PnP dataset (SinkToCounter + CounterToCab
+ CounterToMicrowave + CounterToStove) pick/place vocabulary into Korean.

Same approach as before: NLLB translates only the short phrases (~130
unique across pick+place), with manual overrides for words verified to be
mistranslated in isolation. The full-sentence expression is *not* passed
through NLLB (unreliable on RoboGround's fixed templates -- e.g. added a
spurious "kiwi peel"); instead each place target has its own fixed Korean
verb template (plate -> "놓아줘", cabinet/microwave/pan -> "넣어줘"), filled
in with the translated pick phrase and the correct object particle
(을/를, chosen by batchim).
"""
import json
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, "utils")
from ko_category_glossary import has_batchim, PARTICLE_PAIRS

IN_PATH = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboGround_PnP/instances_roboground_pnp_merged.json"
OUT_PATH = "/mnt/e/Task_Infer_VLM/VLM_dataset/RoboGround_PnP/instances_roboground_pnp_merged_ko.json"
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

# place target -> (Korean noun, verb ending)
PLACE_INFO = {
    "plate": ("접시", "놓아줘"),
    "cabinet": ("캐비닛", "넣어줘"),
    "microwave": ("전자레인지", "넣어줘"),
    "pan": ("팬", "넣어줘"),
}

# Verified corrections for phrases NLLB got wrong or dropped entirely when
# translated in isolation (same failure class as RoboRefit's "cone" bug).
OVERRIDES = {
    "pear": "배",
    "lime": "라임",
    "peach": "복숭아",
    "carrot": "당근",
    "squash": "호박",
    "mushroom": "버섯",
    "eggplant": "가지",
    "cucumber": "오이",
    "tangerine": "귤",
    "bell pepper": "피망",
    "mango": "망고",
    "sweet potato": "고구마",
    "wedge object": "쐐기 모양 물체",
    "tapered object": "가늘어지는 물체",
    "scalloped object": "물결무늬 물체",
    "cone object": "원뿔형 물체",
    "pear-shaped object": "배 모양의 물체",
    "oblong object": "길쭉한 타원형 물체",
    "can": "캔",
    "jam": "잼",
    "jug": "물병",
    "ladle": "국자",
    "shaker": "양념통",
    "ketchup": "케첩",
    "croissant": "크루아상",
    "teapot": "찻주전자",
    "bar soap": "고체 비누",
    "wax object": "왁스 물체",
    "teal object": "청록색 물체",
    "gold object": "금색 물체",
    "boxed drink": "박스 포장 음료",
    "metal object": "금속 물체",
    "dough object": "반죽 물체",
    "glass object": "유리 물체",
    "edible object": "식용 물체",
    "silver object": "은색 물체",
    "ribbed object": "골이 있는 물체",
    "bronze object": "청동색 물체",
    "convex object": "볼록한 물체",
    "frosting object": "설탕 옷을 입힌 물체",
    "kettle electric": "전기 주전자",
    "kettle non electric": "비전기 주전자",
}


def korean_object_particle(korean_phrase):
    with_b, without_b = PARTICLE_PAIRS["을"]
    return with_b if has_batchim(korean_phrase) else without_b


def main():
    with open(IN_PATH) as f:
        d = json.load(f)

    place_by_image = {a["image_id"]: a["expressions"][0] for a in d["annotations"] if a["category_id"] == 2}

    # short-phrase vocab: pick phrases (index 0) + place nouns (fixed dict)
    pick_phrases = sorted({a["expressions"][0] for a in d["annotations"] if a["category_id"] == 1}, key=len)
    print(f"{len(pick_phrases)} unique pick phrases to translate")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    kor_bos = tok.convert_tokens_to_ids("kor_Hang")

    pick_ko = {}
    batch_size = 64
    for i in range(0, len(pick_phrases), batch_size):
        batch = pick_phrases[i:i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, forced_bos_token_id=kor_bos, max_new_tokens=64, num_beams=1)
        out = tok.batch_decode(gen, skip_special_tokens=True)
        for src, tgt in zip(batch, out):
            pick_ko[src] = OVERRIDES.get(src, tgt)

    for src, tgt in pick_ko.items():
        flag = " (수동 교정)" if src in OVERRIDES else ""
        print(f"  {src}  ->  {tgt}{flag}")

    for a in d["annotations"]:
        a["expressions_ko"] = [pick_ko[a["expressions"][0]]] if a["category_id"] == 1 else [PLACE_INFO[a["expressions"][0]][0]]

    for a in d["annotations"]:
        if a["category_id"] != 1:
            continue
        pick_word = a["expressions_ko"][0]
        place_word_en = place_by_image[a["image_id"]]
        place_noun, verb = PLACE_INFO[place_word_en]
        full_ko = f"{pick_word}{korean_object_particle(pick_word)} 집어서 {place_noun}에 {verb}"
        a["expressions_ko"].append(full_ko)
    pick_full_by_image = {a["image_id"]: a["expressions_ko"][1] for a in d["annotations"] if a["category_id"] == 1}
    for a in d["annotations"]:
        if a["category_id"] != 2:
            continue
        a["expressions_ko"].append(pick_full_by_image[a["image_id"]])

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
