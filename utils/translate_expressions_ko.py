"""One-time offline translation of RoboRefit / OCID-Ref referring
expressions (English) into Korean, using a local NLLB-200 model.

This only touches training DATA (translated once, offline); inference-time
user commands are never machine-translated -- see vatex_model.py's
multilingual_clip branch.

NLLB mistranslates some category nouns badly and consistently when
translated in-context (e.g. RoboRefit's "cone" -> "콩고"/"피자" in ~96% of
occurrences). To avoid this, each annotation's known ground-truth category
name (from its `category_id`) is swapped for a placeholder token before
translation, then the placeholder is replaced with a verified glossary term
afterward -- so NLLB only has to translate the surrounding sentence
structure, never the tricky domain noun itself. The particle immediately
following the inserted term is also corrected for batchim (받침), since
NLLB picks the particle allomorph based on the placeholder's (wrong)
phonetic shape.

Each annotation's `expressions` list is preserved as-is, and a parallel
`expressions_ko` list (same length, same order) is added.
"""
import json
import re
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ko_category_glossary import (
    ROBOREFIT_GLOSSARY, OCIDREF_GLOSSARY, CATEGORY_PATTERN_ALIASES,
    has_batchim, PARTICLE_PAIRS,
)

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
BATCH_SIZE = 128
MAX_NEW_TOKENS = 64
PLACEHOLDER = "ZZOBJ"

SPLITS = [
    ("/mnt/e/Task_Infer_VLM/VLM_dataset/RoboRefit/train/instances_roborefit_train.json", ROBOREFIT_GLOSSARY),
    ("/mnt/e/Task_Infer_VLM/VLM_dataset/RoboRefit/testA/instances_roborefit_testA.json", ROBOREFIT_GLOSSARY),
    ("/mnt/e/Task_Infer_VLM/VLM_dataset/RoboRefit/testB/instances_roborefit_testB.json", ROBOREFIT_GLOSSARY),
    ("/mnt/e/Task_Infer_VLM/VLM_dataset/OCID_VLG/multiple/instances_ocidref_train.json", OCIDREF_GLOSSARY),
    ("/mnt/e/Task_Infer_VLM/VLM_dataset/OCID_VLG/multiple/instances_ocidref_val.json", OCIDREF_GLOSSARY),
    ("/mnt/e/Task_Infer_VLM/VLM_dataset/OCID_VLG/multiple/instances_ocidref_test.json", OCIDREF_GLOSSARY),
]


def out_path(path):
    return path.replace(".json", "_ko.json")


def _name_pattern(name):
    # OCID-Ref category names use underscores (e.g. "food_can",
    # "cereal_box") but the actual expression text always uses a plain
    # space ("food can"), so a literal-underscore regex never matches --
    # accept either a space or an underscore at each word boundary.
    return re.escape(name).replace("_", "[ _]")


def substitute_category(text, category_name, base_term):
    """Replace the category's English name (or, if present, a longer
    compound alias like "cocoa powder" / "glue stick" / "tennis ball" --
    see CATEGORY_PATTERN_ALIASES -- tried first/longest) with PLACEHOLDER.
    Returns (substituted_text, korean_term_or_None)."""
    for eng_pattern, term in CATEGORY_PATTERN_ALIASES.get(category_name, []):
        pattern = r"\b" + _name_pattern(eng_pattern) + r"s?\b"
        new_text, n = re.subn(pattern, PLACEHOLDER, text, count=1, flags=re.IGNORECASE)
        if n > 0:
            return new_text, term
    pattern = r"\b" + _name_pattern(category_name) + r"s?\b"
    new_text, n = re.subn(pattern, PLACEHOLDER, text, count=1, flags=re.IGNORECASE)
    return (new_text, base_term) if n > 0 else (text, None)


# checked longest-first so "으로" matches before its "로" suffix does
_PARTICLE_CANDIDATES = sorted(PARTICLE_PAIRS.keys(), key=len, reverse=True)


def apply_glossary(raw_text, korean_term):
    """Replace PLACEHOLDER with `korean_term` and, using plain surface-string
    matching (not a tokenizer -- Kiwi mis-segments 를/는 after some words,
    e.g. splitting "원뿔를" into a spurious noun "르" + "ᆯ"), correct the
    case particle immediately following it: NLLB picked the particle
    allomorph based on the placeholder's Latin shape, not the real Korean
    word's batchim (받침)."""
    idx = raw_text.find(PLACEHOLDER)
    if idx == -1:
        return None  # NLLB dropped/mangled the placeholder; caller falls back
    rest = raw_text[idx + len(PLACEHOLDER):]
    for cand in _PARTICLE_CANDIDATES:
        if rest.startswith(cand):
            with_b, without_b = PARTICLE_PAIRS[cand]
            correct = with_b if has_batchim(korean_term) else without_b
            rest = correct + rest[len(cand):]
            break
    return raw_text[:idx] + korean_term + rest


def main():
    print("loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    kor_bos = tok.convert_tokens_to_ids("kor_Hang")

    # NOTE: `sub_text` (post-placeholder-substitution English) is only a
    # valid dedup key for the *raw NLLB call* -- the surrounding sentence
    # translates the same regardless of which category filled the
    # placeholder. It is NOT a valid key for the *final* Korean text: two
    # different categories can share an identical template (e.g. both
    # "cone" and "tennis" appearing as "I want to grasp the yellow {cat}")
    # and collapse to the same sub_text, so the glossary term must be
    # (re)applied per-annotation below, never cached under `sub_text` alone.
    datasets = {}
    unique_texts = set()
    n_substituted = 0
    n_total = 0
    for path, glossary in SPLITS:
        with open(path) as f:
            d = json.load(f)
        datasets[path] = (d, glossary)
        cat_names = {c["id"]: c["name"] for c in d["categories"]}
        for ann in d["annotations"]:
            cat_name = cat_names.get(ann["category_id"])
            base_term = glossary.get(cat_name) if cat_name else None
            for expr in ann["expressions"]:
                n_total += 1
                sub_text, final_term = (substitute_category(expr, cat_name, base_term) if base_term
                                         else (expr, None))
                unique_texts.add(sub_text)
                n_substituted += final_term is not None

    unique_texts = sorted(unique_texts, key=len)
    print(f"{len(unique_texts)} unique (post-substitution) expressions to translate", flush=True)
    print(f"{n_substituted}/{n_total} expression occurrences have a category placeholder substitution", flush=True)

    raw_translations = {}
    t_start = time.time()
    for i in range(0, len(unique_texts), BATCH_SIZE):
        batch = unique_texts[i:i + BATCH_SIZE]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **inputs, forced_bos_token_id=kor_bos,
                max_new_tokens=MAX_NEW_TOKENS, num_beams=1,
            )
        out = tok.batch_decode(gen, skip_special_tokens=True)
        for src, tgt in zip(batch, out):
            raw_translations[src] = tgt
        if (i // BATCH_SIZE) % 20 == 0:
            done = i + len(batch)
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(unique_texts) - done) / rate if rate > 0 else float("nan")
            print(f"  {done}/{len(unique_texts)}  ({rate:.1f} sent/s, ETA {eta:.0f}s)", flush=True)

    print(f"translation done in {time.time() - t_start:.1f}s", flush=True)

    fallback_count = 0
    with open("/home/seongyeol/VATEX/utils/expr_translations_ko.json", "w", encoding="utf-8") as f:
        json.dump({"raw": raw_translations}, f, ensure_ascii=False)

    for path, (d, glossary) in datasets.items():
        cat_names = {c["id"]: c["name"] for c in d["categories"]}
        for ann in d["annotations"]:
            cat_name = cat_names.get(ann["category_id"])
            base_term = glossary.get(cat_name) if cat_name else None
            ko_list = []
            for expr in ann["expressions"]:
                sub_text, final_term = (substitute_category(expr, cat_name, base_term) if base_term
                                         else (expr, None))
                raw = raw_translations[sub_text]
                if final_term is None:
                    ko_list.append(raw)
                    continue
                fixed = apply_glossary(raw, final_term)
                if fixed is None:
                    fallback_count += 1
                    fixed = f"{final_term} {raw}"  # degrade gracefully, term still correct
                ko_list.append(fixed)
            ann["expressions_ko"] = ko_list
        op = out_path(path)
        with open(op, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        print(f"wrote {op}", flush=True)
    print(f"placeholder-lost fallback used for {fallback_count}/{n_substituted} substituted occurrences", flush=True)


if __name__ == "__main__":
    main()
