"""Verified English->Korean glossary for the RoboRefit / OCID-Ref category
sets, keyed by the exact category `name` string in each dataset's JSON.
Used to force-correct category vocabulary in the NLLB-translated
`expressions_ko` field via placeholder substitution (see
translate_expressions_ko.py) instead of trusting NLLB with these nouns
directly -- spot-checking found NLLB mistranslates some of them badly and
consistently (e.g. "cone" -> "콩고"/"피자" in ~96% of RoboRefit occurrences).
"""

ROBOREFIT_GLOSSARY = {
    "adhesive tape": "테이프",
    "apple": "사과",
    "avocado": "아보카도",
    "banana": "바나나",
    "biscuit bag": "과자 봉지",
    "body lotion": "바디로션",
    "book": "책",
    "bottled facial cleanser": "클렌저",
    "camel": "낙타",
    "chimpanzee": "침팬지",
    "chocolate milk": "초코우유",
    "cocoa": "코코아",
    "coffee": "커피",
    "coffee powder": "커피 가루",
    "cola": "콜라",
    "cone": "원뿔",
    "cow": "소",
    "cylinder": "원기둥",
    "detergent": "세제",
    "dinosaur": "공룡",
    "disposable cup": "일회용 컵",
    "electronic thermometer": "전자 체온계",
    "elephant": "코끼리",
    "facial cleanser": "클렌저",
    "facial wash": "폼클렌징",
    "giraffe": "기린",
    "glass": "유리컵",
    "glue": "풀",  # bare fallback; "glue stick" (the common case) uses CATEGORY_PATTERN_ALIASES
    "juice": "주스",
    "lion": "사자",
    "milk": "우유",
    "mint": "민트",
    "mouse": "마우스",
    "mouthwash": "구강청결제",
    "orange": "오렌지",
    "panda": "판다",
    "pear": "배",
    "pill bottle": "알약 병",
    "pill box": "알약 케이스",
    "polar bear": "북극곰",
    "rectangular box": "직사각형 상자",
    "remote control": "리모컨",
    "round pear": "둥근 배",
    "rubik": "루빅스 큐브",
    "sachima": "사치마",
    "scissor": "가위",
    "screwdriver": "드라이버",
    "shampoo": "샴푸",
    "soap": "비누",
    "square box": "사각 상자",
    "strawberry": "딸기",
    "teddy bear": "테디베어",
    "tennis": "테니스공",
    "thermometer": "체온계",
    "tiger": "호랑이",
    "tomato": "토마토",
    "toothbrush": "칫솔",
    "toothpaste": "치약",
    "toothpaste box": "치약 상자",
    "triangle cake": "삼각형 케이크",
    "umbrella": "우산",
    "vitamin bottle": "비타민 병",
    "water bottle": "물병",
    "wire cutter": "와이어커터",
    "yogurt": "요거트",
}

# Some category names are a modifying head word that, in a large fraction
# of expressions, actually appears as part of a longer compound (e.g.
# "cocoa" -> "cocoa powder", "glue" -> "glue stick", "tennis" -> "tennis
# ball" in ~50% of cases). Matching only the bare category name there would
# leave the completing word ("powder"/"stick"/"ball") dangling outside the
# placeholder, where NLLB translates it separately and often duplicates
# meaning already carried by the glossary term (e.g. "테니스공 공"). Each
# entry is a list of (english_pattern, korean_term) tried longest-pattern-
# first; if none match, callers fall back to the base glossary entry.
CATEGORY_PATTERN_ALIASES = {
    "cocoa": [("cocoa powder", "코코아 가루")],
    "glue": [("glue stick", "딱풀")],
    "tennis": [("tennis ball", "테니스공")],
}


OCIDREF_GLOSSARY = {
    "apple": "사과",
    "ball": "공",
    "banana": "바나나",
    "bell_pepper": "피망",
    "binder": "바인더",
    "bowl": "그릇",
    "cereal_box": "시리얼 상자",
    "coffee_mug": "머그컵",
    "flashlight": "손전등",
    "food_bag": "식품 봉지",
    "food_box": "식품 상자",
    "food_can": "식품 캔",
    "glue_stick": "딱풀",
    "hand_towel": "손수건",
    "instant_noodles": "라면",
    "keyboard": "키보드",
    "kleenex": "티슈",
    "lemon": "레몬",
    "lime": "라임",
    "marker": "마커",
    "orange": "오렌지",
    "peach": "복숭아",
    "pear": "배",
    "potato": "감자",
    "shampoo": "샴푸",
    "soda_can": "탄산음료 캔",
    "sponge": "스펀지",
    "stapler": "스테이플러",
    "tomato": "토마토",
    "toothpaste": "치약",
}


def has_batchim(word: str) -> bool:
    """True if the last syllable of a Korean word ends in a consonant
    (종성/batchim), which determines which particle allomorph to use
    (을 vs 를, 이 vs 가, 은 vs 는, 과 vs 와, 으로 vs 로)."""
    word = word.rstrip()
    if not word:
        return False
    ch = word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:  # precomposed Hangul syllable block
        return (code - 0xAC00) % 28 != 0
    return False  # non-Hangul (digits, Latin, punctuation) - treat as no batchim


# particle -> (with_batchim_form, no_batchim_form)
PARTICLE_PAIRS = {
    "을": ("을", "를"), "를": ("을", "를"),
    "이": ("이", "가"), "가": ("이", "가"),
    "은": ("은", "는"), "는": ("은", "는"),
    "과": ("과", "와"), "와": ("과", "와"),
    "으로": ("으로", "로"), "로": ("으로", "로"),
}
