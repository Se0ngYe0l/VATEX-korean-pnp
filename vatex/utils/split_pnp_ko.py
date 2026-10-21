"""Split one compound Korean pick+place command (e.g. "맥주를 파란 바구니에
넣어") into the two short referring phrases VATEX actually needs, since it
is a single-target model -- one expression in, one mask out. Feeding the
same compound sentence twice cannot work: nothing in the text tells the
model which of the two calls wants the pick object vs. the place location,
so both calls deterministically return the identical mask.
"""
from .noun_phrase_ko import (
    _kiwi,
    _noun_phrase_spans,
    _OBJECT_PARTICLES,
    _OBLIQUE_PARTICLES,
)

# Trailing bound nouns that mark relative position ("위"="on top of",
# "안"/"속"="inside", "옆"="next to" ...) rather than naming the place
# object itself. Training vocab (접시/캐비닛/전자레인지 ...) is bare nouns,
# so strip these to stay in-distribution, e.g. "햇반 위" -> "햇반".
_RELATIVE_POSITION_NOUNS = {"위", "아래", "밑", "안", "속", "옆", "앞", "뒤"}


def split_pnp_instruction_ko(text):
    """Returns (pick_phrase, place_phrase), or (None, None) if either
    can't be found (e.g. the sentence doesn't actually name two objects).
    """
    tokens = list(_kiwi.tokenize(text))
    if not tokens:
        return None, None

    spans = _noun_phrase_spans(tokens, text)

    def particle_after(end_idx):
        return tokens[end_idx + 1].tag if end_idx + 1 < len(tokens) else None

    pick_phrase = None
    place_phrase = None
    for span_text, span_start, span_end in spans:
        tag = particle_after(span_end)
        if pick_phrase is None and tag in _OBJECT_PARTICLES:
            pick_phrase = span_text
        elif place_phrase is None and tag in _OBLIQUE_PARTICLES:
            # drop a trailing relative-position noun, e.g. "햇반 위" -> "햇반"
            last_tok = tokens[span_end]
            if last_tok.tag in {"NNG", "NNB"} and last_tok.form in _RELATIVE_POSITION_NOUNS and span_end > span_start:
                char_end = tokens[span_end - 1].start + tokens[span_end - 1].len
                place_phrase = text[tokens[span_start].start:char_end]
            else:
                place_phrase = span_text

    return pick_phrase, place_phrase
