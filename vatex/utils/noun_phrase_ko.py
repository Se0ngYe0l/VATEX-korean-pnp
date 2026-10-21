from kiwipiepy import Kiwi

_kiwi = Kiwi()

# Nouns/pronouns/numerals/foreign-word tags that can form (or extend) a
# noun-phrase span.
_NOUN_TAGS = {"NNG", "NNP", "NNB", "NP", "NR", "SL", "SH", "SN"}
_MODIFIER_END_TAGS = {"ETM"}  # adnominal ending, e.g. "빨간"="빨갛"+"ᆫ"
_DETERMINER_TAGS = {"MM"}     # 관형사, e.g. "이 컵"

# Case-particle tags, used to rank a candidate span as the likely referred
# object -- mirrors the dobj > nsubj > (bare) > pobj priority used for
# English dependency labels in noun_phrase.py.
_OBJECT_PARTICLES = {"JKO"}                # 을/를 - direct object
_SUBJECT_PARTICLES = {"JKS", "JX"}         # 이/가, 은/는 (topic doubles as subject)
_OBLIQUE_PARTICLES = {"JKB", "JKG", "JC"}  # 에/에서/위에/와 etc. - location/other
_ALL_PARTICLES = _OBJECT_PARTICLES | _SUBJECT_PARTICLES | _OBLIQUE_PARTICLES

_BARE_PRONOUNS = {"저", "나", "너", "당신", "그것", "이것", "저것", "우리"}


# suffixes that turn a preceding noun into a verb/adjective stem (e.g.
# 일+하다="to work", 깨끗+하다="to be clean") -- a noun run followed
# directly by one of these is a predicate, not a nominal referent, and
# must not be treated as a candidate noun phrase.
_VERBALIZING_SUFFIXES = {"XSV", "XSA"}


def _noun_phrase_spans(tokens, text):
    """Group contiguous modifier+noun morphemes into noun-phrase spans.

    Returns a list of (text, start_index, end_index) over `tokens`, so the
    caller can inspect the token right after `end_index` for a case
    particle to determine the span's grammatical role. `text` is sliced
    by each token's original character offsets (`.start`/`.len`) rather
    than rejoining morpheme lemmas, so surface forms (e.g. "빨간", not the
    lemma pieces "빨갛"+"ᆫ") and original spacing come out correct.
    """
    n = len(tokens)
    noun_ranges = []
    i = 0
    while i < n:
        if tokens[i].tag in _NOUN_TAGS:
            start = i
            j = i + 1
            while j < n and tokens[j].tag in _NOUN_TAGS:  # compound noun run
                j += 1
            if j < n and tokens[j].tag in _VERBALIZING_SUFFIXES:
                # noun run is really a predicate stem (e.g. "일하는"), not
                # a nominal referent -- skip it entirely
                i = j + 1
                continue
            noun_ranges.append((start, j - 1))
            i = j
        else:
            i += 1

    spans = []
    for start, end in noun_ranges:
        span_start = start
        k = start - 1
        if k >= 0 and tokens[k].tag in _MODIFIER_END_TAGS:
            # absorb one adnominal modifier (its stem + any adverb before
            # it) immediately preceding the noun, e.g. "가장 가까운" 머그컵
            m = k - 1
            while m >= 0 and tokens[m].tag not in _NOUN_TAGS and tokens[m].tag not in _ALL_PARTICLES:
                m -= 1
            span_start = m + 1
        elif k >= 0 and tokens[k].tag in _DETERMINER_TAGS:
            span_start = k
        char_start = tokens[span_start].start
        char_end = tokens[end].start + tokens[end].len
        spans.append((text[char_start:char_end], span_start, end))
    return spans


def extract_noun_phrase_ko(text):
    """Korean counterpart to noun_phrase.extract_noun_phrase.

    Picks the noun phrase most likely to be the referred object, preferring
    (in order): object-marked (을/를), subject/topic-marked (이/가, 은/는),
    a bare noun phrase with no following particle (common in verb-less
    referring captions like "the green block behind the can"), and finally
    a location/oblique-marked phrase (에/에서/위에 ...) as a last resort.
    Bare pronouns are skipped unless nothing else is available.
    """
    tokens = list(_kiwi.tokenize(text))
    if not tokens:
        return text

    spans = _noun_phrase_spans(tokens, text)
    if not spans:
        return text

    def particle_after(end_idx):
        return tokens[end_idx + 1].tag if end_idx + 1 < len(tokens) else None

    def is_bare_pronoun(span_text, span_start, span_end):
        return span_end == span_start and tokens[span_start].tag == "NP" and span_text in _BARE_PRONOUNS

    def pick(predicate, allow_pronoun):
        for span_text, span_start, span_end in spans:
            if not predicate(particle_after(span_end)):
                continue
            if not allow_pronoun and is_bare_pronoun(span_text, span_start, span_end):
                continue
            return span_text
        return None

    priority = (
        lambda tag: tag in _OBJECT_PARTICLES,
        lambda tag: tag in _SUBJECT_PARTICLES,
        lambda tag: tag is None or tag not in _ALL_PARTICLES,
        lambda tag: tag in _OBLIQUE_PARTICLES,
    )
    for allow_pronoun in (False, True):
        for predicate in priority:
            result = pick(predicate, allow_pronoun)
            if result is not None:
                return result

    return spans[-1][0]  # unreachable in practice; last resort
