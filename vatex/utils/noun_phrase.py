import spacy

def extract_noun_phrase(text, doc, need_index=False):
    # text = text.lower()

    # doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    for token in doc:
        if token.head.i == token.i:
            head = token.head

    if head.i not in chunks:
        # Search the root's whole subtree (not just its immediate first
        # child) for the referred object. This matters for constructions
        # like "I want to grasp the ball" where the object is a
        # grandchild (via an xcomp), and for "bring me the lion" where a
        # leading dative pronoun ("me") would otherwise be mistaken for
        # the referent since it precedes the true object positionally.
        # Bare pronouns (I/me/you/it/...) are never a valid referent, so
        # they're excluded unless nothing else is available. Among the
        # rest, object-role tokens (dobj/pobj/attr/oprd) are preferred
        # over subject-role tokens (nsubj), but subjects still win when
        # no object exists (e.g. "a doctor is working in a hospital").
        candidates = [
            t for t in head.subtree
            if t.i != head.i and t.i in chunks and t.pos_ != "PRON"
        ]
        # pobj (object of a preposition) is deliberately excluded from
        # this priority list: it's usually part of a locative/modifier
        # phrase ("in a hospital", "on the shelf") rather than the
        # referent itself, so it's only used as a last resort below.
        obj = None
        for dep in ("dobj", "nsubj", "attr", "oprd", "nsubjpass"):
            obj = next((t for t in candidates if t.dep_ == dep), None)
            if obj is not None:
                break
        if obj is None and candidates:
            obj = candidates[0]
        if obj is None:
            children = list(head.children)
            if children and children[0].i in chunks:
                obj = children[0]
        if obj is not None:
            head = obj
        else:
            if need_index:
                return text, [], text
            else:
                return text

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        return head.text