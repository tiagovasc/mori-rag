"""Exact source spans: readable parents and independently token-bounded children.

The boundary detector is deliberately conservative, not a linguistic parser.
Long unpunctuated sentences fall back to whole-word boundaries explicitly.
No text is reconstructed from token IDs or normalized during splitting.
"""
from bisect import bisect_left, bisect_right
import re


_ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'etc',
    'e.g', 'i.e', 'cf', 'fig', 'figs', 'eq', 'vol', 'no', 'pp', 'al',
    'approx', 'dept', 'inc', 'ltd', 'ed', 'eds', 'rev',
}


def boundaries(text):
    """Return sentence/paragraph end offsets, including their trailing whitespace."""
    result = {len(text): 'end'}
    for match in re.finditer(r'\n[ \t]*\n+', text):
        result[match.end()] = 'paragraph'
    for match in re.finditer(r'[.!?…]+[\"\'”’\)\]]*(?:\s+|$)|[。！？]+[\"\'”’\)\]]*\s*', text):
        punctuation = match.group().rstrip()
        if punctuation.startswith('.'):
            before = text[:match.start()]
            token = re.search(r'([\w.]+)$', before)
            word = token.group(1) if token else ''
            if (word.casefold() in _ABBREVIATIONS or
                    re.fullmatch(r'(?:[A-Za-z]\.)*[A-Za-z]', word) or
                    word.isdigit()):
                continue
        result.setdefault(match.end(), 'sentence')
    return result


def sentence_chunks(text, words, overlap):
    """Approximately words per passage, with at most 25% boundary flexibility.

    Overlap begins at a complete sentence when possible. Only a sentence longer
    than the hard budget is cut at a word boundary. Very short final tails are
    absorbed when doing so stays inside the same 25% allowance.
    """
    tokens = list(re.finditer(r'\S+', text))
    if not tokens:
        return []
    if words < 1 or overlap < 0 or overlap >= words:
        raise ValueError('Invalid passage size or overlap')
    ends = [token.end() for token in tokens]
    natural = boundaries(text)
    natural_positions = sorted(natural)
    result, start, previous_end = [], 0, 0
    while start < len(text):
        start_word = bisect_right(ends, start)
        remaining = len(tokens) - start_word
        hard_words = max(words, int(words * 1.25))
        if remaining <= hard_words:
            end, boundary = len(text), 'end'
        else:
            target = start_word + words
            hard = start_word + hard_words
            candidates = [(position, bisect_right(ends, position), natural[position])
                          for position in natural_positions
                          if position > max(start, previous_end)
                          and start_word + max(1, int(words * .65)) <= bisect_right(ends, position) <= hard]
            if candidates:
                end, _, boundary = min(candidates, key=lambda item: (
                    abs(item[1] - target) - (words * .08 if item[2] == 'paragraph' else 0),
                    item[0]))
            else:
                next_word = min(start_word + words, len(tokens))
                end = tokens[next_word].start() if next_word < len(tokens) else len(text)
                boundary = 'word-fallback'
        if end <= previous_end:
            raise ValueError('Passage splitting did not advance')
        result.append({'seq': len(result), 'start': start, 'end': end,
                       'text': text[start:end], 'boundary': boundary})
        if end == len(text):
            break
        end_word = bisect_right(ends, end)
        target_overlap = max(start_word + 1, end_word - overlap)
        preferred = [(position, bisect_right(ends, position)) for position in natural_positions
                     if start < position < end
                     and 0 < end_word - bisect_right(ends, position) <= max(overlap * 2, 1)]
        if overlap and preferred:
            next_start, _ = min(preferred, key=lambda item: abs(item[1] - target_overlap))
        elif overlap and boundary == 'word-fallback':
            next_start = tokens[target_overlap].start()
        else:
            next_start = end
        start, previous_end = next_start, end
    return result


def token_spans(text, count_tokens, limit):
    """Exact nonoverlapping spans measured with the provider's complete input.

    count_tokens must include document prefixes and special tokens. A word that
    cannot fit is rejected, not shortened or silently removed. Sentence breaks
    are preferred when they use at least 60% of the available word span.
    """
    words = list(re.finditer(r'\S+', text))
    if not words:
        raise ValueError('Cannot split empty embedding text')
    if count_tokens('') >= limit:
        raise ValueError('Embedding prefix/special tokens leave no content budget')
    natural = sorted(boundaries(text))
    starts = [word.start() for word in words] + [len(text)]
    result, start, first_word = [], 0, 0
    while first_word < len(words):
        low, high, accepted = first_word + 1, len(words), None
        while low <= high:
            middle = (low + high) // 2
            end = starts[middle]
            if count_tokens(text[start:end]) <= limit:
                accepted, low = middle, middle + 1
            else:
                high = middle - 1
        if accepted is None:
            raise ValueError('An unbreakable source word exceeds the embedding token budget; no text was truncated')
        end = starts[accepted]
        candidates = [position for position in natural
                      if start < position <= end
                      and bisect_left(starts, position) - first_word >= max(1, int((accepted - first_word) * .6))]
        if candidates and end < len(text):
            candidate = candidates[-1]
            if count_tokens(text[start:candidate]) <= limit:
                end = candidate
        # Recheck the exact final span, never relying on a words/token ratio.
        count = count_tokens(text[start:end])
        if count > limit or end <= start:
            raise ValueError('Embedding span exceeds measured token budget')
        result.append({'seq': len(result), 'start': start, 'end': end,
                       'text': text[start:end], 'tokens': count})
        start = end
        first_word = bisect_left(starts, start)
    if start != len(text):
        raise ValueError('Embedding spans did not cover the complete source')
    return result
