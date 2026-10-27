"""Нормализация русского текста: токены и грубый префиксный стемминг.

Общая для распознавания сущностей и BM25, чтобы «рассрочка» и «по рассрочке»
совпадали одинаково везде.

Синтаксис фраз в aliases:
- обычное слово режется до max(4, len-2) символов («рассрочка» -> «рассроч»);
- «слово*» — точный префикс без обрезки («кредитк*» не ловит «кредит»);
- «=слово» — слово должно совпасть целиком («=плати» не ловит «платить»);
- числа сравниваются целиком.
"""

import re

WORD_RE = re.compile(r"[a-zа-я0-9]+")
PHRASE_WORD_RE = re.compile(r"=?[a-zа-я0-9]+\*?")


def norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokens(text: str) -> list[str]:
    """Содержательные токены: слова длиннее 2 букв и любые числа."""
    return [w for w in WORD_RE.findall(norm(text)) if len(w) > 2 or w.isdigit()]


def stem(word: str) -> str:
    return word if word.isdigit() else word[: max(4, len(word) - 2)]


def phrase_stems(phrase: str) -> list[str]:
    out = []
    for w in PHRASE_WORD_RE.findall(norm(phrase)):
        core = w.strip("=*")
        if len(core) > 2 or core.isdigit():
            out.append(w if w[0] == "=" or w[-1] == "*" or core.isdigit() else stem(w))
    return out


def word_matches(pattern: str, word: str) -> bool:
    if pattern.startswith("="):
        return word == pattern[1:]
    if pattern.endswith("*"):
        return word.startswith(pattern[:-1])
    if pattern.isdigit():
        return word == pattern
    return word.startswith(pattern)


def contains_all(phrase: str, words: list[str]) -> bool:
    """Все слова фразы есть в вопросе, в любом порядке (для терминов и событий)."""
    stems = phrase_stems(phrase)
    return bool(stems) and all(any(word_matches(s, w) for w in words) for s in stems)


def find_spans(phrase: str, words: list[str]) -> list[tuple[int, int]]:
    """Позиции, где слова фразы идут подряд и по порядку (для названий продуктов и банков)."""
    stems = phrase_stems(phrase)
    n = len(stems)
    if not n:
        return []
    return [(i, i + n) for i in range(len(words) - n + 1)
            if all(word_matches(s, words[i + j]) for j, s in enumerate(stems))]
