"""Схема графа знаний: узлы, допустимые связи и DDL для Kuzu.

Одна и та же схема используется при загрузке эталонной разметки, при проверке
ответа LLM-экстрактора и в промпте экстрактора, поэтому она живёт в одном месте.
"""

# Узлы предметной области. У всех одинаковые свойства, так проще и LLM, и коду:
# value хранит суть (сумму, ставку, определение термина), aliases нужны для
# поиска сущностей в вопросе пользователя (через "|").
ENTITY_LABELS = ["Bank", "Product", "Condition", "Fee", "Exception", "Event", "Term"]

# Тип связи -> допустимые пары (откуда, куда).
REL_TYPES = {
    "OFFERED_BY": [("Product", "Bank")],
    "HAS_CONDITION": [("Product", "Condition")],
    "CHARGES": [("Product", "Fee")],
    "HAS_EXCEPTION": [("Condition", "Exception")],
    "CANCELS": [("Event", "Condition")],
    # Не было в исходной схеме: «пропуск платежа» не отменяет условие, а включает
    # штраф или новую ставку. Без этой связи демо про рассрочку не собрать.
    "TRIGGERS": [("Event", "Condition"), ("Event", "Fee")],
    "DEFINED_AS": [("Condition", "Term"), ("Fee", "Term"), ("Product", "Term")],
    # Тоже расширение: к какому событию относится исключение. «Снятие сверх
    # 50 000 ₽ не входит в грейс» — это про ev_limit_exceeded, а не про ev_cash,
    # и по тексту исключения это надёжно не определить.
    "APPLIES_TO": [("Exception", "Event")],
}

# Узлы, которые создаёт LLM-экстрактор. Банки, продукты, термины и события
# известны заранее (front-matter документа и глоссарий).
EXTRACTED_LABELS = ["Condition", "Fee", "Exception"]


def ddl() -> list[str]:
    stmts = [
        f"CREATE NODE TABLE {label}(id STRING, name STRING, value STRING, aliases STRING, PRIMARY KEY(id))"
        for label in ENTITY_LABELS
    ]
    stmts.append("CREATE NODE TABLE Document(id STRING, title STRING, date STRING, file STRING, PRIMARY KEY(id))")
    for rel, pairs in REL_TYPES.items():
        froms = ", ".join(f"FROM {a} TO {b}" for a, b in pairs)
        # section — пункт документа, где написана сама связь («2.2. Снятие наличных
        # прекращает льготный период»), он может отличаться от пункта узла.
        stmts.append(f"CREATE REL TABLE {rel}({froms}, section STRING)")
    # Любой узел -SOURCE-> Document, в связи хранится пункт документа.
    froms = ", ".join(f"FROM {label} TO Document" for label in ENTITY_LABELS)
    stmts.append(f"CREATE REL TABLE SOURCE({froms}, section STRING)")
    return stmts


def check_edge(rel: str, from_label: str | None, to_label: str | None) -> str | None:
    """Возвращает текст ошибки или None, если связь допустима схемой."""
    if rel not in REL_TYPES:
        return f"неизвестный тип связи {rel}"
    if from_label is None or to_label is None:
        return "ссылка на несуществующий узел"
    if (from_label, to_label) not in REL_TYPES[rel]:
        return f"{rel} не может идти из {from_label} в {to_label}"
    return None
