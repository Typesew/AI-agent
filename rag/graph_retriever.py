"""Графовый ретривер: по найденным в вопросе сущностям обходит граф на 1–2 хопа
и возвращает факты со ссылкой на документ, пункт и дату редакции.

Два режима:
- продукт назван («по карте Полярная 120») — факты этого продукта;
- продукт не назван («какие кредитки сохраняют грейс при снятии наличных») —
  все продукты нужного типа, но только факты по найденным терминам и событиям.
"""

from dataclasses import dataclass, field
from pathlib import Path

from rag.entities import Entities, EntityRecognizer
from rag.graph_store import GraphStore

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "graph.kuzu"

# Событие в вопросе подтягивает связанные: «снятие наличных» -> исключения и комиссии
# «сверх лимита» (Опорная 55: снятие в грейсе только до 50 000 ₽), «пропуск платежа» ->
# «пропуск двух платежей» (по рассрочкам последствия обычно наступают со второго).
RELATED_EVENTS = {
    "ev_cash": ["ev_limit_exceeded"],
    "ev_transfer": ["ev_limit_exceeded"],
    "ev_missed_payment": ["ev_missed_two"],
}
MAX_FACTS = 40


@dataclass
class Fact:
    product: str
    kind: str  # condition | fee | exception | cancels | triggers | definition
    text: str
    node_id: str
    term_id: str | None = None
    event_id: str | None = None
    path: list[tuple[str, str, str]] = field(default_factory=list)  # для DOT / Mermaid
    rel_section: str = ""  # пункт, где написана связь события с условием
    product_id: str = ""
    source: dict = field(default_factory=dict)  # {file, section, date}


@dataclass
class GraphResult:
    entities: Entities
    facts: list[Fact]
    table: str = ""


class GraphRetriever:
    def __init__(self, db_path: Path = DEFAULT_DB, recognizer: EntityRecognizer | None = None):
        self.store = GraphStore(db_path)
        self.recognizer = recognizer or EntityRecognizer()
        self.products = {r[0]: r[1] for r in self.store.catalog("Product")}
        self.terms = {r[0]: {"name": r[1], "value": r[2]} for r in self.store.catalog("Term")}
        self.product_type = self.store.product_types()
        self._sources: dict[str, list] = {}

    def source(self, node_id: str, section: str = "") -> dict:
        if node_id not in self._sources:
            self._sources[node_id] = self.store.sources(node_id)
        rows = self._sources[node_id]
        if not rows:
            return {}
        file, sec, date = rows[0]
        return {"file": file, "section": section or sec, "date": date}

    def product_facts(self, pid: str) -> list[Fact]:
        p = self.products[pid]
        facts = []
        for cid, name, value, tid in self.store.conditions(pid):
            facts.append(Fact(p, "condition", f"{p}: {name} — {value}", cid, tid, path=[(p, "HAS_CONDITION", name)]))
        for fid, name, value, tid in self.store.fees(pid):
            facts.append(Fact(p, "fee", f"{p}: {name} — {value}", fid, tid, path=[(p, "CHARGES", name)]))
        for xid, cname, xname, tid, eid in self.store.exceptions(pid):
            facts.append(Fact(p, "exception", f"{p}: {cname} — не распространяется на: {xname}", xid, tid, eid,
                              path=[(p, "HAS_CONDITION", cname), (cname, "HAS_EXCEPTION", xname)]))
        for cid, eid, ename, cname, tid, sec in self.store.cancels(pid):
            facts.append(Fact(p, "cancels", f"{p}: событие «{ename}» ОТМЕНЯЕТ «{cname}»", cid, tid, eid,
                              path=[(p, "HAS_CONDITION", cname), (ename, "CANCELS", cname)], rel_section=sec))
        for nid, eid, ename, nname, value, tid, sec in self.store.triggers(pid):
            facts.append(Fact(p, "triggers", f"{p}: событие «{ename}» ВЛЕЧЁТ «{nname}» — {value}", nid, tid, eid,
                              path=[(p, "HAS_CONDITION/CHARGES", nname), (ename, "TRIGGERS", nname)], rel_section=sec))
        for f in facts:
            f.product_id = pid
        return facts

    def retrieve(self, question: str, entities: Entities | None = None) -> GraphResult:
        ent = entities or self.recognizer.recognize(question)
        if ent.is_empty():
            return GraphResult(ent, [])

        if ent.products:
            pids = [p for p in ent.products if p in self.products]
        elif ent.type_terms:
            pids = [p for p, t in self.product_type.items() if t in ent.type_terms]
        else:
            pids = list(self.products)

        terms = ent.content_terms
        events = set(ent.events) | {r for e in ent.events for r in RELATED_EVENTS.get(e, [])}

        def keep(f: Fact) -> bool:
            if not (terms or events):
                return True
            if ent.products:
                if f.term_id in terms or f.event_id in events:
                    return True
                # «Что будет, если пропущу платёж» про конкретный продукт: показываем
                # все последствия событий — это и есть подводные камни.
                return bool(events and not terms and f.event_id)
            # Продукт не назван: фильтр строже, нужны и событие, и термин.
            if not events:
                return f.term_id in terms
            if not terms:
                return f.event_id in events
            return f.term_id in terms and (f.event_id in events or f.kind in ("condition", "fee"))

        facts = [f for pid in pids for f in self.product_facts(pid) if keep(f)]
        # Узел, который уже показан как последствие события, не повторяем простым условием.
        triggered = {f.node_id for f in facts if f.kind == "triggers"}
        facts = [f for f in facts if not (f.kind in ("condition", "fee") and f.node_id in triggered)]
        facts.sort(key=lambda f: (f.product, ["condition", "fee", "exception", "cancels", "triggers"].index(f.kind)))
        facts = facts[:MAX_FACTS]

        for tid in terms:
            t = self.terms[tid]
            facts.insert(0, Fact("Глоссарий", "definition", f"{t['name']} — {t['value']}", tid, tid))
        for f in facts:
            node = f.product_id if f.rel_section else f.node_id
            f.source = self.source(node, f.rel_section)

        table = self.compare_table(pids, facts, ent) if len(ent.products) >= 2 else ""
        return GraphResult(ent, facts, table)

    def compare_table(self, pids: list[str], facts: list[Fact], ent: Entities) -> str:
        names = [self.products[p] for p in pids]
        rows: dict[str, dict[str, list[str]]] = {}
        for f in facts:
            if f.kind in ("condition", "fee", "triggers") and f.term_id and (
                    not ent.content_terms or f.term_id in ent.content_terms):
                row = rows.setdefault(self.terms[f.term_id]["name"], {})
                row.setdefault(f.product, []).append(f.text.split(": ", 1)[1])
        if not rows:
            return ""
        lines = ["| Параметр | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for term, row in rows.items():
            lines.append(f"| {term} | " + " | ".join("; ".join(row.get(n, ["нет данных"])) for n in names) + " |")
        return "\n".join(lines)


def _edges(facts: list[Fact]) -> list[tuple[str, str, str]]:
    seen, out = set(), []
    for f in facts:
        for e in f.path:
            if e not in seen:
                seen.add(e)
                out.append(e)
    return out


def to_mermaid(facts: list[Fact]) -> str:
    ids: dict[str, str] = {}
    lines = []
    for a, rel, b in _edges(facts):
        for name in (a, b):
            ids.setdefault(name, f"n{len(ids)}")
        lines.append(f"  {ids[a]} -->|{rel}| {ids[b]}")
    defs = [f'  {i}["{name.replace(chr(34), "")}"]' for name, i in ids.items()]
    return "\n".join(["graph LR"] + defs + lines)


REL_RU = {"HAS_CONDITION": "условие", "CHARGES": "берёт", "HAS_CONDITION/CHARGES": "условие или комиссия",
          "HAS_EXCEPTION": "не распространяется на", "CANCELS": "отменяет", "TRIGGERS": "влечёт"}
REL_COLORS = {"CANCELS": "#c0392b", "TRIGGERS": "#d68910", "HAS_EXCEPTION": "#7d3c98"}


def to_dot(facts: list[Fact]) -> str:
    """DOT для st.graphviz_chart: рисуется локально, без CDN.

    Узлы различаются по продукту: «Льготный период» у «100 дней» и у «Кедр 78» — два разных
    узла, иначе исключения разных карт сливаются на схеме. События общие для всех продуктов.
    """
    products = {f.product_id for f in facts if f.product_id}
    many = len(products) > 1
    nodes: dict[tuple, tuple[str, str]] = {}  # ключ -> (id в DOT, подпись и стиль)
    edges: list[tuple[str, str, str]] = []

    def node(key: tuple, label: str, fill: str) -> str:
        if key not in nodes:
            nodes[key] = (f"n{len(nodes)}", f'label="{label.replace(chr(34), chr(39))}", fillcolor="{fill}"')
        return nodes[key][0]

    def entity(f: Fact, name: str, fill: str = "#eef3f8") -> str:
        label = f"{name}\\n({f.product})" if many else name
        return node(("e", f.product_id, name), label, fill)

    for f in facts:
        for a, rel, b in f.path:
            if rel in ("CANCELS", "TRIGGERS"):
                src, dst = node(("ev", a), a, "#f3e8fa"), entity(f, b)
            elif rel == "HAS_EXCEPTION":
                src, dst = entity(f, a), entity(f, b, "#fbe9e7")
            else:
                src, dst = node(("p", f.product_id), a, "#dff1ec"), entity(f, b)
            if (src, rel, dst) not in edges:
                edges.append((src, rel, dst))

    lines = ["digraph G {", '  rankdir=LR; node [shape=box, style="rounded,filled", fontname="Arial"];']
    lines += [f"  {nid} [{style}];" for nid, style in nodes.values()]
    lines += [f'  {a} -> {b} [label="{REL_RU.get(rel, rel)}", color="{REL_COLORS.get(rel, "#555555")}", fontname="Arial"];'
              for a, rel, b in edges]
    lines.append("}")
    return "\n".join(lines)
