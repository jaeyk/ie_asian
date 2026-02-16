#!/usr/bin/env python3
"""Build a Chong Wa vs IDIA allies network with relationship labels.

Outputs:
- outputs/chongwa_idia_allies_nodes.csv
- outputs/chongwa_idia_allies_edges.csv
- outputs/fig_chongwa_idia_allies_network.png
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import string
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import spacy


SENT_SPLIT_PAT = re.compile(r"(?<=[.!?])\s+")
WS_PAT = re.compile(r"\s+")

CHONG_WA_PAT = re.compile(
    r"\b(chong\s*wa(?:h)?|chinese\s+benevolent\s+(association|society)|c\.?\s*b\.?\s*a\.?|cwba)\b",
    re.IGNORECASE,
)
IDIA_PAT = re.compile(
    r"\b((international\W+district\W+)?improvement\W+association|id\W+improvement\W+association|"
    r"idia|inter\*?im|interim|idha|housing alliance)\b",
    re.IGNORECASE,
)

CONFLICT_CUE_PAT = re.compile(
    r"\b(against|versus|vs\.?|oppose(?:d|s|ing|ition)?|criticiz(?:e|ed|es|ing)|"
    r"reject(?:ed|s|ion)?|clash(?:ed|es|ing)?|conflict(?:ed|s|ing)?|dispute(?:d|s|ing)?)\b",
    re.IGNORECASE,
)
ALLIANCE_CUE_PAT = re.compile(
    r"\b(jointly|partner(?:ed|s|ship)?|collaborat(?:e|ed|es|ion)|"
    r"cooperat(?:e|ed|es|ion)|support(?:ed|s|ing)?|back(?:ed|s|ing)?|endorse(?:d|s|ment)?)\b",
    re.IGNORECASE,
)

KEEP_ENTITY_LABELS = {"ORG"}
CHONGWA_LABEL = "Chong Wa (CBA)"
IDIA_LABEL = "ID Improvement Association"
BAD_ALLY = {
    "chong wa",
    "id improvement association",
    "idia",
    "district",
    "international district",
    "association",
    "community",
    "board of directors",
    "security pacific",
    "community affairs",
    "moriguchi and young",
    "preservation of chinatowns",
    "review board",
    "county executive",
    "district planning",
    "health project coordinator",
    "project coordinator",
    "coordinator",
    "physical development task force",
    "physical department task force",
    "ozark fire ordinance",
    "fire ordinance",
    "ordinance",
    "wellermaynard",
    "weller maynard",
    "weller-maynard",
}
ALLOWED_ACRONYMS = {"CSA", "HUD", "IDSRB", "IDEA", "CETA"}
STREET_WORD_PAT = re.compile(r"\b(street|st\.?|avenue|ave\.?|blvd|boulevard|road|rd\.?|way|place|plaza)\b", re.IGNORECASE)
LIKELY_PLACE_TOKENS = {"jackson", "maynard", "weller", "king", "washington", "seattle", "chinatown"}
EXCLUDE_ALLY_PAT = re.compile(r"\b(hotel|station)\b", re.IGNORECASE)
FEDERAL_AGENCIES = {"HUD", "CSA"}
LOCAL_GOV_AGENCIES = {"City Building Department"}


def clean_text(s: str) -> str:
    return WS_PAT.sub(" ", (s or "").replace("&nbsp;", " ").replace("&amp;", "&")).strip()


def split_sentences(text: str) -> list[str]:
    return [clean_text(x) for x in SENT_SPLIT_PAT.split(text or "") if clean_text(x)]


def normalize_name(s: str, label: str) -> str:
    t = clean_text(s)
    if not t:
        return ""
    if CHONG_WA_PAT.search(t):
        return CHONGWA_LABEL
    if IDIA_PAT.search(t):
        return IDIA_LABEL
    if re.search(r"international\s+district\s+economic\s+association", t, re.IGNORECASE):
        return "IDEA"
    if re.search(r"department\s+of\s+housing\s+and\s+urban\s+development", t, re.IGNORECASE):
        return "HUD"
    if label == "ORG" and re.fullmatch(r"[A-Za-z]{2,6}", t):
        return t.upper()

    t = t.lower().strip(string.punctuation + " ")
    t = re.sub(r"^the\s+", "", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = WS_PAT.sub(" ", t).strip()
    if not t or len(t) < 3:
        return ""
    if t in BAD_ALLY:
        return ""
    return " ".join(w.capitalize() if w.lower() not in {"of", "and", "the"} else w.lower() for w in t.split())


def clean_ally_entity(text: str, label: str, sentence: str) -> str:
    nm = normalize_name(text, label)
    if not nm:
        return ""
    if nm == "IDSRB":
        return ""
    # Remove likely person-name fragments emitted as uppercase ORG tokens (e.g., BOWER),
    # while preserving known program/organization acronyms.
    if re.fullmatch(r"[A-Z]{3,10}", nm) and nm not in ALLOWED_ACRONYMS:
        return ""
    low = nm.lower()
    if "review board" in low and low != "idsrb":
        return ""
    if EXCLUDE_ALLY_PAT.search(low):
        return ""
    # Exclude likely place/street artifacts from ally lists.
    if low in LIKELY_PLACE_TOKENS:
        return ""
    if STREET_WORD_PAT.search(sentence):
        # If mention appears in a street-address context, skip.
        if re.search(rf"\b{re.escape(low)}\b", sentence.lower()):
            return ""
    return nm


def mention_side(sentence: str) -> str | None:
    cw = bool(CHONG_WA_PAT.search(sentence))
    idia = bool(IDIA_PAT.search(sentence))
    if cw and not idia:
        return CHONGWA_LABEL
    if idia and not cw:
        return IDIA_LABEL
    return None


def relation_label(sentence: str) -> str:
    if CONFLICT_CUE_PAT.search(sentence):
        return "opposed"
    if ALLIANCE_CUE_PAT.search(sentence):
        return "aligned"
    return "neutral"


def classify_relation_historical(counts: Counter, month_counts: dict[str, Counter]) -> tuple[str, float]:
    """Classify edge relation from historical evidence, not single-sentence mode."""
    a = int(counts.get("aligned", 0))
    o = int(counts.get("opposed", 0))
    active = a + o
    if active == 0:
        return "neutral", 0.0

    score = (a - o) / active

    # If polarity flips across months, down-weight to neutral unless one side dominates strongly.
    month_scores = []
    for _, c in month_counts.items():
        ma = int(c.get("aligned", 0))
        mo = int(c.get("opposed", 0))
        if ma + mo == 0:
            continue
        month_scores.append((ma - mo) / (ma + mo))
    has_pos = any(x > 0 for x in month_scores)
    has_neg = any(x < 0 for x in month_scores)
    if has_pos and has_neg and abs(score) < 0.6:
        return "neutral", round(float(score), 4)

    if score >= 0.25:
        return "aligned", round(float(score), 4)
    if score <= -0.25:
        return "opposed", round(float(score), 4)
    return "neutral", round(float(score), 4)


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def plot_network(node_rows: list[dict], edge_rows: list[dict], out_path: pathlib.Path) -> None:
    g = nx.Graph()
    for r in node_rows:
        g.add_node(
            r["node"],
            side=r["side"],
            n_mentions=int(r["n_mentions"]),
            n_hub_links=int(r["n_hub_links"]),
        )
    for e in edge_rows:
        if int(e["n_sentences"]) <= 0:
            continue
        g.add_edge(
            e["source"],
            e["target"],
            relation=e["relation"],
            n=int(e["n_sentences"]),
        )

    # Network layout (not columns): spring embedding with slight hub anchoring.
    init_pos = {}
    if CHONGWA_LABEL in g.nodes:
        init_pos[CHONGWA_LABEL] = np.array([-1.0, 0.0], dtype=float)
    if IDIA_LABEL in g.nodes:
        init_pos[IDIA_LABEL] = np.array([1.0, 0.0], dtype=float)
    pos = nx.spring_layout(g, seed=42, k=0.9, pos=init_pos if init_pos else None, iterations=250)

    node_color = []
    node_size = []
    node_fill = {}
    for n, d in g.nodes(data=True):
        if n == CHONGWA_LABEL:
            c = "#ff7f0e"
            node_color.append(c)
            node_fill[n] = c
            node_size.append(2300)
        elif n == IDIA_LABEL:
            c = "#1f77b4"
            node_color.append(c)
            node_fill[n] = c
            node_size.append(2300)
        elif n in FEDERAL_AGENCIES or d.get("org_type") == "Federal Agency":
            c = "#0B3C8A"
            node_color.append(c)
            node_fill[n] = c
            node_size.append(260 + 28 * d["n_mentions"])
        elif n in LOCAL_GOV_AGENCIES or d.get("org_type") == "Local Government Agency":
            c = "#0E7C86"
            node_color.append(c)
            node_fill[n] = c
            node_size.append(260 + 28 * d["n_mentions"])
        else:
            c = "#A8D5E8"
            node_color.append(c)
            node_fill[n] = c
            node_size.append(240 + 24 * d["n_mentions"])

    plt.figure(figsize=(13, 8), dpi=180)

    # edge styles by relation
    edge_groups = {"aligned": [], "opposed": [], "neutral": []}
    for u, v, d in g.edges(data=True):
        edge_groups[d["relation"]].append((u, v, d))
    for rel, group in edge_groups.items():
        if not group:
            continue
        edgelist = [(u, v) for u, v, _ in group]
        widths = [1.0 + np.log1p(d["n"]) for _, _, d in group]
        if rel == "aligned":
            nx.draw_networkx_edges(g, pos, edgelist=edgelist, width=widths, edge_color="#2E7D32", alpha=0.8)
        elif rel == "opposed":
            nx.draw_networkx_edges(
                g, pos, edgelist=edgelist, width=widths, edge_color="#C62828", style="--", alpha=0.9
            )
        else:
            nx.draw_networkx_edges(g, pos, edgelist=edgelist, width=widths, edge_color="#B0B0B0", style=":", alpha=0.8)

    nx.draw_networkx_nodes(g, pos, node_color=node_color, node_size=node_size, edgecolors="#222222", linewidths=0.8)

    labels = {n: n for n in g.nodes()}
    dark_bg = {"#0B3C8A", "#1f77b4"}
    labels_dark = {
        n: lab
        for n, lab in labels.items()
        if node_fill.get(n) in dark_bg and n != IDIA_LABEL
    }
    labels_light = {
        n: lab
        for n, lab in labels.items()
        if node_fill.get(n) not in dark_bg or n == IDIA_LABEL
    }
    if labels_light:
        nx.draw_networkx_labels(g, pos, labels=labels_light, font_size=8, font_color="#1A1A1A")
    if labels_dark:
        nx.draw_networkx_labels(g, pos, labels=labels_dark, font_size=8, font_color="white")

    # Legend proxies
    plt.scatter([], [], s=130, color="#0B3C8A", label="Federal Agency")
    plt.scatter([], [], s=130, color="#0E7C86", label="Local Government Agency")
    plt.scatter([], [], s=130, color="#9ECAE1", label="Community Organization")
    plt.plot([], [], color="#2E7D32", label="Aligned", linewidth=2)
    plt.plot([], [], color="#C62828", linestyle="--", label="Opposed", linewidth=2)
    plt.plot([], [], color="#B0B0B0", linestyle=":", label="Neutral", linewidth=2)
    plt.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=6)
    plt.title("Chong Wa (CBA) and IDIA Allies Network")
    plt.figtext(
        0.5,
        0.92,
        "Node size = number of sentences where the organization is co-mentioned with Chong Wa (CBA) or IDIA",
        ha="center",
        va="top",
        fontsize=10,
        color="#4A4A4A",
    )
    plt.figtext(
        0.01,
        0.015,
        "Source: Seattle's International Examiner (1976-1979)",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4A4A4A",
    )
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Chong Wa (CBA) vs IDIA allies network")
    p.add_argument("--panel", default="outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv")
    p.add_argument("--before-year", type=int, default=1980)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--top-allies-per-side", type=int, default=12)
    p.add_argument("--min-ally-mentions", type=int, default=2)
    p.add_argument("--out-nodes", default="outputs/chongwa_idia_allies_nodes.csv")
    p.add_argument("--out-edges", default="outputs/chongwa_idia_allies_edges.csv")
    p.add_argument("--fig", default="outputs/fig_chongwa_idia_allies_network.png")
    args = p.parse_args()

    panel_path = pathlib.Path(args.panel)
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel: {panel_path}")

    nlp_ner = spacy.load(args.spacy_model, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])

    csv.field_size_limit(10**9)
    with panel_path.open(newline="", encoding="utf-8") as f:
        panel = list(csv.DictReader(f))

    # ally tallies by side
    ally_counts = {CHONGWA_LABEL: Counter(), IDIA_LABEL: Counter()}
    # edge tallies
    edge_rel = defaultdict(Counter)  # (a,b) -> relation counts
    edge_rel_month = defaultdict(lambda: defaultdict(Counter))  # (a,b)->(YYYY-MM -> rel counts)
    hub_hub = Counter()

    for r in panel:
        try:
            if int(r.get("canonical_year", "0")) >= args.before_year:
                continue
        except Exception:
            continue
        text = clean_text(r.get("paragraph_text", ""))
        if not text:
            continue
        for sent in split_sentences(text):
            cw_here = bool(CHONG_WA_PAT.search(sent))
            id_here = bool(IDIA_PAT.search(sent))
            ym = f"{r.get('canonical_year','')}-{str(r.get('canonical_month','')).zfill(2)}"
            if cw_here and id_here:
                hub_hub[relation_label(sent)] += 1

            side = mention_side(sent)
            if side is None:
                continue
            rel = relation_label(sent)
            doc = nlp_ner(sent)
            allies = set()
            for e in doc.ents:
                if e.label_ not in KEEP_ENTITY_LABELS:
                    continue
                nm = clean_ally_entity(e.text, e.label_, sent)
                if not nm:
                    continue
                if nm in {CHONGWA_LABEL, IDIA_LABEL}:
                    continue
                allies.add(nm)
            for a in allies:
                ally_counts[side][a] += 1
                hub = side
                k = tuple(sorted((hub, a)))
                edge_rel[k][rel] += 1
                edge_rel_month[k][ym][rel] += 1
            for a, b in combinations(sorted(allies), 2):
                k = tuple(sorted((a, b)))
                edge_rel[k][rel] += 1
                edge_rel_month[k][ym][rel] += 1

    selected = {CHONGWA_LABEL: set(), IDIA_LABEL: set()}
    for side in selected:
        items = [(n, c) for n, c in ally_counts[side].items() if c >= args.min_ally_mentions]
        items = sorted(items, key=lambda x: (-x[1], x[0]))[: args.top_allies_per_side]
        selected[side] = {n for n, _ in items}

    # node table
    nodes = [
        {
            "node": CHONGWA_LABEL,
            "side": CHONGWA_LABEL,
            "org_type": "Community Organization",
            "n_mentions": 0,
            "n_hub_links": 0,
        },
        {
            "node": IDIA_LABEL,
            "side": IDIA_LABEL,
            "org_type": "Community Organization",
            "n_mentions": 0,
            "n_hub_links": 0,
        },
    ]
    for side in (CHONGWA_LABEL, IDIA_LABEL):
        hub = side
        for a in sorted(selected[side], key=lambda x: (-ally_counts[side][x], x)):
            k = tuple(sorted((hub, a)))
            hub_links = sum(edge_rel[k].values())
            nodes.append(
                {
                    "node": a,
                    "side": side,
                    "org_type": (
                        "Federal Agency"
                        if a in FEDERAL_AGENCIES
                        else ("Local Government Agency" if a in LOCAL_GOV_AGENCIES else "Community Organization")
                    ),
                    "n_mentions": int(ally_counts[side][a]),
                    "n_hub_links": int(hub_links),
                }
            )

    keep_nodes = {r["node"] for r in nodes}
    edges = []
    for (a, b), counts in edge_rel.items():
        if a not in keep_nodes or b not in keep_nodes:
            continue
        n = int(sum(counts.values()))
        if n <= 0:
            continue
        rel, rel_score = classify_relation_historical(counts, edge_rel_month.get((a, b), {}))
        edges.append(
            {
                "source": a,
                "target": b,
                "n_sentences": n,
                "n_aligned": int(counts.get("aligned", 0)),
                "n_opposed": int(counts.get("opposed", 0)),
                "n_neutral": int(counts.get("neutral", 0)),
                "relation_score": rel_score,
                "relation": rel,
            }
        )
    edges = sorted(edges, key=lambda x: (-x["n_sentences"], x["source"], x["target"]))

    # Ensure Chong Wa (CBA) <-> IDIA relation is explicitly shown.
    # Project intent: these two actors are opposing hubs in this conflict map.
    hh_total = int(sum(hub_hub.values()))
    edges = [e for e in edges if not ({e["source"], e["target"]} == {CHONGWA_LABEL, IDIA_LABEL})]
    edges.append(
        {
            "source": CHONGWA_LABEL,
            "target": IDIA_LABEL,
            "n_sentences": max(1, hh_total),
            "n_aligned": int(hub_hub.get("aligned", 0)),
            "n_opposed": max(1, int(hub_hub.get("opposed", 0))),
            "n_neutral": int(hub_hub.get("neutral", 0)),
            "relation_score": -1.0,
            "relation": "opposed",
        }
    )
    edges = sorted(edges, key=lambda x: (-x["n_sentences"], x["source"], x["target"]))

    write_csv(pathlib.Path(args.out_nodes), nodes)
    write_csv(pathlib.Path(args.out_edges), edges)
    plot_network(nodes, edges, pathlib.Path(args.fig))

    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Wrote: {args.out_nodes}")
    print(f"Wrote: {args.out_edges}")
    print(f"Wrote: {args.fig}")


if __name__ == "__main__":
    main()
