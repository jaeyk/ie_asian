#!/usr/bin/env python3
"""Enhanced v2 full-text analysis focused on direction CI output."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import string
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import spacy


SENT_SPLIT_PAT = re.compile(r"(?<=[.!?])\s+")
WS_PAT = re.compile(r"\s+")

CHONG_WA_PAT = re.compile(r"\b(chong\s*wa(?:h)?|chinese\s+benevolent\s+association|cba)\b", re.IGNORECASE)
IDIA_PAT = re.compile(
    r"\b((international\W+district\W+)?improvement\W+association|id\W+improvement\W+association|inter\*?im|"
    r"(international\W+district\W+)?housing\W+alliance|idha)\b",
    re.IGNORECASE,
)

QUOTE_PAT = re.compile(r"\"([^\"]+)\"|“([^”]+)”")
SPEECH_VERB_PAT = re.compile(
    r"\b(said|says|stated|told|reported|asked|replied|noted|argued|claimed|announced|warned)\b",
    re.IGNORECASE,
)

SUPPORT_LEMMAS = {
    "support",
    "approve",
    "back",
    "endorse",
    "cooperate",
    "collaborate",
    "promote",
    "help",
    "assist",
    "fund",
    "build",
}
OPPOSE_LEMMAS = {
    "oppose",
    "object",
    "reject",
    "criticize",
    "dispute",
    "challenge",
    "warn",
    "protest",
    "block",
    "appeal",
    "spearhead",
    "fight",
}

FRAME_SUB_ORDER = [
    "Resident: Housing Affordability",
    "Resident: Anti-Eviction/Displacement",
    "State Resource Access / Service Delivery",
    "Commercial Growth",
]
ACTOR_CORE_ORDER = ["Chong Wa", "ID Improvement Association"]

HOUSING_AFF_PAT = re.compile(r"\b(low[\s-]*income housing|affordable housing|housing project|housing units?)\b", re.IGNORECASE)
DISPLACE_PAT = re.compile(r"\b(evict|eviction|displac|tenant|rent|relocat)\b", re.IGNORECASE)
ENV_PAT = re.compile(
    r"\b(traffic|congestion|pollution|air quality|land speculation|union station|terminal|stadium traffic)\b",
    re.IGNORECASE,
)
STATE_PAT = re.compile(
    r"\b(model cities|office of economic opportunity|oeo|hud|grant|fund|funding|application|proposal|program|service|clinic|agency|board)\b",
    re.IGNORECASE,
)
REP_PAT = re.compile(r"\b(chinatown|international district|represent|majority|community leadership|identity)\b", re.IGNORECASE)
COMM_PAT = re.compile(r"\b(commercial|business|economic growth|development|parking lot|domed stadium|stadium jobs?)\b", re.IGNORECASE)
IMPLICIT_EVICT_OPPOSE_PAT = re.compile(
    r"\b(pending eviction|face eviction|forced out|displaced residents|threaten(?:s|ed)? .* housing|"
    r"without heat|unsafe wiring|have to search for new housing|hotel closures?|evict(?:ion|ed)? tenants?)\b",
    re.IGNORECASE,
)
IDIA_CONTEXT_PAT = re.compile(
    r"\b(inter\*?im|id improvement association|idia|idha|housing alliance|international district housing alliance|bob santos|santos wrote)\b",
    re.IGNORECASE,
)
COREF_PAT = re.compile(r"\b(they|them|their|it|its|the group|the association|this organization)\b", re.IGNORECASE)
GENERIC_TARGET_PAT = re.compile(r"^(us|it|this|that|them|they|unspecified target|the project|this plan)$", re.IGNORECASE)
TOKEN_PAT = re.compile(r"[a-z][a-z\-]{2,}")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "have", "has", "had", "were", "was",
    "are", "but", "not", "their", "they", "them", "into", "onto", "also", "will", "would", "could",
    "about", "there", "which", "when", "what", "where", "while", "after", "before", "through",
    "district", "international", "chinatown", "interim", "inter", "im", "association", "group",
}


def clean_text(s: str) -> str:
    return WS_PAT.sub(" ", (s or "").replace("&nbsp;", " ").replace("&amp;", "&")).strip()


def split_sentences(text: str) -> list[str]:
    return [clean_text(x) for x in SENT_SPLIT_PAT.split(text) if clean_text(x)]


def actor_mentions(sentence: str) -> tuple[bool, bool]:
    return bool(CHONG_WA_PAT.search(sentence)), bool(IDIA_PAT.search(sentence))


def normalize_actor_name(name: str) -> str:
    s = name.lower()
    if CHONG_WA_PAT.search(s):
        return "Chong Wa"
    if IDIA_PAT.search(s):
        return "ID Improvement Association"
    return "Other"


def infer_speaker(sentence: str) -> str:
    # Strict attribution: actor mention must occur near a speech verb.
    s = sentence
    if not SPEECH_VERB_PAT.search(s):
        return "Unknown"

    tokens = s.split()
    lower_tokens = [t.lower() for t in tokens]
    speech_ix = [i for i, t in enumerate(lower_tokens) if SPEECH_VERB_PAT.search(t)]
    if not speech_ix:
        return "Unknown"

    cw_hits = 0
    id_hits = 0
    # +/- 7 token window around each speech verb
    for i in speech_ix:
        lo = max(0, i - 7)
        hi = min(len(tokens), i + 8)
        window = " ".join(tokens[lo:hi])
        if CHONG_WA_PAT.search(window):
            cw_hits += 1
        if IDIA_PAT.search(window):
            id_hits += 1

    if cw_hits > 0 and id_hits == 0:
        return "Chong Wa"
    if id_hits > 0 and cw_hits == 0:
        return "ID Improvement Association"
    if cw_hits > 0 and id_hits > 0:
        return "Both/Unclear"
    return "Unknown"


def quote_scope(sentence: str) -> str:
    quoted = []
    for m in QUOTE_PAT.finditer(sentence):
        chunk = m.group(1) or m.group(2) or ""
        chunk = clean_text(chunk)
        if chunk:
            quoted.append(chunk)
    if quoted:
        return " ".join(quoted)
    return sentence


def frame_sub(sentence: str) -> tuple[str, bool]:
    s = sentence
    if HOUSING_AFF_PAT.search(s):
        return "Resident: Housing Affordability", True
    if DISPLACE_PAT.search(s):
        return "Resident: Anti-Eviction/Displacement", True
    if ENV_PAT.search(s):
        return "Resident: Environmental/Traffic Impacts", True
    if STATE_PAT.search(s):
        return "State Resource Access / Service Delivery", True
    if REP_PAT.search(s):
        return "Representation / Identity", True
    if COMM_PAT.search(s):
        return "Commercial Growth", True
    return "Other", False


def frame_sub_with_lexicon(sentence: str, lexicon: dict[str, set[str]]) -> tuple[str, bool]:
    frame, explicit = frame_sub(sentence)
    if frame != "Other":
        return frame, explicit
    toks = set(TOKEN_PAT.findall(sentence.lower()))
    best = ("Other", 0)
    for k, terms in lexicon.items():
        hit = len(toks & terms)
        if hit > best[1]:
            best = (k, hit)
    if best[1] >= 2:
        return best[0], False
    return "Other", False


def build_seeded_lexicon(panel_rows: list[dict], before_year: int) -> dict[str, set[str]]:
    buckets = {k: Counter() for k in FRAME_SUB_ORDER}
    for r in panel_rows:
        try:
            if int(r.get("canonical_year", "0")) >= before_year:
                continue
        except Exception:
            continue
        text = clean_text(r.get("paragraph_text", ""))
        if not text:
            continue
        for sent in split_sentences(text):
            frame, explicit = frame_sub(sent)
            if frame == "Other":
                continue
            if frame not in buckets:
                continue
            toks = [t for t in TOKEN_PAT.findall(sent.lower()) if t not in STOPWORDS]
            buckets[frame].update(toks)
    lexicon: dict[str, set[str]] = {}
    for frame, c in buckets.items():
        top_terms = [t for t, n in c.most_common(40) if n >= 2]
        lexicon[frame] = set(top_terms)
    return lexicon


def token_phrase(token) -> str:
    subtree = list(token.subtree)
    text = " ".join(t.text for t in subtree).strip()
    return clean_text(text)[:120]


def extract_stance_events(nlp, text: str) -> list[dict]:
    doc = nlp(text)
    events = []
    for tok in doc:
        if tok.pos_ not in {"VERB", "AUX"}:
            continue
        lemma = tok.lemma_.lower().strip()
        if lemma in SUPPORT_LEMMAS:
            stance = "Support"
        elif lemma in OPPOSE_LEMMAS:
            stance = "Oppose"
        else:
            continue
        neg = any(c.dep_ == "neg" for c in tok.children)
        if neg:
            stance = "Oppose" if stance == "Support" else "Support"

        target = ""
        for c in tok.children:
            if c.dep_ in {"dobj", "obj", "pobj", "attr", "xcomp", "ccomp"}:
                target = token_phrase(c)
                break
        if not target:
            rights = [r for r in tok.rights if r.pos_ in {"NOUN", "PROPN", "PRON"}]
            if rights:
                target = token_phrase(rights[0])
        target = target or "unspecified target"
        events.append(
            {
                "verb": tok.lemma_.lower(),
                "stance": stance,
                "target": target,
                "negated": int(neg),
            }
        )
    return events


def extract_implicit_events(text: str, frame: str) -> list[dict]:
    s = text.lower()
    events = []
    if frame == "Resident: Anti-Eviction/Displacement" and IMPLICIT_EVICT_OPPOSE_PAT.search(s):
        tgt = "eviction/displacement threat"
        if "hotel" in s:
            tgt = "hotel tenant displacement"
        events.append({"verb": "implicit", "stance": "Oppose", "target": tgt, "negated": 0})
    return events


def override_frame_from_target(frame: str, target: str) -> str:
    t = (target or "").lower()
    if re.search(r"\b(evict\w*|displac\w*|tenant\w*|forced out|hotel closures?)\b", t):
        return "Resident: Anti-Eviction/Displacement"
    return frame


def confidence_score(
    explicit_actor: bool,
    speaker_actor_match: bool,
    has_event: bool,
    has_target: bool,
    in_quote: bool,
    frame_explicit: bool,
) -> float:
    score = 0.1
    if explicit_actor:
        score += 0.25
    if speaker_actor_match:
        score += 0.2
    if has_event:
        score += 0.2
    if has_target:
        score += 0.15
    if in_quote:
        score += 0.1
    if frame_explicit:
        score += 0.1
    return round(min(1.0, score), 4)


def attribution_tier(sentence: str, speaker: str) -> str:
    has_quote = bool(QUOTE_PAT.search(sentence))
    has_speech = bool(SPEECH_VERB_PAT.search(sentence))
    if has_quote and has_speech and speaker in ACTOR_CORE_ORDER:
        return "direct_quote"
    if has_speech:
        return "reported_speech"
    return "narrative_assertion"


def canonicalize_target(target: str, article_state: dict[str, str]) -> str:
    t = clean_text((target or "").lower())
    if not t:
        return "unspecified target"
    if GENERIC_TARGET_PAT.search(t):
        return article_state.get("last_target", "unspecified target")
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = WS_PAT.sub(" ", t).strip()
    if t and not GENERIC_TARGET_PAT.search(t):
        article_state["last_target"] = t
    return t or "unspecified target"


def split_stance_axis(stance: str, frame_sub_name: str) -> tuple[str, str]:
    # Two-axis stance: policy vs process
    policy_frames = {
        "Resident: Housing Affordability",
        "Resident: Anti-Eviction/Displacement",
        "Resident: Environmental/Traffic Impacts",
        "Commercial Growth",
    }
    if frame_sub_name in policy_frames:
        return stance, "Neutral"
    return "Neutral", stance


def plot_attribution_tiers(rows: list[dict], out_path: pathlib.Path) -> None:
    tiers = ["direct_quote", "reported_speech", "narrative_assertion"]
    actors = ACTOR_CORE_ORDER
    data = {a: [0] * len(tiers) for a in actors}
    for r in rows:
        a = r["actor"]
        t = r["attribution_tier"]
        if a in actors and t in tiers:
            data[a][tiers.index(t)] += int(r["n_unique_claims"])
    x = np.arange(len(tiers))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    ax.bar(x - w / 2, data["Chong Wa"], width=w, color="#1f77b4", label="Chong Wa")
    ax.bar(x + w / 2, data["ID Improvement Association"], width=w, color="#ff7f0e", label="ID Improvement Association")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Unique Claims")
    ax.set_title("Attribution Tier Mix")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def bootstrap_direction_ci(items: list[dict], n_boot: int = 600) -> tuple[float, float]:
    if not items:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    vals = []
    n = len(items)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [items[i] for i in idx]
        confs = np.array([float(x["confidence"]) for x in sample], dtype=float)
        stances = [x["stance"] for x in sample]
        sup = np.array([1.0 if s == "Support" else 0.0 for s in stances], dtype=float)
        opp = np.array([1.0 if s == "Oppose" else 0.0 for s in stances], dtype=float)
        direction = float(np.sum((sup - opp) * confs) / max(1.0, np.sum(confs)))
        vals.append(direction)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def plot_direction_ci(values: list[dict], out_path: pathlib.Path, actor_order: list[str]) -> None:
    ybase = np.arange(len(FRAME_SUB_ORDER))
    if len(actor_order) == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.24, 0.24, num=len(actor_order))
    offset = {a: float(offsets[i]) for i, a in enumerate(actor_order)}
    color = {
        "Chong Wa": "#1f77b4",
        "ID Improvement Association": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=180)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    for actor in actor_order:
        rows = [r for r in values if r["actor"] == actor]
        if not rows:
            continue
        ys = [ybase[FRAME_SUB_ORDER.index(r["frame_sub"])] + offset[actor] for r in rows]
        x = [float(r["direction_index_weighted"]) for r in rows]
        lo = [float(r["ci_low"]) for r in rows]
        hi = [float(r["ci_high"]) for r in rows]
        xerr = [np.array(x) - np.array(lo), np.array(hi) - np.array(x)]
        ax.errorbar(
            x,
            ys,
            xerr=xerr,
            fmt="o",
            color=color.get(actor, "#333333"),
            capsize=3,
            markersize=5,
            label=actor,
        )
    ax.set_yticks(ybase)
    ax.set_yticklabels(FRAME_SUB_ORDER)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Direction Index (Support minus Oppose), confidence-weighted")
    ax.set_title("Direction with 95% Bootstrap CI")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=max(2, min(4, len(actor_order))),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: pathlib.Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Enhanced v2 full-text Chong Wa vs IDIA analysis")
    p.add_argument("--panel", default="outputs/ie_chongwa_id_pre1980_paragraph_panel_filtered.csv")
    p.add_argument("--before-year", type=int, default=1980)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--out-claims", default="outputs/enhanced_v2_chongwa_idia_claims.csv")
    p.add_argument("--out-unique-claims", default="outputs/enhanced_v2_chongwa_idia_unique_claims.csv")
    p.add_argument("--out-values", default="outputs/enhanced_v2_chongwa_idia_heatmap_values.csv")
    p.add_argument("--fig-ci", default="outputs/fig_enhanced_v2_chongwa_idia_direction_ci.png")
    args = p.parse_args()

    panel_path = pathlib.Path(args.panel)
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel: {panel_path}")

    nlp = spacy.load(args.spacy_model, disable=["ner"])

    csv.field_size_limit(10**9)
    with panel_path.open(newline="", encoding="utf-8") as f:
        panel = list(csv.DictReader(f))

    lexicon = build_seeded_lexicon(panel, args.before_year)
    actor_order = ACTOR_CORE_ORDER
    claim_rows = []
    for r in panel:
        try:
            year = int(r.get("canonical_year", "0"))
        except Exception:
            continue
        if year >= args.before_year:
            continue

        raw = clean_text(r.get("paragraph_text", ""))
        if not raw:
            continue
        file_name = r.get("file_name", "")
        article_id = r.get("article_id", "")
        paragraph_id = r.get("paragraph_id", "")
        ym = f"{r.get('canonical_year','')}-{str(r.get('canonical_month','')).zfill(2)}"
        paragraph_state = {"last_actor": "", "last_target": "unspecified target"}
        for sent in split_sentences(raw):
            cw, idia = actor_mentions(sent)
            if not (cw or idia):
                # Context fallback: assign IDIA for implicit anti-eviction statements in IDIA-coded paragraphs.
                if IMPLICIT_EVICT_OPPOSE_PAT.search(sent) and IDIA_CONTEXT_PAT.search(raw):
                    cw = False
                    idia = True
                elif paragraph_state["last_actor"] and COREF_PAT.search(sent):
                    cw = paragraph_state["last_actor"] == "Chong Wa"
                    idia = paragraph_state["last_actor"] == "ID Improvement Association"
                else:
                    continue
            actor_list = []
            if cw:
                actor_list.append("Chong Wa")
            if idia:
                actor_list.append("ID Improvement Association")
            if len(actor_list) == 1:
                paragraph_state["last_actor"] = actor_list[0]
            speaker = infer_speaker(sent)
            scoped = quote_scope(sent)
            frame, frame_explicit = frame_sub_with_lexicon(sent, lexicon)
            if frame == "Other":
                continue
            events = extract_stance_events(nlp, scoped)
            if not events:
                events = extract_implicit_events(scoped, frame)
            if not events:
                events = [{"verb": "", "stance": "Neutral", "target": "unspecified target", "negated": 0}]
            in_quote = scoped != sent

            for actor in actor_list:
                explicit_actor = True
                speaker_actor_match = speaker == actor
                tier = attribution_tier(sent, speaker)
                for ev in events:
                    target_norm = canonicalize_target(ev.get("target", ""), paragraph_state)
                    frame_eff = override_frame_from_target(frame, target_norm)
                    has_target = target_norm != "unspecified target"
                    context_fallback = not actor_mentions(sent)[0] and not actor_mentions(sent)[1]
                    conf = confidence_score(
                        explicit_actor=explicit_actor,
                        speaker_actor_match=speaker_actor_match,
                        has_event=ev["verb"] != "",
                        has_target=has_target,
                        in_quote=in_quote,
                        frame_explicit=frame_explicit,
                    )
                    if context_fallback:
                        conf = round(max(0.35, conf - 0.2), 4)
                    stance_policy, stance_process = split_stance_axis(ev["stance"], frame_eff)
                    claim_key = "|".join(
                        [
                            file_name,
                            article_id,
                            actor,
                            frame_eff,
                            stance_policy + "/" + stance_process,
                            " ".join(target_norm.split()[:6]),
                        ]
                    )
                    claim_rows.append(
                        {
                            "file_name": file_name,
                            "article_id": article_id,
                            "paragraph_id": paragraph_id,
                            "year_month": ym,
                            "actor": actor,
                            "speaker_actor": speaker,
                            "attribution_tier": tier,
                            "frame_sub": frame_eff,
                            "stance": ev["stance"],
                            "stance_policy": stance_policy,
                            "stance_process": stance_process,
                            "verb": ev["verb"],
                            "target": target_norm or "unspecified target",
                            "negated": ev["negated"],
                            "in_quote_scope": int(in_quote),
                            "confidence": conf,
                            "claim_key": claim_key,
                            "sentence": sent,
                        }
                    )

    write_csv(pathlib.Path(args.out_claims), claim_rows)

    # Article-level claim clustering (unique by claim key).
    unique = {}
    for r in claim_rows:
        k = r["claim_key"]
        if k not in unique:
            unique[k] = dict(r)
            unique[k]["n_mentions"] = 1
        else:
            unique[k]["n_mentions"] += 1
            unique[k]["confidence"] = max(float(unique[k]["confidence"]), float(r["confidence"]))
    unique_rows = list(unique.values())
    write_csv(pathlib.Path(args.out_unique_claims), unique_rows)

    # Aggregate heatmap values.
    by_cell = defaultdict(list)
    for r in unique_rows:
        if r["actor"] not in actor_order or r["frame_sub"] not in FRAME_SUB_ORDER:
            continue
        by_cell[(r["actor"], r["frame_sub"])].append(r)

    value_rows = []
    for frame in FRAME_SUB_ORDER:
        for actor in actor_order:
            items = by_cell.get((actor, frame), [])
            n_unique = len(items)
            if n_unique == 0:
                value_rows.append(
                    {
                        "actor": actor,
                        "frame_sub": frame,
                        "n_unique_claims": 0,
                        "n_support": 0,
                        "n_oppose": 0,
                        "n_neutral": 0,
                        "n_mixed": 0,
                        "mean_confidence": 0.0,
                        "direction_index_weighted": 0.0,
                    }
                )
                continue

            confs = np.array([float(x["confidence"]) for x in items], dtype=float)
            stances = [x["stance"] for x in items]
            sup = np.array([1.0 if s == "Support" else 0.0 for s in stances], dtype=float)
            opp = np.array([1.0 if s == "Oppose" else 0.0 for s in stances], dtype=float)
            neu = int(sum(1 for s in stances if s == "Neutral"))
            mix = int(sum(1 for s in stances if s == "Mixed"))

            direction = float(np.sum((sup - opp) * confs) / max(1.0, np.sum(confs)))
            ci_low, ci_high = bootstrap_direction_ci(items)
            value_rows.append(
                {
                    "actor": actor,
                    "frame_sub": frame,
                    "n_unique_claims": n_unique,
                    "n_support": int(np.sum(sup)),
                    "n_oppose": int(np.sum(opp)),
                    "n_neutral": neu,
                    "n_mixed": mix,
                    "mean_confidence": round(float(np.mean(confs)), 4),
                    "direction_index_weighted": round(direction, 4),
                    "ci_low": round(ci_low, 4),
                    "ci_high": round(ci_high, 4),
                }
            )

    write_csv(pathlib.Path(args.out_values), value_rows)

    plot_direction_ci(value_rows, pathlib.Path(args.fig_ci), actor_order)

    print(f"Claim rows (raw): {len(claim_rows)}")
    print(f"Unique clustered claims: {len(unique_rows)}")
    print(f"Wrote: {args.out_claims}")
    print(f"Wrote: {args.out_unique_claims}")
    print(f"Wrote: {args.out_values}")
    print(f"Wrote: {args.fig_ci}")


if __name__ == "__main__":
    main()
