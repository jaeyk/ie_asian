#!/usr/bin/env python3
"""Build IE paragraph panel with Chong Wa/Benevolent Association vs International District labels.

Input:
- raw_data/ie.csv

Output:
- outputs/ie_chongwa_id_paragraph_panel.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import pathlib
import re
import sys
from datetime import datetime

from build_paragraph_panel import (  # type: ignore
    DOMAIN_PATTERNS,
    FRAME_PATTERNS,
    bool_int,
    extract_matches,
    normalize_text,
)


TAG_PATTERN = re.compile(r"<[^>]+>")
SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
MULTI_BLANK_PATTERN = re.compile(r"\n\s*\n+")
ALNUM_PATTERN = re.compile(r"[^a-z0-9]")

CHONGWA_PATTERN = re.compile(
    r"\b(chong wa(?:h)?|chong wa benevolent association|cwba|"
    r"chinese benevolent association|chinese benevolet association|benevolent association)\b",
    re.IGNORECASE,
)
INTL_DISTRICT_PHRASE_PATTERN = re.compile(
    r"\b(international district)\b",
    re.IGNORECASE,
)
INTL_DISTRICT_ID_PATTERN = re.compile(r"\bID\b")


def clean_web_text(text: str) -> str:
    t = text or ""
    t = html.unescape(t)
    t = SCRIPT_STYLE_PATTERN.sub(" ", t)
    t = TAG_PATTERN.sub(" ", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u00a0", " ")
    t = MULTI_SPACE_PATTERN.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_paragraphs(text: str) -> list[str]:
    chunks = [c.strip() for c in MULTI_BLANK_PATTERN.split(text) if c.strip()]
    out: list[str] = []
    for c in chunks:
        if len(ALNUM_PATTERN.sub("", c.lower())) < 20:
            continue
        out.append(c)
    return out


def parse_date_parts(date_raw: str, year_raw: str) -> tuple[str, str, str]:
    d = (date_raw or "").strip()
    if d:
        try:
            dt = datetime.strptime(d[:10], "%Y-%m-%d")
            return str(dt.year), str(dt.month), dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    y = (year_raw or "").strip()
    if re.fullmatch(r"\d{4}", y):
        return y, "", f"{y}-01-01"
    return "", "", ""


def keyword_type(text: str) -> str:
    has_ct = bool(CHONGWA_PATTERN.search(text))
    has_id = bool(INTL_DISTRICT_PHRASE_PATTERN.search(text))
    if has_ct and has_id:
        return "both"
    if has_ct:
        return "chongwa_benevolent_association"
    if has_id:
        return "international_district"
    return "none"


def international_district_hits(text_raw: str, text_norm: str) -> str:
    hits = set()
    for m in INTL_DISTRICT_PHRASE_PATTERN.finditer(text_norm):
        hits.add(m.group(0).lower())
    if INTL_DISTRICT_ID_PATTERN.search(text_raw):
        hits.add("ID")
    return "|".join(sorted(hits, key=lambda x: (x != "ID", x)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IE panel with Chong Wa/Benevolent Association vs International District labeling")
    parser.add_argument("--in-csv", default="raw_data/ie.csv")
    parser.add_argument("--out", default="outputs/ie_chongwa_id_paragraph_panel.csv")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N articles (0 = all)")
    parser.add_argument("--before-year", type=int, default=0, help="Keep rows with canonical_year < this year (0 = no filter)")
    parser.add_argument("--require-date", action="store_true", help="Drop rows without parsed canonical year")
    parser.add_argument("--source", default="", help="Keep rows where source exactly matches this value (empty = no filter)")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    in_path = pathlib.Path(args.in_csv)
    out_path = pathlib.Path(args.out)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_path}")

    rows_out: list[dict] = []
    n_articles = 0
    n_paragraphs = 0

    with in_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for idx, row in enumerate(r, start=1):
            if args.limit and idx > args.limit:
                break

            raw_text = row.get("text") or ""
            cleaned = clean_web_text(raw_text)
            paragraphs = split_paragraphs(cleaned)
            if not paragraphs:
                continue

            article_id = (row.get("") or str(idx)).strip() or str(idx)
            source = (row.get("source") or "").strip()
            if args.source and source != args.source:
                continue
            author = (row.get("author") or "").strip()
            date_raw = (row.get("date") or "").strip()
            year_raw = (row.get("year") or "").strip()
            c_year, c_month, c_date = parse_date_parts(date_raw, year_raw)
            year_int = int(c_year) if c_year.isdigit() else None
            if args.require_date and year_int is None:
                continue
            if args.before_year and (year_int is None or year_int >= args.before_year):
                continue
            file_name = f"ie_{article_id}"

            for p_idx, para in enumerate(paragraphs, start=1):
                txt_norm = normalize_text(para)
                ct_hits = extract_matches(CHONGWA_PATTERN, txt_norm)
                id_hits = international_district_hits(para, txt_norm)
                has_ct = bool(ct_hits)
                has_id = bool(id_hits)
                if has_ct and has_id:
                    kw_type = "both"
                elif has_ct:
                    kw_type = "chongwa_benevolent_association"
                elif has_id:
                    kw_type = "international_district"
                else:
                    kw_type = "none"
                frame_data_hits = extract_matches(FRAME_PATTERNS["policy_frame_data"], txt_norm)
                frame_fund_hits = extract_matches(FRAME_PATTERNS["policy_frame_funding"], txt_norm)

                domain_hits: dict[str, str] = {}
                for key, pattern in DOMAIN_PATTERNS.items():
                    domain_hits[key] = extract_matches(pattern, txt_norm)

                has_any_domain = any(bool(domain_hits[k]) for k in DOMAIN_PATTERNS if k.startswith("domain_"))

                row_out = {
                    "paragraph_id": f"{file_name}_n{p_idx}",
                    "file_name": file_name,
                    "article_id": article_id,
                    "source": source,
                    "author": author,
                    "article_date": date_raw,
                    "canonical_year": c_year,
                    "canonical_month": c_month,
                    "canonical_issue_date": c_date,
                    "validation_status": "web_date_parsed" if c_date else "web_date_missing",
                    "page_number": 1,
                    "column_index": 0,
                    "paragraph_in_column": p_idx,
                    "paragraph_in_page": p_idx,
                    "identity_type": kw_type,
                    # Keep downstream compatibility with existing analysis scripts.
                    "has_panethnic_label": bool_int(has_ct),
                    "has_ethnic_label": bool_int(has_id),
                    "panethnic_hits": ct_hits,
                    "ethnic_hits": id_hits,
                    # Explicit columns for this keyword analysis.
                    "has_chongwa_label": bool_int(has_ct),
                    "has_international_district_label": bool_int(has_id),
                    "chongwa_hits": ct_hits,
                    "international_district_hits": id_hits,
                    "policy_frame_data": bool_int(bool(frame_data_hits)),
                    "policy_frame_data_hits": frame_data_hits,
                    "policy_frame_funding": bool_int(bool(frame_fund_hits)),
                    "policy_frame_funding_hits": frame_fund_hits,
                }
                for key in DOMAIN_PATTERNS:
                    row_out[key] = bool_int(bool(domain_hits[key]))
                    row_out[f"{key}_hits"] = domain_hits[key]

                row_out["domain_other"] = bool_int(not has_any_domain)
                row_out["paragraph_text"] = para
                rows_out.append(row_out)
                n_paragraphs += 1

            n_articles += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        print("No paragraph rows generated.")
        return

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print(f"Input articles processed: {n_articles}")
    print(f"Paragraph rows: {n_paragraphs}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
