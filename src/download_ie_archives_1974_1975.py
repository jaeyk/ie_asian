#!/usr/bin/env python3
"""Discover and download IE archive files for 1974 (Vol 1) to 1975 (Vol 2).

Primary source page:
  https://iexaminer.org/archives/

Behavior:
1) Parse archive page links whose nearby context references:
   - 1974 + Volume 1
   - 1975 + Volume 2
2) Download direct file links with curl.
3) Download Google Drive links with gdown (if installed), including folders.
4) Write a manifest CSV with status for every discovered link.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.parse
from html import unescape


ANCHOR_RE = re.compile(
    r"<a\s+[^>]*href=[\"'](?P<href>[^\"']+)[\"'][^>]*>(?P<label>.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
YEAR_VOL_RE = re.compile(r"(1974|1975)\s*[-–]?\s*[,;:]?\s*volume\s*(1|2)", re.IGNORECASE)
LABEL_YEAR_VOL_RE = re.compile(r"^\s*(\d{4})\s*[-–]\s*volume\s*(\d+)\s*$", re.IGNORECASE)
DIRECT_FILE_RE = re.compile(r"\.(pdf|jpg|jpeg|png|tif|tiff|zip)$", re.IGNORECASE)
SAFE_CHAR_RE = re.compile(r"[^A-Za-z0-9._-]+")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def fetch_html(url: str) -> str:
    # Curl is used intentionally because it is robust in this workspace setup.
    cp = run(["curl", "-fsSL", url])
    if cp.returncode != 0:
        msg = cp.stderr.strip() or cp.stdout.strip() or f"curl exit {cp.returncode}"
        raise RuntimeError(f"Failed to fetch {url}: {msg}")
    return cp.stdout


def clean_text(s: str) -> str:
    txt = TAG_RE.sub(" ", s)
    txt = unescape(txt)
    return WS_RE.sub(" ", txt).strip()


def guess_year_volume(context: str) -> tuple[str, str]:
    m = YEAR_VOL_RE.search(context)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def parse_year_volume_from_label(label: str) -> tuple[str, str]:
    m = LABEL_YEAR_VOL_RE.match(label.strip())
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def find_candidate_links(html: str) -> list[dict]:
    out: list[dict] = []
    for m in ANCHOR_RE.finditer(html):
        href = unescape(m.group("href")).strip()
        label = clean_text(m.group("label"))
        if not href:
            continue
        if href.startswith("#"):
            continue

        # Inspect nearby context (archive page usually includes year/volume text nearby).
        start, end = m.span()
        nearby = clean_text(html[max(0, start - 260): min(len(html), end + 320)])
        # Prefer exact label parsing (e.g., "1975 – Volume 2"), then fallback to nearby context.
        year, volume = parse_year_volume_from_label(label)
        if not year:
            year, volume = guess_year_volume(nearby)
        if not year:
            year, volume = guess_year_volume(label)

        if not year:
            continue
        if not ((year == "1974" and volume == "1") or (year == "1975" and volume == "2")):
            continue

        out.append(
            {
                "href": href,
                "label": label,
                "year": year,
                "volume": volume,
                "context": nearby,
            }
        )

    # Dedupe by href, keep first.
    dedup: list[dict] = []
    seen = set()
    for row in out:
        if row["href"] in seen:
            continue
        seen.add(row["href"])
        dedup.append(row)
    return dedup


def classify_link(href: str) -> str:
    low = href.lower()
    if "drive.google.com" in low and "/folders/" in low:
        return "google_drive_folder"
    if "drive.google.com" in low:
        return "google_drive_file"
    path = urllib.parse.urlparse(href).path
    if DIRECT_FILE_RE.search(path):
        return "direct_file"
    return "web_link"


def safe_name(s: str, fallback: str) -> str:
    base = SAFE_CHAR_RE.sub("_", s.strip()).strip("._-")
    return base or fallback


def download_direct_file(url: str, out_path: pathlib.Path) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cp = run(["curl", "-fL", "--retry", "3", "-o", str(out_path), url])
    if cp.returncode != 0:
        return False, (cp.stderr.strip() or cp.stdout.strip() or f"curl exit {cp.returncode}")
    return True, ""


def extract_drive_file_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    m = re.search(r"/d/([A-Za-z0-9_-]+)", parsed.path)
    if m:
        return m.group(1)
    return ""


def gdown_available() -> bool:
    return shutil.which("gdown") is not None


def download_google_drive(url: str, out_dir: pathlib.Path, link_type: str) -> tuple[bool, str, str]:
    # Folder download needs gdown.
    if link_type == "google_drive_folder" and not gdown_available():
        return False, "", "gdown not found (install via `pip install gdown`)"

    # File links: if gdown is unavailable, try direct uc?id fallback.
    if link_type == "google_drive_file" and not gdown_available():
        fid = extract_drive_file_id(url)
        if not fid:
            return False, "", "could not parse Google Drive file id"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fid}.bin"
        dl_url = f"https://drive.google.com/uc?export=download&id={fid}"
        ok, err = download_direct_file(dl_url, out_path)
        if ok:
            return True, str(out_path), ""
        return False, "", f"direct Drive download failed; install gdown. {err}"

    out_dir.mkdir(parents=True, exist_ok=True)
    if link_type == "google_drive_folder":
        cp = run(["gdown", "--folder", "--fuzzy", url, "-O", str(out_dir)])
        if cp.returncode != 0:
            return False, "", cp.stderr.strip() or cp.stdout.strip() or f"gdown exit {cp.returncode}"
        return True, str(out_dir), ""
    file_name = safe_name(pathlib.Path(urllib.parse.urlparse(url).path).name or "drive_file", "drive_file")
    out_path = out_dir / file_name
    cp = run(["gdown", "--fuzzy", url, "-O", str(out_path)])
    if cp.returncode != 0:
        # Some legacy open?id links on IE archives are actually folders.
        # Try folder URL fallback if file download fails.
        fid = extract_drive_file_id(url)
        if fid:
            folder_url = f"https://drive.google.com/drive/folders/{fid}"
            cp2 = run(["gdown", "--folder", "--fuzzy", folder_url, "-O", str(out_dir)])
            if cp2.returncode == 0:
                return True, str(out_dir), ""
        return False, "", cp.stderr.strip() or cp.stdout.strip() or f"gdown exit {cp.returncode}"
    return True, str(out_path), ""


def write_manifest(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "year",
        "volume",
        "label",
        "href",
        "link_type",
        "status",
        "local_path",
        "error",
        "context",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Download IE archives for 1974 Vol 1 and 1975 Vol 2")
    p.add_argument("--archives-url", default="https://iexaminer.org/archives/")
    p.add_argument("--out-dir", default="raw_data/ie_archives_1974_1975")
    p.add_argument("--manifest", default="outputs/ie_archives_1974_1975_manifest.csv")
    p.add_argument("--discover-only", action="store_true", help="Only discover and write manifest, do not download")
    args = p.parse_args()

    try:
        html = fetch_html(args.archives_url)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    links = find_candidate_links(html)
    if not links:
        print("No candidate links found for 1974 Vol 1 / 1975 Vol 2.")
        sys.exit(1)

    out_dir = pathlib.Path(args.out_dir)
    manifest_rows: list[dict] = []

    for i, row in enumerate(links, start=1):
        href = row["href"]
        link_type = classify_link(href)
        status = "discovered"
        local_path = ""
        err = ""

        year_dir = out_dir / f"{row['year']}_volume_{row['volume']}"
        label_part = safe_name(row["label"], f"item_{i:03d}")

        if not args.discover_only:
            if link_type == "direct_file":
                parsed = urllib.parse.urlparse(href)
                file_name = pathlib.Path(parsed.path).name or f"{label_part}.bin"
                ok, err = download_direct_file(href, year_dir / file_name)
                status = "downloaded" if ok else "failed"
                if ok:
                    local_path = str(year_dir / file_name)
            elif link_type in {"google_drive_folder", "google_drive_file"}:
                ok, lp, err = download_google_drive(href, year_dir / label_part, link_type)
                status = "downloaded" if ok else "skipped_or_failed"
                local_path = lp
            else:
                # Try as plain web URL fallback.
                file_name = f"{label_part}.html"
                ok, err = download_direct_file(href, year_dir / file_name)
                status = "downloaded" if ok else "failed"
                if ok:
                    local_path = str(year_dir / file_name)

        manifest_rows.append(
            {
                "year": row["year"],
                "volume": row["volume"],
                "label": row["label"],
                "href": href,
                "link_type": link_type,
                "status": status,
                "local_path": local_path,
                "error": err,
                "context": row["context"],
            }
        )

    write_manifest(pathlib.Path(args.manifest), manifest_rows)
    downloaded = sum(1 for r in manifest_rows if r["status"] == "downloaded")
    print(f"Candidates: {len(manifest_rows)}")
    print(f"Downloaded: {downloaded}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
