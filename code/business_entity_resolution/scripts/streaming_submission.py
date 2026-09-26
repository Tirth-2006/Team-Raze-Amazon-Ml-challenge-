"""Disk-backed, bounded-memory rule-based submission pipeline.

The target files are indexed once in SQLite, then Source 1 is processed in
small chunks. No full S2+S3 dataframe or all-results dictionary is retained.
"""
import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from normalization import normalize_address, normalize_business_name
from paths import DATASET_ROOT, OUTPUT_ROOT


FIELDS = ["entity_id", "business_name", "business_address", "country"]


def add_features(frame):
    frame = frame[FIELDS].copy()
    for field in FIELDS[1:]:
        frame[field] = frame[field].fillna("").astype(str)
    frame["name_norm"] = frame["business_name"].map(normalize_business_name)
    frame["addr_norm"] = frame["business_address"].map(normalize_address)
    frame["name_prefix"] = frame["name_norm"].str[:4]
    frame["addr_num"] = frame["addr_norm"].str.extract(r"(\d+)", expand=False).fillna("")
    return frame


def build_index(db_path, source_paths, chunksize):
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        """CREATE TABLE targets (
            entity_id TEXT PRIMARY KEY, business_name TEXT NOT NULL,
            business_address TEXT NOT NULL, country TEXT NOT NULL,
            name_norm TEXT NOT NULL, addr_norm TEXT NOT NULL,
            name_prefix TEXT NOT NULL, addr_num TEXT NOT NULL
        )"""
    )
    insert_sql = (
        "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for source_path in source_paths:
        print(f"Indexing {source_path.name}...")
        for chunk in pd.read_csv(source_path, sep="\t", usecols=FIELDS, chunksize=chunksize):
            prepared = add_features(chunk)
            connection.executemany(
                insert_sql,
                prepared.itertuples(index=False, name=None),
            )
            connection.commit()
    connection.execute("CREATE INDEX idx_targets_prefix ON targets(country, name_prefix)")
    connection.execute("CREATE INDEX idx_targets_num ON targets(country, addr_num)")
    connection.commit()
    connection.close()


def score_chunk(connection, chunk, max_candidates, name_threshold, address_threshold):
    """Fetch and score all candidates for one Source 1 chunk in one SQL query."""
    prepared = add_features(chunk)
    connection.execute("DROP TABLE IF EXISTS source_chunk")
    connection.execute(
        """CREATE TEMP TABLE source_chunk (
            entity_id TEXT PRIMARY KEY, country TEXT, name_norm TEXT,
            addr_norm TEXT, name_prefix TEXT, addr_num TEXT
        )"""
    )
    connection.executemany(
        "INSERT INTO source_chunk VALUES (?, ?, ?, ?, ?, ?)",
        prepared[["entity_id", "country", "name_norm", "addr_norm",
                   "name_prefix", "addr_num"]].itertuples(index=False, name=None),
    )
    rows = connection.execute(
        """SELECT s.entity_id, s.name_norm, s.addr_norm,
                  t.entity_id, t.name_norm, t.addr_norm
           FROM source_chunk s
           JOIN targets t ON t.country = s.country
             AND (t.name_prefix = s.name_prefix
                  OR (s.addr_num <> '' AND t.addr_num = s.addr_num))"""
    )
    scored = {}
    for source_id, source_name, source_addr, target_id, target_name, target_addr in rows:
        name_score = fuzz.ratio(source_name, target_name)
        addr_score = fuzz.ratio(source_addr, target_addr)
        scored.setdefault(source_id, []).append(
            (0.6 * name_score + 0.4 * addr_score, target_id, name_score, addr_score)
        )
    result = {}
    for source_id in prepared["entity_id"]:
        ranked = sorted(scored.get(source_id, ()), reverse=True)[:max_candidates]
        result[source_id] = (
            {item[1] for item in ranked},
            {
                item[1] for item in ranked
                if item[2] >= name_threshold and item[3] >= address_threshold
            },
        )
    return result


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_index or not args.index.exists():
        build_index(
            args.index,
            [args.test_dir / "test_source2.tsv", args.test_dir / "test_source3.tsv"],
            args.target_chunk_size,
        )
    connection = sqlite3.connect(args.index)
    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    with (
        matching_path.open("w", encoding="utf-8", newline="") as matching_file,
        candidate_path.open("w", encoding="utf-8", newline="") as candidate_file,
    ):
        matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        source1 = args.test_dir / "test_source1.tsv"
        processed = 0
        for chunk in pd.read_csv(source1, sep="\t", usecols=FIELDS, chunksize=args.s1_chunk_size):
            chunk_results = score_chunk(
                connection, chunk, args.max_candidates,
                args.name_threshold, args.address_threshold,
            )
            for source_id in chunk["entity_id"]:
                candidates, matches = chunk_results[source_id]
                candidate_writer.writerow([source_id, ",".join(sorted(candidates))])
                matching_writer.writerow([source_id, ",".join(sorted(matches))])
            matching_file.flush()
            candidate_file.flush()
            processed += len(chunk)
            print(f"Processed {processed:,} Source 1 entities")
    connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DATASET_ROOT / "test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--index", type=Path, default=OUTPUT_ROOT / "targets.sqlite")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--target-chunk-size", type=int, default=25_000)
    parser.add_argument("--s1-chunk-size", type=int, default=1_000)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--name-threshold", type=int, default=85)
    parser.add_argument("--address-threshold", type=int, default=70)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
