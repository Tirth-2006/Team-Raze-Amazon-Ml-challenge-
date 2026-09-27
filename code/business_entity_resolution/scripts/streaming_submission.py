"""Disk-backed, bounded-memory rule-based submission pipeline.

The target files are indexed once in SQLite, then Source 1 is processed in
small chunks. No full S2+S3 dataframe or all-results dictionary is retained.
"""
import argparse
import csv
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

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
    frame["name_first"] = frame["name_norm"].map(first_name_token)
    frame["name_top3"] = frame["name_norm"].map(top_name_tokens)
    frame["addr_comp"] = frame["addr_norm"].map(address_composite_key)
    return frame


def first_name_token(value):
    return next((token for token in value.split() if len(token) >= 3), "")


def top_name_tokens(value):
    tokens = sorted((token for token in value.split() if len(token) >= 3),
                    key=lambda token: (-len(token), token))[:3]
    return "|".join(sorted(tokens))


def address_composite_key(value):
    numbers = re.findall(r"\d+", value)
    tokens = [token for token in value.split()
              if len(token) >= 3 and not token.isdigit()]
    if numbers and tokens:
        return f"{numbers[0]}_{tokens[0]}"
    return ""


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
            name_first TEXT NOT NULL, name_top3 TEXT NOT NULL,
            addr_comp TEXT NOT NULL
        )"""
    )
    insert_sql = (
        "INSERT INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
    connection.execute("CREATE INDEX idx_targets_name_first ON targets(country, name_first)")
    connection.execute("CREATE INDEX idx_targets_name_top3 ON targets(country, name_top3)")
    connection.execute("CREATE INDEX idx_targets_addr_comp ON targets(country, addr_comp)")
    connection.execute("CREATE INDEX idx_targets_name_norm ON targets(country, name_norm)")
    connection.commit()
    connection.close()


def ensure_index_schema(connection):
    columns = {row[1] for row in connection.execute("PRAGMA table_info(targets)")}
    required = {"name_first", "name_top3", "addr_comp"}
    if not required.issubset(columns):
        raise RuntimeError(
            "Existing SQLite index uses an obsolete schema. "
            "Rerun with --rebuild-index to build composite blocking keys."
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_targets_name_norm ON targets(country, name_norm)"
    )
    connection.commit()


def score_chunk(connection, chunk, max_candidates, name_threshold, address_threshold):
    """Fetch and score all candidates for one Source 1 chunk.

    Composite blocking keys are queried in three indexed joins. The Python
    dictionary removes duplicates when a target matches multiple passes.
    """
    prepared = add_features(chunk)
    connection.execute("DROP TABLE IF EXISTS source_chunk")
    connection.execute(
        """CREATE TEMP TABLE source_chunk (
            entity_id TEXT PRIMARY KEY, country TEXT, name_norm TEXT,
            addr_norm TEXT, name_first TEXT, name_top3 TEXT, addr_comp TEXT
        )"""
    )
    connection.executemany(
        "INSERT INTO source_chunk VALUES (?, ?, ?, ?, ?, ?, ?)",
        prepared[["entity_id", "country", "name_norm", "addr_norm",
                   "name_first", "name_top3", "addr_comp"]].itertuples(index=False, name=None),
    )
    rows = connection.execute(
        """SELECT s.entity_id, s.name_norm, s.addr_norm,
                  t.entity_id, t.name_norm, t.addr_norm
           FROM source_chunk AS s
           JOIN targets AS t INDEXED BY idx_targets_name_top3
             ON t.country = s.country AND s.name_top3 <> ''
            AND t.name_top3 = s.name_top3
           UNION ALL
           SELECT s.entity_id, s.name_norm, s.addr_norm,
                  t.entity_id, t.name_norm, t.addr_norm
           FROM source_chunk AS s
           JOIN targets AS t INDEXED BY idx_targets_addr_comp
             ON t.country = s.country AND s.addr_comp <> ''
            AND t.addr_comp = s.addr_comp"""
        """ UNION ALL
           SELECT s.entity_id, s.name_norm, s.addr_norm,
                  t.entity_id, t.name_norm, t.addr_norm
           FROM source_chunk AS s
           JOIN targets AS t INDEXED BY idx_targets_name_norm
             ON t.country = s.country AND s.name_norm <> ''
            AND t.name_norm = s.name_norm"""
    )
    candidates = {}
    for source_id, source_name, source_addr, target_id, target_name, target_addr in rows:
        by_target = candidates.setdefault(source_id, {})
        by_target.setdefault(target_id, (target_name, target_addr))

    shortlist_size = max_candidates * 5
    scored = {}
    for source_id, target_data in candidates.items():
        source_row = prepared.loc[prepared["entity_id"].eq(source_id)].iloc[0]
        target_ids = list(target_data)
        target_names = [target_data[target_id][0] for target_id in target_ids]
        name_hits = process.extract(
            source_row["name_norm"],
            target_names,
            scorer=fuzz.ratio,
            limit=shortlist_size,
        )
        ranked = []
        for _, name_score, position in name_hits:
            target_id = target_ids[position]
            target_addr = target_data[target_id][1]
            addr_score = fuzz.ratio(source_row["addr_norm"], target_addr)
            ranked.append((
                0.6 * name_score + 0.4 * addr_score,
                target_id,
                name_score,
                addr_score,
            ))
        scored[source_id] = ranked
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
    ensure_index_schema(connection)
    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    processed = 0
    file_mode = "w"
    if args.resume:
        if not matching_path.exists() or not candidate_path.exists():
            raise RuntimeError("--resume requires both existing output files")
        with matching_path.open("r", encoding="utf-8", newline="") as file:
            matching_rows = list(csv.reader(file, delimiter="\t"))
        with candidate_path.open("r", encoding="utf-8", newline="") as file:
            candidate_rows = list(csv.reader(file, delimiter="\t"))
        if not matching_rows or not candidate_rows:
            raise RuntimeError("--resume requires non-empty output files")
        if matching_rows[0] != ["source1_entity_id", "matched_entity_ids"]:
            raise RuntimeError("Invalid matching output header")
        if candidate_rows[0] != ["source1_entity_id", "candidate_entity_ids"]:
            raise RuntimeError("Invalid candidate output header")
        matching_count = len(matching_rows) - 1
        candidate_count = len(candidate_rows) - 1
        if matching_count != candidate_count:
            raise RuntimeError("Output files contain different row counts")
        if matching_rows[1:] and [row[0] for row in matching_rows[1:]] != [
            row[0] for row in candidate_rows[1:]
        ]:
            raise RuntimeError("Output files contain different Source 1 IDs")
        processed = matching_count
        file_mode = "a"
    with (
        matching_path.open(file_mode, encoding="utf-8", newline="") as matching_file,
        candidate_path.open(file_mode, encoding="utf-8", newline="") as candidate_file,
    ):
        matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        if not args.resume:
            matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        source1 = args.test_dir / "test_source1.tsv"
        skipped = 0
        for chunk in pd.read_csv(source1, sep="\t", usecols=FIELDS, chunksize=args.s1_chunk_size):
            if skipped < args.start_row:
                remaining = args.start_row - skipped
                if len(chunk) <= remaining:
                    skipped += len(chunk)
                    continue
                chunk = chunk.iloc[remaining:].copy()
                skipped = args.start_row
            if args.end_row is not None and skipped + len(chunk) > args.end_row:
                chunk = chunk.iloc[:args.end_row - skipped].copy()
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
            skipped += len(chunk)
            print(f"Processed range rows {args.start_row:,}-{skipped:,}")
            if args.end_row is not None and skipped >= args.end_row:
                break
    connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=DATASET_ROOT / "test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--index", type=Path, default=OUTPUT_ROOT / "targets.sqlite")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-chunk-size", type=int, default=25_000)
    parser.add_argument("--s1-chunk-size", type=int, default=1_000)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--name-threshold", type=int, default=85)
    parser.add_argument("--address-threshold", type=int, default=70)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
