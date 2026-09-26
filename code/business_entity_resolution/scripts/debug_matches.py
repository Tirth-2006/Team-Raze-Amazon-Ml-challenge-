"""
Debug script to understand why matches are failing
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from normalization import normalize_business_name, normalize_address

print("="*80)
print("DEBUG: Checking why matches are failing")
print("="*80)

base = '../../../student_resource/dataset/train'

# Load first 100 S1 entities
print("\nLoading first 100 S1 entities...")
s1 = pd.read_csv(f'{base}/train_source1.tsv', sep='\t', nrows=100)
ground_truth = pd.read_csv(f'{base}/train_ground_truth.tsv', sep='\t', nrows=100)

# Get the first entity with matches
first_entity = None
for _, row in ground_truth.iterrows():
    s1_id = row['source1_entity_id']
    matched_ids = row['matched_entity_ids']

    if pd.notna(matched_ids) and matched_ids != '':
        first_entity = s1_id
        first_matches = matched_ids.split(',')
        break

if not first_entity:
    print("ERROR: No entities with matches found!")
    sys.exit(1)

print(f"\nFirst entity with matches: {first_entity}")
print(f"Ground truth matches: {first_matches}")

# Get S1 details - need to load full S1 to find this entity
print(f"Loading full S1 to find entity {first_entity}...")
s1_full = pd.read_csv(f'{base}/train_source1.tsv', sep='\t')
s1_row = s1_full[s1_full['entity_id'] == first_entity].iloc[0]
print(f"\nS1 Entity Details:")
print(f"  Name: {s1_row['business_name']}")
print(f"  Address: {s1_row['business_address']}")
print(f"  Country: {s1_row['country']}")

# Try to load the matched entities
print(f"\nLooking for matched entities in full dataset...")

# Check if they're S2 or S3
s2_matches = [m for m in first_matches if m.startswith('S2-')]
s3_matches = [m for m in first_matches if m.startswith('S3-')]

print(f"  S2 matches: {len(s2_matches)}")
print(f"  S3 matches: {len(s3_matches)}")

if s2_matches:
    print(f"\nSearching for {s2_matches[0]} in S2...")
    # Load S2 in chunks and search
    found = False
    chunk_num = 0
    for chunk in pd.read_csv(f'{base}/train_source2.tsv', sep='\t', chunksize=100000):
        chunk_num += 1
        if s2_matches[0] in chunk['entity_id'].values:
            match_row = chunk[chunk['entity_id'] == s2_matches[0]].iloc[0]
            print(f"  FOUND in chunk {chunk_num}!")
            print(f"  Name: {match_row['business_name']}")
            print(f"  Address: {match_row['business_address']}")
            print(f"  Country: {match_row['country']}")

            # Compare similarity
            from rapidfuzz import fuzz
            s1_name_norm = normalize_business_name(s1_row['business_name'])
            s2_name_norm = normalize_business_name(match_row['business_name'])
            s1_addr_norm = normalize_address(s1_row['business_address'])
            s2_addr_norm = normalize_address(match_row['business_address'])

            name_sim = fuzz.ratio(s1_name_norm, s2_name_norm)
            addr_sim = fuzz.ratio(s1_addr_norm, s2_addr_norm)

            print(f"\n  Similarity scores:")
            print(f"    Name: {name_sim}/100 (threshold: 85)")
            print(f"    Address: {addr_sim}/100 (threshold: 70)")
            print(f"    Normalized S1 name: {s1_name_norm}")
            print(f"    Normalized S2 name: {s2_name_norm}")
            print(f"    Name prefix (3 chars): S1='{s1_name_norm[:3]}' vs S2='{s2_name_norm[:3]}'")

            if name_sim >= 85 and addr_sim >= 70:
                print(f"\n  => Would PASS with thresholds 85/70")
            else:
                print(f"\n  => Would FAIL with thresholds 85/70")
                print(f"     This is a TRUE match but below thresholds!")

            found = True
            break

        if chunk_num >= 10:  # Stop after 1M rows
            break

    if not found:
        print(f"  NOT FOUND in first 1M rows of S2")

print("\n" + "="*80)
print("Analysis complete")
print("="*80)
