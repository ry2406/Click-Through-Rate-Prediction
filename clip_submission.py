"""
Post-process submission: clip predicted probabilities to [0, 1].

The weighted ensemble can produce values slightly below 0 due to
negative blending weights.  This script clamps them so the output
is a valid probability distribution.

Usage:
    python clip_submission.py
"""

import pandas as pd

INPUT_FILE  = "submission.csv"
OUTPUT_FILE = "submission.csv"   # overwrite in place

df = pd.read_csv(INPUT_FILE)

n_neg = (df["click"] < 0).sum()
n_above = (df["click"] > 1).sum()

print(f"Before clip:")
print(f"  Rows < 0 : {n_neg}")
print(f"  Rows > 1 : {n_above}")
print(f"  Min       : {df['click'].min():.6f}")
print(f"  Max       : {df['click'].max():.6f}")

df["click"] = df["click"].clip(0, 1)

print(f"\nAfter clip:")
print(f"  Min       : {df['click'].min():.6f}")
print(f"  Max       : {df['click'].max():.6f}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")