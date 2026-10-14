#!/usr/bin/env python3
# ============================================================
# src/aml_pipeline/eda_cycles.py
#
# Exploratory analysis of laundering cycle patterns from
# HI-Small_Patterns.txt.
#
# Calculates and prints:
#   - Number of cycle patterns
#   - Cycle length distribution (min, max, mean, median)
#   - Per-cycle hop counts
#   - Temporal span of cycles (how many days they span)
#   - Accounts involved in cycles (unique, most frequent)
#   - Comparison with other pattern types
#
# Run: python -m src.aml_pipeline.eda_cycles
# ============================================================

import os
import sys
import re
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aml_pipeline.aml_config import PATTERNS_TXT


# ─────────────────────────────────────────────────────────────
# PATTERN PARSER
# ─────────────────────────────────────────────────────────────

def parse_all_patterns(patterns_path: str) -> list:
    """
    Parse HI-Small_Patterns.txt → list of pattern dicts.

    Each dict:
      pattern_type : str  (CYCLE, FAN-OUT, STACK, etc.)
      description  : str  (descriptor after the colon)
      transactions : list of dicts with keys:
          timestamp, from_bank, from_account, to_bank, to_account,
          amount_received, receiving_currency, amount_paid,
          payment_currency, payment_format, is_laundering
    """
    patterns = []
    current = None

    with open(patterns_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                match = re.match(
                    r"BEGIN LAUNDERING ATTEMPT - ([^:\n]+?)(?::\s*(.*))?$", line
                )
                if match:
                    ptype = match.group(1).strip()
                    desc = (match.group(2) or "").strip()
                else:
                    ptype = "UNKNOWN"
                    desc = line
                current = {
                    "pattern_type": ptype,
                    "description": desc,
                    "transactions": [],
                }
                continue

            if line.startswith("END LAUNDERING ATTEMPT"):
                if current and current["transactions"]:
                    patterns.append(current)
                current = None
                continue

            if current is not None:
                parts = line.split(",")
                if len(parts) >= 11:
                    try:
                        ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M")
                    except ValueError:
                        ts = None
                    current["transactions"].append({
                        "timestamp": parts[0],
                        "datetime": ts,
                        "from_bank": parts[1],
                        "from_account": parts[2],
                        "to_bank": parts[3],
                        "to_account": parts[4],
                        "amount_received": float(parts[5]) if parts[5] else 0.0,
                        "receiving_currency": parts[6],
                        "amount_paid": float(parts[7]) if parts[7] else 0.0,
                        "payment_currency": parts[8],
                        "payment_format": parts[9],
                        "is_laundering": int(parts[10]),
                    })

    return patterns


# ─────────────────────────────────────────────────────────────
# CYCLE-SPECIFIC ANALYSIS
# ─────────────────────────────────────────────────────────────

def is_true_cycle(pattern: dict) -> bool:
    """Check if the first sender is also the last receiver (closed cycle)."""
    txns = pattern["transactions"]
    if len(txns) < 2:
        return False
    first_sender = txns[0]["from_account"]
    last_receiver = txns[-1]["to_account"]
    return first_sender == last_receiver


def get_cycle_accounts(pattern: dict) -> list:
    """Return ordered list of unique accounts in the cycle path."""
    accounts = []
    for txn in pattern["transactions"]:
        if txn["from_account"] not in accounts:
            accounts.append(txn["from_account"])
    return accounts


def get_temporal_span_days(pattern: dict) -> float:
    """Return the temporal span of a pattern in days."""
    datetimes = [t["datetime"] for t in pattern["transactions"] if t["datetime"]]
    if len(datetimes) < 2:
        return 0.0
    span = max(datetimes) - min(datetimes)
    return span.total_seconds() / 86400.0


def get_total_amount(pattern: dict) -> float:
    """Return total amount moved through the pattern."""
    return sum(t["amount_paid"] for t in pattern["transactions"])


# ─────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────

def run_cycle_analysis():
    print("\n" + "=" * 60)
    print("  CYCLE PATTERN ANALYSIS — HI-Small Dataset")
    print("=" * 60)

    if not os.path.exists(PATTERNS_TXT):
        print(f"  [!] Patterns file not found: {PATTERNS_TXT}")
        return

    # Parse all patterns
    all_patterns = parse_all_patterns(PATTERNS_TXT)
    print(f"\n  Total laundering patterns: {len(all_patterns)}")

    # ── Overview: pattern type distribution ────────────────────
    type_counts = Counter(p["pattern_type"] for p in all_patterns)
    print(f"\n  Pattern Type Distribution:")
    print(f"  {'Type':<20} {'Count':>8} {'Share':>8}")
    print(f"  {'-'*40}")
    for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        share = count / len(all_patterns) * 100
        print(f"  {ptype:<20} {count:>8} {share:>7.1f}%")

    # Total transactions across all patterns
    total_txns = sum(len(p["transactions"]) for p in all_patterns)
    print(f"\n  Total laundering transactions: {total_txns:,}")

    # ── Filter to CYCLE patterns only ─────────────────────────
    cycles = [p for p in all_patterns if p["pattern_type"] == "CYCLE"]
    n_cycles = len(cycles)

    if n_cycles == 0:
        print("\n  No CYCLE patterns found.")
        return

    print(f"\n{'='*60}")
    print(f"  CYCLE PATTERNS IN DETAIL ({n_cycles} patterns)")
    print(f"{'='*60}")

    # ── 1. Cycle length distribution (number of hops) ─────────
    hop_counts = [len(c["transactions"]) for c in cycles]
    print(f"\n  1. Cycle Length (number of hops/transactions):")
    print(f"     Min    : {min(hop_counts)}")
    print(f"     Max    : {max(hop_counts)}")
    print(f"     Mean   : {np.mean(hop_counts):.1f}")
    print(f"     Median : {np.median(hop_counts):.1f}")
    print(f"     Std    : {np.std(hop_counts):.1f}")

    hop_dist = Counter(hop_counts)
    print(f"\n     Hop Count Distribution:")
    print(f"     {'Hops':>6} {'Count':>8} {'Pct':>8}")
    print(f"     {'-'*26}")
    for hops in sorted(hop_dist.keys()):
        pct = hop_dist[hops] / n_cycles * 100
        print(f"     {hops:>6} {hop_dist[hops]:>8} {pct:>7.1f}%")

    # ── 2. Closed cycle verification ──────────────────────────
    closed_cycles = [c for c in cycles if is_true_cycle(c)]
    open_cycles = n_cycles - len(closed_cycles)
    print(f"\n  2. Cycle Closure:")
    print(f"     Closed (first sender == last receiver): {len(closed_cycles)}")
    print(f"     Open (not closed):                      {open_cycles}")

    # ── 3. Temporal span ──────────────────────────────────────
    spans = [get_temporal_span_days(c) for c in cycles]
    print(f"\n  3. Temporal Span (days):")
    print(f"     Min    : {min(spans):.2f}")
    print(f"     Max    : {max(spans):.2f}")
    print(f"     Mean   : {np.mean(spans):.2f}")
    print(f"     Median : {np.median(spans):.2f}")

    # Span buckets
    span_buckets = {"<1 day": 0, "1-2 days": 0, "2-5 days": 0, "5+ days": 0}
    for s in spans:
        if s < 1:
            span_buckets["<1 day"] += 1
        elif s < 2:
            span_buckets["1-2 days"] += 1
        elif s < 5:
            span_buckets["2-5 days"] += 1
        else:
            span_buckets["5+ days"] += 1

    print(f"\n     Span Distribution:")
    for bucket, count in span_buckets.items():
        pct = count / n_cycles * 100
        print(f"     {bucket:<12} {count:>5} ({pct:.1f}%)")

    # ── 4. Unique accounts per cycle ──────────────────────────
    accounts_per_cycle = [len(get_cycle_accounts(c)) for c in cycles]
    print(f"\n  4. Unique Accounts per Cycle:")
    print(f"     Min    : {min(accounts_per_cycle)}")
    print(f"     Max    : {max(accounts_per_cycle)}")
    print(f"     Mean   : {np.mean(accounts_per_cycle):.1f}")
    print(f"     Median : {np.median(accounts_per_cycle):.1f}")

    # ── 5. Account participation frequency ────────────────────
    all_cycle_accounts = Counter()
    for c in cycles:
        for txn in c["transactions"]:
            all_cycle_accounts[txn["from_account"]] += 1
            all_cycle_accounts[txn["to_account"]] += 1

    print(f"\n  5. Account Participation in Cycles:")
    print(f"     Total unique accounts: {len(all_cycle_accounts)}")
    top_10 = all_cycle_accounts.most_common(10)
    print(f"     Top 10 most-involved accounts:")
    for acc, count in top_10:
        print(f"       {acc} : {count} appearances")

    # ── 6. Amount statistics ──────────────────────────────────
    total_amounts = [get_total_amount(c) for c in cycles]
    per_hop_amounts = []
    for c in cycles:
        for txn in c["transactions"]:
            per_hop_amounts.append(txn["amount_paid"])

    print(f"\n  6. Amount Statistics:")
    print(f"     Total amount moved (all cycles): {sum(total_amounts):,.2f}")
    print(f"     Per-cycle total:")
    print(f"       Min    : {min(total_amounts):,.2f}")
    print(f"       Max    : {max(total_amounts):,.2f}")
    print(f"       Mean   : {np.mean(total_amounts):,.2f}")
    print(f"       Median : {np.median(total_amounts):,.2f}")
    print(f"     Per-hop amount:")
    print(f"       Min    : {min(per_hop_amounts):,.2f}")
    print(f"       Max    : {max(per_hop_amounts):,.2f}")
    print(f"       Mean   : {np.mean(per_hop_amounts):,.2f}")
    print(f"       Median : {np.median(per_hop_amounts):,.2f}")

    # ── 7. Currency usage in cycles ───────────────────────────
    currencies = Counter()
    for c in cycles:
        for txn in c["transactions"]:
            currencies[txn["payment_currency"]] += 1
    print(f"\n  7. Currencies Used in Cycles:")
    for curr, count in currencies.most_common():
        pct = count / sum(currencies.values()) * 100
        print(f"     {curr:<20} {count:>6} ({pct:.1f}%)")

    # ── 8. Cross-bank hops in cycles ──────────────────────────
    total_hops = 0
    cross_bank_hops = 0
    for c in cycles:
        for txn in c["transactions"]:
            total_hops += 1
            if txn["from_bank"] != txn["to_bank"]:
                cross_bank_hops += 1
    print(f"\n  8. Cross-Bank Hops:")
    print(f"     Total hops      : {total_hops}")
    print(f"     Cross-bank hops : {cross_bank_hops} ({cross_bank_hops/max(total_hops,1)*100:.1f}%)")
    print(f"     Same-bank hops  : {total_hops - cross_bank_hops} ({(total_hops - cross_bank_hops)/max(total_hops,1)*100:.1f}%)")

    # ── 9. Described max hops vs actual hops ──────────────────
    print(f"\n  9. Described vs Actual Hops:")
    print(f"     {'Pattern':<6} {'Described':>12} {'Actual':>8} {'Match':>8}")
    print(f"     {'-'*38}")
    mismatches = 0
    for i, c in enumerate(cycles):
        desc = c["description"]
        # Extract number from description like "Max 10 hops"
        match = re.search(r"Max (\d+)", desc)
        described = int(match.group(1)) if match else -1
        actual = len(c["transactions"])
        match_str = "✓" if described == actual else "✗"
        if described != actual:
            mismatches += 1
        if i < 10 or described != actual:  # show first 10 + all mismatches
            print(f"     {i+1:<6} {described:>12} {actual:>8} {match_str:>8}")
    if mismatches == 0:
        print(f"     (all {n_cycles} cycles match)")
    else:
        print(f"     Mismatches: {mismatches}/{n_cycles}")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CYCLE STATISTICS SUMMARY")
    print(f"{'='*60}")
    print(f"  Cycle patterns          : {n_cycles}")
    print(f"  Total cycle transactions: {sum(hop_counts)}")
    print(f"  Closed cycles           : {len(closed_cycles)}/{n_cycles}")
    print(f"  Avg hops per cycle      : {np.mean(hop_counts):.1f}")
    print(f"  Avg span (days)         : {np.mean(spans):.2f}")
    print(f"  Unique accounts in cycles: {len(all_cycle_accounts)}")
    print(f"  Cross-bank hop rate     : {cross_bank_hops/max(total_hops,1)*100:.1f}%")
    print(f"  Total amount moved      : {sum(total_amounts):,.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_cycle_analysis()
