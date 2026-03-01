import argparse
import math
import struct
import time
from collections import Counter

import lmdb


DEFAULT_THRESHOLDS = [1, 2, 3, 5, 10, 20, 50, 100]


def _parse_thresholds(raw: str):
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    vals = sorted(set(v for v in vals if v >= 0))
    return vals if vals else DEFAULT_THRESHOLDS


def analyze_visit_distribution(
    lmdb_path: str,
    value_fmt: str = "dI",
    key_bytes: int = 32,
    max_entries: int | None = None,
    progress_every: int = 500_000,
    thresholds: list[int] | None = None,
):
    thresholds = sorted(thresholds or DEFAULT_THRESHOLDS)
    value_size = struct.calcsize(value_fmt)
    unpack = struct.unpack

    env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=False,
    )

    total = 0
    valid = 0
    bad_key_len = 0
    bad_val_len = 0

    visit_sum = 0
    visit_min = None
    visit_max = None
    exact_small_counts = Counter()  # exact for visits up to 1000
    log2_bins = Counter()
    ge_threshold = {t: 0 for t in thresholds}

    start = time.time()
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            total += 1
            if max_entries is not None and total > max_entries:
                break

            if progress_every and total % progress_every == 0:
                elapsed = max(time.time() - start, 1e-9)
                print(f"[scan] {total:,} entries ({total / elapsed:,.0f} entries/s)")

            if len(k) != key_bytes:
                bad_key_len += 1
                continue
            if len(v) != value_size:
                bad_val_len += 1
                continue

            _q_value, visits = unpack(value_fmt, v)
            visits = int(visits)
            valid += 1

            visit_sum += visits
            visit_min = visits if visit_min is None else min(visit_min, visits)
            visit_max = visits if visit_max is None else max(visit_max, visits)

            if visits <= 1000:
                exact_small_counts[visits] += 1

            b = 0 if visits == 0 else int(math.log2(visits))
            log2_bins[b] += 1

            for t in thresholds:
                if visits >= t:
                    ge_threshold[t] += 1

    env.close()

    avg_visits = (visit_sum / valid) if valid else 0.0
    return {
        "total": total,
        "valid": valid,
        "bad_key_len": bad_key_len,
        "bad_val_len": bad_val_len,
        "visit_min": visit_min,
        "visit_max": visit_max,
        "avg_visits": avg_visits,
        "exact_small_counts": exact_small_counts,
        "log2_bins": log2_bins,
        "ge_threshold": ge_threshold,
        "thresholds": thresholds,
    }


def print_report(stats):
    valid = max(stats["valid"], 1)
    print("\n===== VISIT COUNT DISTRIBUTION =====")
    print(f"Scanned entries : {stats['total']:,}")
    print(f"Valid entries   : {stats['valid']:,}")
    print(f"Bad key length  : {stats['bad_key_len']:,}")
    print(f"Bad value length: {stats['bad_val_len']:,}")
    print(f"Min visits      : {stats['visit_min']}")
    print(f"Max visits      : {stats['visit_max']}")
    print(f"Avg visits      : {stats['avg_visits']:.4f}")

    print("\n----- Keep Ratio by Threshold (visits >= T) -----")
    for t in stats["thresholds"]:
        keep = stats["ge_threshold"][t]
        keep_pct = 100.0 * keep / valid
        drop = stats["valid"] - keep
        drop_pct = 100.0 - keep_pct
        print(
            f"T={t:>4}: keep={keep:>12,} ({keep_pct:6.2f}%), "
            f"drop={drop:>12,} ({drop_pct:6.2f}%)"
        )

    print("\n----- Exact counts for visits 0..20 -----")
    for v in range(0, 21):
        c = stats["exact_small_counts"].get(v, 0)
        pct = 100.0 * c / valid
        print(f"visits={v:>2}: {c:>12,} ({pct:6.2f}%)")

    print("\n----- Log2 bins (visits in [2^b, 2^(b+1)-1]) -----")
    for b in sorted(stats["log2_bins"]):
        lo = 0 if b == 0 else (1 << b)
        hi = (1 << (b + 1)) - 1
        c = stats["log2_bins"][b]
        pct = 100.0 * c / valid
        print(f"[{lo:>7},{hi:>7}] : {c:>12,} ({pct:6.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LMDB key visit-count distribution and pruning impact."
    )
    parser.add_argument("--lmdb-path", default="fixed_qtable.lmdb")
    parser.add_argument("--value-fmt", default="dI")
    parser.add_argument("--key-bytes", type=int, default=32)
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500_000)
    parser.add_argument(
        "--thresholds",
        default="1,2,3,5,10,20,50,100",
        help="Comma-separated thresholds for keep/drop report.",
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    stats = analyze_visit_distribution(
        lmdb_path=args.lmdb_path,
        value_fmt=args.value_fmt,
        key_bytes=args.key_bytes,
        max_entries=args.max_entries,
        progress_every=args.progress_every,
        thresholds=thresholds,
    )
    print_report(stats)


if __name__ == "__main__":
    main()
