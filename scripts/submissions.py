#!/usr/bin/env python3
"""Accept a measurement from elsewhere without letting it into the tables.

A result someone else took is worth keeping and worth showing. It is not worth merging: the
protocol it was taken under, the machine it ran on and the build that answered are all
unverified, and a table that mixes them with our own reads as one measurement.

So a submission is quarantined by construction. It enters a pool, it is validated against the
schema and nothing else, and it leaves the pool only when one of our own records sits in the
same equivalence class - which is what "reproduced here" means and the only thing that earns a
row a place beside ours.

    submissions.py file <record.json> --pool <dir>        # take one in, or refuse it
    submissions.py review --pool <dir> --ours <record.json> [more...]
"""
import argparse, json, pathlib, shutil, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import classes as C


def read(path):
    try:
        return json.loads(pathlib.Path(path).read_text()), None
    except Exception as e:
        return None, str(e)


def file_one(path: pathlib.Path, pool: pathlib.Path) -> int:
    record, why = read(path)
    if record is None:
        print(f"refused {path.name}: not readable json ({why})", file=sys.stderr)
        return 1
    if C.major(record.get("schema")) is None:
        print(f"refused {path.name}: no readable schema - a record must say what it is",
              file=sys.stderr)
        return 1
    if not record.get("results"):
        print(f"refused {path.name}: no cells in it", file=sys.stderr)
        return 1
    pool.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, pool / path.name)
    print(f"quarantined {path.name}: {len(record['results'])} cell(s), class:")
    for line in C.class_of(record):
        print(f"    {line}")
    print("  it stays here until one of ours is measured in the same class")
    return 0


def review(pool: pathlib.Path, ours) -> int:
    mine = {}
    for path in ours:
        record, why = read(path)
        if record is None or C.major(record.get("schema")) is None:
            print(f"  ignoring {pathlib.Path(path).name}: {why or 'no schema'}", file=sys.stderr)
            continue
        mine.setdefault(C.class_of(record), []).append(pathlib.Path(path).name)

    promotable, held = [], []
    for path in sorted(pool.glob("*.json")):
        record, _ = read(path)
        if record is None:
            continue
        key = C.class_of(record)
        (promotable if key in mine else held).append((path.name, key))

    print(f"pool: {len(promotable) + len(held)} record(s)")
    for name, key in promotable:
        print(f"  REPRODUCIBLE  {name}")
        print(f"                we hold {', '.join(mine[key])} in the same class")
    for name, key in held:
        print(f"  held          {name}")
        print(f"                no record of ours in class: {key[3]} / {key[4]}")
    print("\nA held record is not wrong. It is one person's measurement on hardware we do not"
          "\nhave, and it is shown as that or not at all.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("file"); f.add_argument("record", type=pathlib.Path); f.add_argument("--pool", required=True, type=pathlib.Path)
    r = sub.add_parser("review"); r.add_argument("--pool", required=True, type=pathlib.Path); r.add_argument("--ours", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "file":
        return file_one(a.record, a.pool)
    return review(a.pool, a.ours)


if __name__ == "__main__":
    raise SystemExit(main())
