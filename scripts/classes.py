#!/usr/bin/env python3
"""Say which measurements may be compared with which, and refuse the rest.

A table of numbers from several machines, builds and protocols is not one measurement. The
cell key already separates a model from a model; this separates a *run* from a run, so that
two rows only ever meet when what produced them agrees.

A class is the tuple a comparison is honest within:

    schema major, protocol profile, the flags the profile declares significant,
    the machine, and the backend each engine answered on

Anything the record does not state is `unknown`, and unknown is a class of its own rather
than a wildcard: an archived run whose machine was never recorded cannot quietly join a run
whose machine is known, because the difference between them is exactly what is unrecorded.

    classes.py <record.json> [more.json ...]
"""
import argparse, json, pathlib, sys

SCHEMA_NAME = "assay"

# The flags a comparison is only honest within. A run that asserts none of them is not
# thereby equal to one that asserts them - it is a run that did not say.
SIGNIFICANT = ("streaming", "max_tokens", "iterations", "warmup", "engines_alone")


def major(schema):
    if not schema or "/" not in schema:
        return None
    name, _, version = schema.partition("/")
    if name != SCHEMA_NAME:
        return None
    return version.split(".")[0]


def machine_of(record):
    m = record.get("machine")
    if not m:
        return "unknown machine"
    cards = ",".join(sorted(g.get("uuid") or g.get("name") or "?" for g in m.get("gpus", [])))
    return f"{m.get('cpu') or '?'} | {m.get('kernel') or '?'} | driver {m.get('driver') or '?'} | {cards or 'no card'}"


def backends_of(record):
    subjects = record.get("subjects")
    if not subjects:
        return "unknown engines"
    parts = []
    for name in sorted(subjects):
        said = (subjects[name] or {}).get("says") or {}
        # A commit where one is offered, a version otherwise, and the name alone when the
        # server answered nothing: three different facts, never collapsed.
        stamp = said.get("commit") or said.get("version") or "unstated"
        parts.append(f"{name}@{stamp}")
    return ", ".join(parts)


def class_of(record):
    m = major(record.get("schema"))
    protocol = record.get("protocol") or {}
    flags = tuple(f"{k}={protocol.get(k)!r}" for k in SIGNIFICANT)
    return (
        f"schema {m or 'unreadable'}",
        protocol.get("profile") or "no profile",
        " ".join(flags),
        machine_of(record),
        backends_of(record),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", type=pathlib.Path)
    args = ap.parse_args()

    classes, unreadable = {}, []
    for path in args.records:
        try:
            record = json.loads(path.read_text())
        except Exception as e:
            unreadable.append((path, e))
            continue
        if major(record.get("schema")) is None:
            unreadable.append((path, "no readable schema"))
            continue
        classes.setdefault(class_of(record), []).append((path, len(record.get("results", []))))

    for key, members in sorted(classes.items(), key=lambda kv: -len(kv[1])):
        cells = sum(n for _, n in members)
        print(f"\nclass of {len(members)} record(s), {cells} cell(s)")
        for line in key:
            print(f"  {line}")
        for path, n in members[:4]:
            print(f"    - {path.name} ({n} cells)")
        if len(members) > 4:
            print(f"    - and {len(members) - 4} more")

    if unreadable:
        print(f"\nrefused {len(unreadable)} record(s):", file=sys.stderr)
        for path, why in unreadable[:5]:
            print(f"  {path.name}: {why}", file=sys.stderr)
    print(f"\n{len(classes)} class(es). Rows may be compared inside one, never across two.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
