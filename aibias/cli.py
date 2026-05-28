"""Command-line interface: `python -m aibias.cli ...`"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import (
    TextBiasDetector,
    ImageBiasDetector,
    BiasMitigator,
    BiasAuditor,
)


def _read_text_input(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    p = Path(value)
    if p.is_file():
        return p.read_text()
    return value  # treat as literal string


def _emit(obj, fmt: str):
    if fmt == "json":
        print(json.dumps(obj, indent=2, default=str))
    elif fmt == "markdown":
        if hasattr(obj, "to_markdown"):
            print(obj.to_markdown())
        elif hasattr(obj, "summary"):
            print(obj.summary())
        else:
            print(json.dumps(obj, indent=2, default=str))
    else:
        if hasattr(obj, "summary"):
            print(obj.summary())
        else:
            print(obj)


def cmd_text(args):
    text = _read_text_input(args.input)
    report = TextBiasDetector().analyze(text)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())


def cmd_image(args):
    paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(
                str(q) for q in p.rglob("*")
                if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            )
        else:
            paths.append(str(p))
    report = ImageBiasDetector().analyze(paths)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.summary())


def cmd_mitigate(args):
    text = _read_text_input(args.input)
    m = BiasMitigator()
    clean, report = m.neutralize(text)
    if args.format == "json":
        print(json.dumps({
            "neutralized_text": clean,
            "report": {
                "replacements": report.replacements,
                "removed_terms": report.removed_terms,
            },
        }, indent=2))
    else:
        print("=== Neutralized text ===")
        print(clean)
        print()
        print("=== Mitigation report ===")
        print(report.summary())


def cmd_audit(args):
    texts = []
    image_paths = []
    if args.text:
        texts = [_read_text_input(t) for t in args.text]
    if args.images:
        for inp in args.images:
            p = Path(inp)
            if p.is_dir():
                image_paths.extend(
                    str(q) for q in p.rglob("*")
                    if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                )
            else:
                image_paths.append(str(p))
    classifier_outputs = None
    if args.classifier_json:
        classifier_outputs = json.loads(Path(args.classifier_json).read_text())

    auditor = BiasAuditor()
    report = auditor.audit(
        texts=texts or None,
        image_paths=image_paths or None,
        classifier_outputs=classifier_outputs,
    )
    if args.out:
        paths = auditor.save(report, args.out)
        print(f"Wrote {paths['json']} and {paths['markdown']}")
    if args.format == "json":
        print(report.to_json())
    else:
        print(report.to_markdown())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aibias", description="AI bias detection toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("text", help="Analyze text for bias")
    t.add_argument("--input", required=True, help="text, file path, or '-' for stdin")
    t.add_argument("--format", choices=["text", "json"], default="text")
    t.set_defaults(func=cmd_text)

    i = sub.add_parser("image", help="Analyze image(s) for bias")
    i.add_argument("--inputs", nargs="+", required=True, help="files and/or directories")
    i.add_argument("--format", choices=["text", "json"], default="text")
    i.set_defaults(func=cmd_image)

    m = sub.add_parser("mitigate", help="Neutralize biased language in text")
    m.add_argument("--input", required=True)
    m.add_argument("--format", choices=["text", "json"], default="text")
    m.set_defaults(func=cmd_mitigate)

    a = sub.add_parser("audit", help="End-to-end audit (text + images + classifier)")
    a.add_argument("--text", nargs="*", help="texts or file paths")
    a.add_argument("--images", nargs="*", help="image files or directories")
    a.add_argument("--classifier-json", help="path to JSON with classifier_outputs")
    a.add_argument("--out", help="directory to write bias_audit.{json,md}")
    a.add_argument("--format", choices=["markdown", "json"], default="markdown")
    a.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
