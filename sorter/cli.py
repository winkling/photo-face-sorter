import argparse
import os
import sys

from .config import load_config
from .pipeline import run_scan, run_commit, run_status


def main():
    parser = argparse.ArgumentParser(
        prog="sorter",
        description="Sort photos into per-person folders using face recognition.",
    )
    parser.add_argument(
        "--config", default="config.yaml", metavar="PATH",
        help="Path to config.yaml (default: ./config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    scan_p = sub.add_parser("scan", help="Detect and route photos")
    scan_p.add_argument("input_dir", nargs="?", default=None,
                        help="Input directory (overrides config)")
    scan_p.add_argument("--apply", action="store_true",
                        help="Actually place files (default is dry run)")

    # commit
    sub.add_parser("commit", help="Enroll labeled review groups")

    # status
    sub.add_parser("status", help="Show database and queue stats")

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "scan":
        input_dir = args.input_dir or cfg["input_dir"]
        input_dir = os.path.expanduser(input_dir)
        if not os.path.isdir(input_dir):
            print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
            sys.exit(1)
        run_scan(cfg, input_dir, apply=args.apply)

    elif args.command == "commit":
        run_commit(cfg)

    elif args.command == "status":
        run_status(cfg)


if __name__ == "__main__":
    main()
