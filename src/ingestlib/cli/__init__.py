"""The `ingestlib` command — setup scaffold and stack verification.

    ingestlib init            write config.yaml + .env for the default stack
    ingestlib init --local    write the zero-cloud config (Ollama + sqlite)
    ingestlib doctor          verify every configured choice with real calls
"""
import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ingestlib",
        description="Setup and health checks for ingestlib.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser(
        "init", help="write config.yaml (+ .env) into the current directory"
    )
    init_parser.add_argument(
        "--local", action="store_true",
        help="zero-cloud preset: Ollama + sqlite + local artifacts, no keys",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="overwrite existing files"
    )

    sub.add_parser(
        "doctor", help="verify the configured stack with real calls"
    )

    args = parser.parse_args(argv)
    if args.command == "init":
        from ingestlib.cli.scaffold import run_init

        return run_init(local=args.local, force=args.force)
    from ingestlib.cli.doctor import run_doctor

    return run_doctor()
