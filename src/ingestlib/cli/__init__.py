"""The `ingestlib` command — setup, health checks, and corpus management.

    ingestlib init            write config.yaml + .env for the default stack
    ingestlib init --local    write the zero-cloud config (Ollama + sqlite)
    ingestlib doctor          verify every configured choice with real calls
    ingestlib ingest PATH...  index files or folders into the corpus
    ingestlib sync DIR        reconcile a folder with the corpus
    ingestlib list            show every stored document
    ingestlib remove TARGET   erase a document (by path or doc_id)
    ingestlib backfill        rebuild the vector store from stored artifacts
    ingestlib search "..."    cited retrieval from the shell
    ingestlib mcp             serve the corpus to agents over MCP
"""
import argparse


def _add_namespace(p: argparse.ArgumentParser) -> None:
    p.add_argument("--namespace", default="", help="corpus partition (default: none)")


def main(argv: list[str] | None = None) -> int:
    from ingestlib import __version__

    parser = argparse.ArgumentParser(
        prog="ingestlib",
        description="Setup, health checks, and corpus management for ingestlib.",
    )
    parser.add_argument(
        "--version", action="version", version=f"ingestlib {__version__}"
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

    sub.add_parser("doctor", help="verify the configured stack with real calls")

    ingest_parser = sub.add_parser(
        "ingest", help="index one or more files or folders into the corpus"
    )
    ingest_parser.add_argument("paths", nargs="+", help="files or folders")
    _add_namespace(ingest_parser)

    sync_parser = sub.add_parser(
        "sync", help="reconcile a folder with the corpus (new/changed/moved)"
    )
    sync_parser.add_argument("directory", help="folder to reconcile against")
    sync_parser.add_argument(
        "--prune", action="store_true",
        help="also delete documents whose file is gone (root-scoped)",
    )
    sync_parser.add_argument(
        "--dry-run", action="store_true", help="print the plan, change nothing"
    )
    _add_namespace(sync_parser)

    list_parser = sub.add_parser("list", help="show every stored document")
    list_parser.add_argument(
        "--namespace", default=None, help="only this partition (default: all)"
    )

    remove_parser = sub.add_parser(
        "remove", help="erase a document from both stores (by path or doc_id)"
    )
    remove_parser.add_argument("target", help="source path, or doc_id (full or prefix)")
    _add_namespace(remove_parser)

    backfill_parser = sub.add_parser(
        "backfill", help="rebuild the vector store from stored artifacts"
    )
    _add_namespace(backfill_parser)

    search_parser = sub.add_parser("search", help="cited retrieval from the shell")
    search_parser.add_argument("question", help="the query")
    search_parser.add_argument("--top-k", type=int, default=5, help="how many hits")
    search_parser.add_argument(
        "--no-rerank", action="store_true", help="skip the reranker (vector order)"
    )
    search_parser.add_argument(
        "--sources", default="",
        help="comma-separated sources.yaml names to query (documents and/or SQL databases)",
    )
    _add_namespace(search_parser)

    mcp_parser = sub.add_parser(
        "mcp", help="serve the corpus to agents over MCP (needs ingestlib[mcp])"
    )
    mcp_parser.add_argument(
        "--transport", choices=("stdio", "http"), default="stdio",
        help="stdio for local clients (default); http (streamable) for remote — needs MCP_TOKEN",
    )
    mcp_parser.add_argument("--host", default=None, help="http bind address (default 127.0.0.1)")
    mcp_parser.add_argument("--port", type=int, default=None, help="http port (default 8000)")
    mcp_parser.add_argument(
        "--read-only", action="store_true",
        help="expose only read tools (hide ingest/remove/sync/backfill)",
    )

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:
        # Library errors carry their own fix (a dead server, a missing model,
        # exhausted quota) — surface that one line, not a traceback. Set
        # INGESTLIB_LOG_LEVEL=DEBUG to see the full stack for real debugging.
        import os

        if os.environ.get("INGESTLIB_LOG_LEVEL", "").upper() == "DEBUG":
            raise
        print(f"✗ {exc}")
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "init":
        from ingestlib.cli.scaffold import run_init

        return run_init(local=args.local, force=args.force)
    if args.command == "doctor":
        from ingestlib.cli.doctor import run_doctor

        return run_doctor()
    if args.command == "ingest":
        from ingestlib.cli.corpus import run_ingest

        return run_ingest(args.paths, namespace=args.namespace)
    if args.command == "sync":
        from ingestlib.cli.corpus import run_sync

        return run_sync(
            args.directory, prune=args.prune, dry_run=args.dry_run,
            namespace=args.namespace,
        )
    if args.command == "list":
        from ingestlib.cli.corpus import run_list

        return run_list(namespace=args.namespace)
    if args.command == "remove":
        from ingestlib.cli.corpus import run_remove

        return run_remove(args.target, namespace=args.namespace)
    if args.command == "backfill":
        from ingestlib.cli.corpus import run_backfill

        return run_backfill(namespace=args.namespace)
    if args.command == "search":
        from ingestlib.cli.search import run_search

        return run_search(
            args.question, top_k=args.top_k, namespace=args.namespace,
            rerank=not args.no_rerank,
            sources=[s.strip() for s in args.sources.split(",") if s.strip()] or None,
        )
    if args.command == "mcp":
        from ingestlib.mcp import serve

        serve(transport=args.transport, host=args.host, port=args.port,
              read_only=args.read_only or None)
        return 0
    raise ValueError(f"unknown command {args.command!r}")
