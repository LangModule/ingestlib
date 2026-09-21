"""The `ingestlib` command — setup, health checks, and corpus management.

    ingestlib init            write config.yaml + .env for the default stack
    ingestlib init --local    write the zero-cloud config (Ollama + sqlite)
    ingestlib doctor          verify every configured choice with real calls
    ingestlib ingest PATH...  index files or folders into the corpus
    ingestlib sync DIR        reconcile a folder with the corpus
    ingestlib list            show every stored document
    ingestlib show TARGET     print everything stored about one document
    ingestlib collections     list the collections and their document counts
    ingestlib recollect       re-classify the corpus after changing rules.yaml
    ingestlib verify          audit durability across the three stores
    ingestlib remove TARGET   erase a document (by path or doc_id)
    ingestlib reindex         rebuild the vector store from the registry
    ingestlib search "..."    cited retrieval from the shell
    ingestlib describe-schema NAME   auto-document a SQL source's tables
    ingestlib eval-sql NAME   measure text2SQL accuracy on your own schema
    ingestlib mcp             serve the corpus to agents over MCP
    ingestlib registry init | status | backup | restore   manage the registry DB
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

    show_parser = sub.add_parser(
        "show", help="print everything stored about one document"
    )
    show_parser.add_argument("target", help="source path, or doc_id (full or prefix)")

    collections_parser = sub.add_parser(
        "collections", help="list the collections in the corpus and their counts"
    )
    collections_parser.add_argument(
        "--namespace", default=None, help="only this partition (default: all)"
    )

    recollect_parser = sub.add_parser(
        "recollect", help="re-classify the stored corpus after changing rules.yaml"
    )
    recollect_parser.add_argument(
        "--namespace", default=None, help="only this partition (default: all)"
    )

    verify_parser = sub.add_parser(
        "verify", help="audit durability (registry vs vector store vs blob store)"
    )
    verify_parser.add_argument(
        "--namespace", default=None, help="only this partition (default: all)"
    )
    verify_parser.add_argument(
        "--repair", action="store_true",
        help="re-embed vector-drifted documents from their registry chunks",
    )

    remove_parser = sub.add_parser(
        "remove", help="erase a document from both stores (by path or doc_id)"
    )
    remove_parser.add_argument("target", help="source path, or doc_id (full or prefix)")
    _add_namespace(remove_parser)

    reindex_parser = sub.add_parser(
        "reindex", help="rebuild the vector store from the registry (re-embed, no re-parse)"
    )
    _add_namespace(reindex_parser)

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

    describe_parser = sub.add_parser(
        "describe-schema", help="auto-document a SQL source's tables (LLM-generated hints)"
    )
    describe_parser.add_argument("source", help="a SQL source name from sources.yaml")
    describe_parser.add_argument(
        "--out", default=None, help="write the tables: block to this file (default: stdout)"
    )

    eval_parser = sub.add_parser(
        "eval-sql", help="measure text2SQL accuracy on a source against a question set"
    )
    eval_parser.add_argument("source", help="a SQL source name from sources.yaml")
    eval_parser.add_argument(
        "--dataset", default=None,
        help="YAML of question/expected pairs (default: <source>_eval.yaml beside sources.yaml)",
    )

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
        help="expose only read tools (hide ingest/remove/sync/reindex)",
    )

    registry_parser = sub.add_parser(
        "registry", help="manage the internal registry database (schema migrations)"
    )
    registry_sub = registry_parser.add_subparsers(dest="registry_command", required=True)
    registry_sub.add_parser(
        "init", help="create or upgrade the registry schema to the latest revision"
    )
    registry_sub.add_parser(
        "status", help="show the registry's current revision and reachability"
    )
    registry_sub.add_parser(
        "backup", help="pg_dump the registry into the blob store"
    )
    registry_sub.add_parser(
        "restore", help="restore the registry from the latest blob-store backup"
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
    if args.command == "show":
        from ingestlib.cli.corpus import run_show

        return run_show(args.target)
    if args.command == "collections":
        from ingestlib.cli.corpus import run_collections

        return run_collections(namespace=args.namespace)
    if args.command == "recollect":
        from ingestlib.cli.corpus import run_recollect

        return run_recollect(namespace=args.namespace)
    if args.command == "verify":
        from ingestlib.cli.corpus import run_verify

        return run_verify(namespace=args.namespace, repair=args.repair)
    if args.command == "remove":
        from ingestlib.cli.corpus import run_remove

        return run_remove(args.target, namespace=args.namespace)
    if args.command == "reindex":
        from ingestlib.cli.corpus import run_reindex

        return run_reindex(namespace=args.namespace)
    if args.command == "search":
        from ingestlib.cli.search import run_search

        return run_search(
            args.question, top_k=args.top_k, namespace=args.namespace,
            rerank=not args.no_rerank,
            sources=[s.strip() for s in args.sources.split(",") if s.strip()] or None,
        )
    if args.command == "describe-schema":
        from ingestlib.cli.describe import run_describe_schema

        return run_describe_schema(args.source, out=args.out)
    if args.command == "eval-sql":
        from ingestlib.cli.eval_sql import run_eval_sql

        return run_eval_sql(args.source, dataset=args.dataset)
    if args.command == "mcp":
        from ingestlib.mcp import serve

        serve(transport=args.transport, host=args.host, port=args.port,
              read_only=args.read_only or None)
        return 0
    if args.command == "registry":
        if args.registry_command == "init":
            from ingestlib.cli.registry import run_registry_init

            return run_registry_init()
        if args.registry_command == "status":
            from ingestlib.cli.registry import run_registry_status

            return run_registry_status()
        if args.registry_command == "backup":
            from ingestlib.cli.registry import run_registry_backup

            return run_registry_backup()
        if args.registry_command == "restore":
            from ingestlib.cli.registry import run_registry_restore

            return run_registry_restore()
        raise ValueError(f"unknown registry command {args.registry_command!r}")
    raise ValueError(f"unknown command {args.command!r}")
