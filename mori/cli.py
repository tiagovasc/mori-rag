import argparse
import json
from pathlib import Path
import sys

from .config import load
from .ingest import fetch_public, ingest, load_file
from .pipeline import plan, query
from .providers import Embeddings, LLM
from .store import Store


def main(argv=None):
    parser = argparse.ArgumentParser(description='Mori: RAG with whole-source context')
    parser.add_argument('--config', help='TOML configuration file')
    parser.add_argument('--data', default='.mori', help='Private local corpus/run directory')
    sub = parser.add_subparsers(dest='command', required=True)
    initialize = sub.add_parser('init', help='Choose MiniLM or OpenAI embeddings and write a new configuration')
    initialize.add_argument('--embeddings', choices=('minilm', 'openai'), default='minilm')
    initialize.add_argument('--output', default='mori.toml')
    sub.add_parser('status', help='Inspect corpus without provider calls')
    sub.add_parser('plan', help='Review rating decision budget without provider calls')
    importer = sub.add_parser('ingest', help='Import explicit units or text files')
    importer.add_argument('files', nargs='+')
    importer.add_argument('--split', choices=('whole', 'headings'), default='whole')
    importer.add_argument('--summarize', action='store_true', help='Generate missing summaries using configured LLM')
    fetcher = sub.add_parser('fetch', help='Explicit public HTTPS article import; no crawling')
    fetcher.add_argument('url')
    fetcher.add_argument('--summarize', action='store_true')
    asking = sub.add_parser('query', help='Build ranked evidence packet; optionally generate an answer')
    asking.add_argument('question')
    asking.add_argument('--answer', action='store_true')
    room = sub.add_parser('serve', help='Open the local upload/research interface')
    room.add_argument('--port', type=int, default=8840)
    export = sub.add_parser('export', help='Export portable units with summaries as JSONL, excluding vectors')
    export.add_argument('output')
    args = parser.parse_args(argv)
    try:
        config = load(args.config)
        if args.command == 'init':
            target = Path(args.output)
            if target.exists():
                raise ValueError('Configuration already exists; choose a new output path')
            content = (Path(__file__).parent / 'presets' / (args.embeddings + '.toml')).read_text(encoding='utf-8')
            target.write_text(content, encoding='utf-8')
            print(json.dumps({'config': str(target), 'embedding_preset': args.embeddings, 'note': 'Use a separate data directory when changing embedding models'}))
            return
        store = Store(args.data)
        if args.command == 'status':
            result = store.stats()
        elif args.command == 'plan':
            result = plan(store, config)
        elif args.command in ('ingest', 'fetch'):
            units = fetch_public(args.url) if args.command == 'fetch' else [u for path in args.files for u in load_file(path, args.split)]
            result = ingest(store, units, config, Embeddings(config), LLM(config), args.summarize)
        elif args.command == 'query':
            result = query(store, args.question, config, answer=args.answer)
        elif args.command == 'serve':
            from .web import serve
            if not 1024 <= args.port <= 65535:
                raise ValueError('Port must be 1024–65535')
            serve(store, config, args.port)
            return
        else:
            output = Path(args.output)
            if output.exists():
                raise ValueError('Export target already exists; choose a new path')
            rows = [{k: v for k, v in u.items() if k not in ('summary_vector', 'hash', 'created')} for u in store.units()]
            output.write_text(''.join(json.dumps(u, ensure_ascii=False) + '\n' for u in rows), encoding='utf-8')
            result = {'exported_units': len(rows), 'path': str(output)}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError) as exc:
        print('Mori: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
