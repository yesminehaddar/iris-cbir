"""Build the fuzzy-matching vocabulary for a base that is already indexed.

    python scripts/build_tagvocab.py flickr1m
    python scripts/build_tagvocab.py flickr1m --show vien

The vocabulary is every word the full-text index can match, with how many images
carry it, read straight out of the tsvector column. Fuzzy search compares a typed
word against *this* -- 295 000 known words -- rather than against 805 842
documents, which is what makes a near-miss lookup cost single-digit
milliseconds.

`build_index.py` and `import_mirflickr.py --tags` both rebuild it, since both
change what words exist. This script exists so an index built earlier can get
one without redoing the descriptor pass, and so the ranking can be inspected:
`--show <word>` prints what a typed word would expand to, and why.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import datasets, db


def build(dataset):
    """Rebuild the vocabulary for one base, reporting how many words it holds."""
    with db.connect() as conn:
        db.init_schema()
        if not db.count(conn, dataset.name):
            raise SystemExit(
                f"{dataset.name} has no rows; run build_index.py {dataset.name} first"
            )
        print(f"reading the lexemes of {dataset.name} (one pass over the tsvectors)")
        started = time.perf_counter()
        written = db.build_tag_vocab(conn, dataset.name)
        elapsed = time.perf_counter() - started

    if not written:
        print("no vocabulary written; fuzzy matching stays off")
        return
    print(
        f"{written:,} words in {elapsed:.0f}s "
        f"(words in {db.VOCAB_MIN_NDOC}+ images, filename tokens excluded)"
    )


def show(dataset, term):
    """Print what one typed word expands to, with the score behind each choice."""
    with db.connect() as conn:
        if not db.has_trigrams(conn):
            raise SystemExit("pg_trgm is not installed, so there is nothing to show")
        terms = db.lexemes_of(conn, term)
        if not terms:
            raise SystemExit(f"{term!r} contains no searchable word")

        for one in terms:
            print(f"\n{one!r} expands to:")
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM ({db.EXPANSION_CANDIDATES}) AS scored
                    ORDER BY too_rare, score DESC, ndoc DESC, word
                    LIMIT %(limit)s
                    """,
                    {
                        **db.expansion_params(dataset.name, one),
                        "limit": db.EXPANSIONS_PER_TERM + 4,
                    },
                )
                rows = cur.fetchall()
            if not rows:
                print("  (nothing similar enough; only the exact word is searched)")
            elif rows[0]["typed_ndoc"]:
                print(f"  (typed word is in {rows[0]['typed_ndoc']:,} images)")
            for row in rows:
                print(
                    f"  {row['word']:<24} {row['ndoc']:>7,} images   "
                    f"sim {row['sim']:.2f}"
                    f"{'  prefix' if row['is_prefix'] else '        '}"
                    f"   score {row['score']:.2f}"
                    f"{'   DROPPED: too rare beside the typed word' if row['too_rare'] else ''}"
                )

        query, expansions = db.build_query(conn, dataset.name, term, fuzzy=True)
        print(f"\ntsquery: {query}")
        if expansions:
            hits = db.keyword_search(conn, dataset.name, query, limit=1, ranked=False)
            exact, _ = db.build_query(conn, dataset.name, term, fuzzy=False)
            print(
                f"exact-only tsquery: {exact}"
                f"\n{'fuzzy finds matches' if hits else 'fuzzy finds nothing'}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset", nargs="?", default="flickr1m", choices=sorted(datasets.ALL_DATASETS)
    )
    parser.add_argument(
        "--show", metavar="WORD",
        help="do not rebuild; print what WORD would expand to and why",
    )
    args = parser.parse_args()
    dataset = datasets.ALL_DATASETS[args.dataset]
    if args.show:
        show(dataset, args.show)
    else:
        build(dataset)


if __name__ == "__main__":
    main()
