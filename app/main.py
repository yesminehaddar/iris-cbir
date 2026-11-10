"""FastAPI service for browsing images and running CBIR queries."""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
import psycopg

from app import datasets, db
from app.descriptors import (
    DESCRIPTOR_META,
    IMPORTED_DESCRIPTORS,
    compute,
    load_image_bytes,
)
from app.search import StoreSet
from app.visualtags import facets_for

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_RESULTS = 200

stores = StoreSet(datasets.DATASETS.values())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bring the schema up to date and memory-map the feature matrices, once.

    Loading here rather than per request is what makes startup instant and a
    query cheap: `mmap_mode="r"` means the 3 GB of matrices are never read in
    full, and the page cache decides what stays resident.
    """
    # An unreachable database is the usual way a fresh checkout fails, and
    # psycopg's own message does not mention that a container has to be running.
    try:
        db.init_schema()
    except psycopg.OperationalError as exc:
        print(db.unreachable_message(exc), file=sys.stderr, flush=True)
        raise
    db.open_pool()
    stores.load()
    if not any(stores[ds].size for ds in datasets.DATASETS):
        print(
            "\n  No feature index found. The server will start, but the base is\n"
            "  empty until you build one:\n"
            "      python setup.py          (does everything)\n"
            "      python scripts/build_index.py flickr1m\n",
            file=sys.stderr,
            flush=True,
        )
    yield
    db.close_pool()


app = FastAPI(title="CBIR", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# One mount per base, matching the url_prefix the indexer stored in the rows.
for _dataset in datasets.DATASETS.values():
    if _dataset.images_dir.is_dir():
        app.mount(
            _dataset.url_prefix,
            StaticFiles(directory=_dataset.images_dir),
            name=f"images-{_dataset.name}",
        )


class SearchRequest(BaseModel):
    """The body of POST /api/search.

    Every criterion is optional, but at least one of `keywords`, `visual` and
    `image_id` must be given. They compose rather than compete: the first two
    narrow the base, the third orders what is left. The multipart form of
    POST /api/search/image takes the same fields plus a file, minus `image_id`.
    """

    dataset: str = datasets.DEFAULT_DATASET
    #: Free text, matched against the full-text index. An open vocabulary.
    keywords: str | None = None
    #: Chosen visual-property words (`dark`, `vivid`, `blue`, …). Separate from
    #: `keywords` on purpose: see app/visualtags.py.
    visual: list[str] = []
    #: An image already in the base, to rank by similarity to.
    image_id: int | None = None
    #: Which descriptors to compare with; empty means all of the base's.
    descriptors: list[str] = []
    #: How many results, capped at MAX_RESULTS.
    k: int = 24
    #: "vptree" to use the metric tree, anything else for the scan. See
    #: FeatureStore.search: the scan is the default because it is what wins here.
    index: str = "auto"
    #: Expand each typed word to the vocabulary words it plausibly meant, so
    #: `vien` also searches `vienna`. Additive: the exact word is always kept.
    fuzzy: bool = True


def _dataset_or_404(name):
    dataset = datasets.get(name)
    if dataset is None:
        raise HTTPException(404, f"unknown dataset {name!r}")
    return dataset


def _add_note(stats, note):
    """Append a note to the index stats, keeping any note already there.

    Several things can want to explain themselves about one query — the VP-tree
    falling back to the scan, a filter ceiling, a dropped descriptor — and the
    last one to write must not silently erase the others.
    """
    if not note:
        return stats
    existing = stats.get("note")
    stats["note"] = f"{existing}; {note}" if existing else note
    return stats


def _row_json(row, score=None):
    out = {
        "id": row["id"],
        "url": row["url"],
        "class_label": row["class_label"],
        "tags": row.get("tags") or "",
    }
    if score is not None:
        out["score"] = score
    return out


@app.get("/")
def index():
    """The single-page client. Everything else it needs it fetches from the API."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/datasets")
def list_datasets():
    """What the client needs to build the form: bases, sizes and descriptors."""
    out = []
    with db.connection() as conn:
        for dataset in datasets.DATASETS.values():
            store = stores[dataset.name]
            out.append(
                {
                    "name": dataset.name,
                    "label": dataset.label,
                    # The index's own row count, which is free. An exact
                    # count(*) of the table was reported here too, and cost
                    # 120 ms of the page's first request to say the same number;
                    # real drift between the two is caught per query instead,
                    # as `stale_rows` in a search's index stats.
                    "indexed": store.size,
                    "keyword_source": dataset.keyword_source,
                    "on_disk": dataset.images_dir.is_dir(),
                    # Words available for near-miss matching; 0 means the
                    # vocabulary was never built, so only exact words match.
                    "fuzzy_words": db.vocab_size(conn, dataset.name),
                    # The closed visual vocabulary, grouped the way the form shows
                    # it. Free-text keywords are a separate, open vocabulary.
                    "facets": facets_for(store.tag_bits),
                    "descriptors": [
                        {
                            "name": name,
                            "dim": store.dimension(name),
                            "vptree": store.has_tree(name),
                            "computable": name not in IMPORTED_DESCRIPTORS,
                            **DESCRIPTOR_META.get(name, {}),
                        }
                        for name in store.descriptor_names
                    ],
                }
            )
    return out


@app.get("/api/browse")
def browse(
    dataset: str = datasets.DEFAULT_DATASET,
    limit: int = Query(24, ge=1, le=MAX_RESULTS),
    offset: int = Query(0, ge=0),
):
    """A page of the base, so the user has images to start a search from."""
    _dataset_or_404(dataset)
    with db.connection() as conn:
        rows = db.browse(conn, dataset, limit, offset)
    return {"query": None, "results": [_row_json(row) for row in rows]}


class Candidates(NamedTuple):
    """The filtered row set a visual search is allowed to rank."""

    rows: object       #: matrix rows, or None for "the whole base"
    how: str | None    #: which structures answered the filter, for the status line
    note: str | None   #: a warning to pass back to the client, if any
    matched: list      #: [(typed word, [words searched])] where fuzzy expanded


def _expansion_note(expansions):
    """Say what the fuzzy matcher searched, so it is never a silent substitution."""
    if not expansions:
        return None
    parts = [f"{term} → {', '.join(words)}" for term, words in expansions]
    return "also searched " + "; ".join(parts)


def _candidates(conn, dataset, store, keywords, visual, fuzzy=True):
    """How the text and facet filters narrow a visual search.

    Two independent filters, each answered by the structure that suits it:

    * the chosen visual words are a closed vocabulary, so the tag bitmask settles
      them in one masked comparison over the whole base (~2 ms, no ceiling);
    * free-text keywords are an open vocabulary of real Flickr tags, which is
      what Postgres' full-text index is for.

    When both are given the answer is their intersection, computed on rows.
    """
    parts = []
    rows = None
    note = None
    expansions = []

    if visual:
        chosen = store.rows_for_words(" ".join(visual))
        if chosen is None:
            unknown = [w for w in visual if w not in store.tag_bits]
            raise HTTPException(
                400,
                f"unknown visual property: {', '.join(unknown) or ', '.join(visual)}",
            )
        rows = chosen
        parts.append("tag-bitmask")

    if keywords:
        query, expansions = db.build_query(conn, dataset, keywords, fuzzy=fuzzy)
        found = db.keyword_ids(conn, dataset, query)
        # The query asks for at most CANDIDATE_LIMIT ids. Hitting that ceiling
        # means the visual ranking only saw part of what matched, so say so
        # rather than presenting a partial answer as a complete one.
        if found.size >= db.CANDIDATE_LIMIT:
            note = (
                f"more than {db.CANDIDATE_LIMIT:,} images match those tags; the "
                "visual ranking only considered the first "
                f"{db.CANDIDATE_LIMIT:,} by id"
            )
        matched = store.rows_of(found)
        rows = matched if rows is None else np.intersect1d(rows, matched)
        parts.append("fuzzy full-text" if expansions else "full-text")

    return Candidates(
        rows, " + ".join(parts) if parts else None, note, expansions
    )


def _rank(conn, store, query_vecs, names, k, candidates, index):
    """Run the visual search and attach the database fields to each hit."""
    hits, stats = store.search(
        query_vecs, names, k, index=index, candidate_rows=candidates.rows
    )
    if candidates.how:
        stats["filter"] = candidates.how
    _add_note(stats, _expansion_note(candidates.matched))
    _add_note(stats, candidates.note)
    rows_by_id = db.get_images(conn, [i for i, _ in hits])
    results = [
        _row_json(rows_by_id[image_id], score)
        for image_id, score in hits
        if image_id in rows_by_id
    ]
    # A hit with no row means features and database have drifted apart, which
    # would otherwise show up only as a result list quietly shorter than k.
    if len(results) != len(hits):
        stats["stale_rows"] = len(hits) - len(results)
    return results, stats


def _requested(store, names):
    """The descriptors to use: those asked for that exist, else all of them."""
    chosen = [n for n in names if n in store.indexes]
    if chosen:
        return chosen
    if not store.descriptor_names:
        raise HTTPException(
            409, f"no features for {store.dataset.name}; run scripts/build_index.py"
        )
    return store.descriptor_names


def _filter_only(conn, dataset, store, keywords, visual, k, started, fuzzy=True):
    """Answer a filter-only query, where there is no query image to rank against.

    The bitmask gives rows, not an ordering, so the first k are returned as they
    fall. With a query image the visual distance would decide the order instead.
    """
    candidates = _candidates(conn, dataset.name, store, keywords, visual, fuzzy)
    rows = candidates.rows
    total = 0 if rows is None else int(rows.size)
    ids = [] if rows is None else store.ids[rows[:k]].tolist()
    found = db.get_images(conn, ids)
    index = {"method": candidates.how or "none", "scanned": 0, "candidates": total}
    _add_note(index, _expansion_note(candidates.matched))
    _add_note(index, candidates.note)
    return {
        "dataset": dataset.name,
        "query": None,
        "results": [_row_json(found[i]) for i in ids if i in found],
        "index": index,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "score": f"{total:,} images match; unranked, pick one to rank by similarity",
    }


@app.post("/api/search")
def search(req: SearchRequest):
    """Search by keywords and/or visual properties, by an indexed image, or by both."""
    dataset = _dataset_or_404(req.dataset)
    store = stores[dataset.name]
    keywords = (req.keywords or "").strip()
    visual = [w for w in req.visual if w]
    k = max(1, min(req.k, MAX_RESULTS))
    started = time.perf_counter()

    if not keywords and not visual and req.image_id is None:
        raise HTTPException(400, "give keywords, visual properties, an image_id, or some combination")

    with db.connection() as conn:
        # No query image: return the filter's own matches, no visual ranking.
        if req.image_id is None:
            if visual:
                return _filter_only(
                    conn, dataset, store, keywords, visual, k, started, req.fuzzy
                )
            query, expansions = db.build_query(
                conn, dataset.name, keywords, fuzzy=req.fuzzy
            )
            matches = db.keyword_search(
                conn, dataset.name, query, limit=k, ranked=dataset.rank_keywords
            )
            # No candidate count here: this path asks Postgres for the top k
            # directly, and counting the rest would double the query's cost for a
            # number the result list already implies.
            index = {
                "method": "fuzzy full-text" if expansions else "full-text",
                "scanned": 0,
            }
            _add_note(index, _expansion_note(expansions))
            return {
                "dataset": dataset.name,
                "query": None,
                "results": [
                    _row_json(row, float(row["rank"]) if dataset.rank_keywords else None)
                    for row in matches
                ],
                "index": index,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "score": (
                    "full-text rank, highest first"
                    if dataset.rank_keywords
                    else "keyword match, unranked"
                ),
            }

        query_row = db.get_image(conn, req.image_id)
        if query_row is None:
            raise HTTPException(404, "unknown image_id")
        if query_row["dataset"] != dataset.name:
            raise HTTPException(
                400, f"image {req.image_id} belongs to {query_row['dataset']!r}"
            )

        query_vecs = store.vectors_of(req.image_id)
        if query_vecs is None:
            raise HTTPException(409, "image has no features; run scripts/build_index.py")

        names = _requested(store, req.descriptors)
        results, stats = _rank(
            conn,
            store,
            query_vecs,
            names,
            k,
            _candidates(conn, dataset.name, store, keywords, visual, req.fuzzy),
            req.index,
        )

    return {
        "dataset": dataset.name,
        "query": _row_json(query_row),
        "descriptors": names,
        "results": results,
        "index": stats,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "score": _score_note(names),
    }


@app.post("/api/search/image")
async def search_by_upload(
    file: UploadFile = File(...),
    dataset: str = Form(datasets.DEFAULT_DATASET),
    keywords: str | None = Form(None),
    visual: list[str] = Form([]),
    descriptors: list[str] = Form([]),
    k: int = Form(24),
    index: str = Form("auto"),
    fuzzy: bool = Form(True),
):
    """Same visual search, but the query is an uploaded image instead of an indexed one."""
    target = _dataset_or_404(dataset)
    store = stores[target.name]
    keywords = (keywords or "").strip()
    visual = [w for w in visual if w]
    k = max(1, min(k, MAX_RESULTS))
    names = _requested(store, descriptors)
    started = time.perf_counter()

    # The provided descriptors were computed on the full-resolution originals,
    # which this base does not ship, so there is no way to compute them for a
    # file the user just handed us. Say so instead of failing on a missing key.
    imported = [n for n in names if n in IMPORTED_DESCRIPTORS]
    names = [n for n in names if n not in IMPORTED_DESCRIPTORS]
    if not names:
        raise HTTPException(
            400,
            f"{', '.join(imported)} can only compare images already in the base: "
            "they come from the distribution's feature archives, not from pixels. "
            "Pick a computed descriptor to search with an uploaded file.",
        )

    data = await file.read()
    try:
        img = load_image_bytes(data)
    except Exception:
        raise HTTPException(400, "could not read that file as an image")

    # The query goes through the same scaling the stored matrix was built with.
    query_vecs = store.apply_scale(compute(img, names))
    with db.connection() as conn:
        results, stats = _rank(
            conn,
            store,
            query_vecs,
            names,
            k,
            _candidates(conn, target.name, store, keywords, visual, fuzzy),
            index,
        )
    if imported:
        _add_note(
            stats,
            f"ignored {', '.join(imported)}: provided descriptors cannot be "
            "computed for an uploaded image",
        )

    return {
        "dataset": target.name,
        "query": {"id": None, "url": None, "filename": file.filename, "tags": ""},
        "descriptors": names,
        "results": results,
        "index": stats,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "score": _score_note(names),
    }


def _score_note(names):
    if len(names) == 1:
        return "Euclidean distance, 0 is identical"
    return "mean of the min-max normalised distances, comparable within this list only"
