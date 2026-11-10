"""Postgres connection, schema, fuzzy matching, and application queries."""

import os
import re

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row, tuple_row
from psycopg_pool import ConnectionPool

load_dotenv()


# --- connection --------------------------------------------------------------

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://cbir:cbir@localhost:5432/cbir"
)

#: Seconds to wait for the TCP connection and handshake. Without a limit, a
#: stopped Docker container is worse than a missing one: the Desktop port proxy
#: still accepts the connection and then never answers, so the server hangs at
#: "Waiting for application startup" with nothing to go on. This turns that into
#: a prompt, explicable failure. It bounds connection setup only, not queries.
CONNECT_TIMEOUT = int(os.environ.get("DB_CONNECT_TIMEOUT", "10"))


def connect():
    """Open a connection; rows come back as dicts."""
    return psycopg.connect(
        DATABASE_URL, row_factory=dict_row, connect_timeout=CONNECT_TIMEOUT
    )


#: The server's connection pool, opened in its lifespan. Opening a connection per
#: request costs a TCP handshake plus SCRAM authentication through Docker's port
#: proxy on every query; a pool pays that once. The scripts keep `connect()`.
_pool = None


def open_pool(size=8):
    """Open the server's pool, failing fast like `connect()` if Postgres is down."""
    global _pool
    _pool = ConnectionPool(
        DATABASE_URL,
        min_size=1,
        max_size=size,
        kwargs={"row_factory": dict_row, "connect_timeout": CONNECT_TIMEOUT},
        open=False,
    )
    _pool.open(wait=True, timeout=CONNECT_TIMEOUT)


def close_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def connection():
    """A pooled connection when the server opened a pool, else a fresh one.

    Either way it is a context manager that commits on success and rolls back on
    an exception, so callers do not need to know which they got.
    """
    return _pool.connection() if _pool is not None else connect()


def _redacted_url():
    """DATABASE_URL without its password, safe to print in an error message."""
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", DATABASE_URL)


def unreachable_message(exc):
    """What to tell the user when Postgres cannot be reached."""
    detail = str(exc).strip().splitlines()
    return (
        "\n"
        "  Cannot reach Postgres.\n"
        f"    tried:  {_redacted_url()}\n"
        f"    error:  {detail[0] if detail else exc!r}\n"
        "\n"
        "  Start the database, then try again:\n"
        "      docker compose up -d db\n"
        "\n"
        "  Or let the setup script do everything:\n"
        "      python setup.py\n"
    )


# --- schema ------------------------------------------------------------------

#: Bumped whenever KEYWORDS_EXPRESSION changes, so init_schema() knows to rebuild
#: the generated column instead of silently leaving a stale one in place.
SCHEMA_VERSION = 2

#: The generated full-text column. Two deliberate choices:
#:
#: * Derived visual words are NOT in here. They are reachable through their own
#:   facet controls and the tag bitmask, because 23 of the 25 also occur as real
#:   user tags (see app/visualtags.py), and one word cannot mean both.
#: * The filename's extension is stripped before tokenising. Otherwise every
#:   single row carries the token "jpg", which matched all 805 842 images and was
#:   by far the commonest "word" in the index.
KEYWORDS_EXPRESSION = """
    to_tsvector(
        'simple',
        class_label
        || ' ' || regexp_replace(
                      regexp_replace(filename, '\\.[^.]*$', ''), '[_.-]+', ' ', 'g')
        || ' ' || user_tags
    )
"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS images (
    id          bigserial PRIMARY KEY,
    dataset     text NOT NULL,
    url         text NOT NULL,
    filename    text NOT NULL,
    class_label text NOT NULL,
    -- Visual words the indexer measured from the pixels. Display and bitmask
    -- only; a base with words of its own in its filenames leaves this empty.
    tags        text NOT NULL DEFAULT '',
    -- Real tags people typed, loaded by scripts/import_mirflickr.py. This is
    -- what the full-text index searches, along with class and filename --
    -- separators in the filename are blanked out first, since Postgres' parser
    -- would otherwise keep "circle_blue_06.png" as one token instead of three.
    user_tags   text NOT NULL DEFAULT '',
    keywords    tsvector GENERATED ALWAYS AS ({KEYWORDS_EXPRESSION}) STORED
);
CREATE INDEX IF NOT EXISTS images_keywords_idx ON images USING GIN (keywords);
CREATE INDEX IF NOT EXISTS images_dataset_idx ON images (dataset);
"""

META = """
CREATE TABLE IF NOT EXISTS cbir_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);
"""

#: The searchable vocabulary of one base: every lexeme in the full-text index,
#: with how many images carry it. See the fuzzy-matching section below for why it
#: exists.
TAG_VOCAB = """
CREATE TABLE IF NOT EXISTS tag_vocab (
    dataset text    NOT NULL,
    word    text    NOT NULL,
    ndoc    integer NOT NULL,
    PRIMARY KEY (dataset, word)
);
"""

#: Trigram index, which serves both `word %% term` (similarity) and
#: `word LIKE 'term%%'` (prefix). One index answers both kinds of near-miss.
TAG_VOCAB_INDEX = """
CREATE INDEX IF NOT EXISTS tag_vocab_trgm_idx
    ON tag_vocab USING GIN (word gin_trgm_ops);
"""

#: Columns that cannot be added to an existing table without losing its rows.
CORE_COLUMNS = frozenset({"id", "dataset", "url", "filename", "class_label", "tags"})
#: Columns that can be back-filled, so a table missing them is migrated instead.
ADDABLE_COLUMNS = {"user_tags": "text NOT NULL DEFAULT ''"}


def _rebuild_keywords(cur):
    """Recreate the generated full-text column and its index.

    This rewrites the table -- a couple of minutes on 800k rows -- but it is the
    only way to change a generated column, and far better than dropping the rows
    and re-running a descriptor pass over the images.
    """
    cur.execute("DROP INDEX IF EXISTS images_keywords_idx")
    cur.execute("ALTER TABLE images DROP COLUMN IF EXISTS keywords")
    cur.execute(
        "ALTER TABLE images ADD COLUMN keywords tsvector "
        f"GENERATED ALWAYS AS ({KEYWORDS_EXPRESSION}) STORED"
    )
    cur.execute("CREATE INDEX images_keywords_idx ON images USING GIN (keywords)")
    cur.execute("ANALYZE images")


def init_schema(reset=False):
    """Create the tables, or bring an older set up to date.

    A table missing one of the core columns cannot be migrated and is rebuilt;
    nothing is lost, since the indexer is the only thing that writes rows. Missing
    tag columns, or a stale keywords expression, are migrated in place so that
    805k rows and an 8-minute descriptor pass survive.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(META)
        if reset:
            cur.execute("DROP TABLE IF EXISTS images")
            cur.execute("DELETE FROM cbir_meta WHERE key = 'schema_version'")
        else:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'images'"
            )
            existing = {row["column_name"] for row in cur.fetchall()}
            cur.execute("SELECT value FROM cbir_meta WHERE key = 'schema_version'")
            found = cur.fetchone()
            version = int(found["value"]) if found else 0

            if existing and not CORE_COLUMNS <= existing:
                print("images table predates the dataset column; rebuilding it")
                cur.execute("DROP TABLE images")
                version = 0
            elif existing:
                missing = [c for c in ADDABLE_COLUMNS if c not in existing]
                for column in missing:
                    print(f"migrating images: adding {column}")
                    cur.execute(
                        f"ALTER TABLE images ADD COLUMN {column} {ADDABLE_COLUMNS[column]}"
                    )
                if missing or version < SCHEMA_VERSION:
                    print(
                        "rebuilding the keywords column "
                        f"(schema v{version} -> v{SCHEMA_VERSION})"
                    )
                    _rebuild_keywords(cur)

        cur.execute(SCHEMA)
        cur.execute(TAG_VOCAB)
        cur.execute(
            "INSERT INTO cbir_meta (key, value) VALUES ('schema_version', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )


# --- fuzzy matching ----------------------------------------------------------
#
# `to_tsvector('simple')` does not stem, which is deliberate -- what goes in is
# what has to be typed -- but it makes the text search unforgiving: `vien` found
# 7 images and missed Vienna's 1 353. So each typed word is expanded to the words
# it plausibly meant.
#
# The expansion happens in the *vocabulary*, not over the images: `tag_vocab`
# holds every word the index can match, so one typed word is compared against
# 295 177 known words rather than 805 842 documents. That is what makes it cost
# 4-12 ms and lets candidates be ranked by how common they actually are.
#
# The constants below are the tuning, and each one is there because the obvious
# version of this was wrong in a specific way.

#: Lexemes appearing in only one image are left out of the vocabulary. 439 153 of
#: the Flickr base's 734 330 are one-offs -- somebody's private string, almost
#: never the word a searcher meant -- and excluding them halves the index while
#: improving what it suggests. They stay searchable: the word actually typed is
#: always included in the query, whether or not it is in the vocabulary.
VOCAB_MIN_NDOC = 2

#: Filename tokens (`im1`, `im805842`) are in the full-text index so an image can
#: be found by name, but they are not words and must never be suggested.
VOCAB_SKIP = r"^im[0-9]+$"

#: Below this many characters a word is not expanded: trigram similarity stops
#: discriminating, and a one- or two-letter prefix matches a large part of the
#: vocabulary.
MIN_FUZZY_LENGTH = 3

#: How many near-misses one typed word may turn into.
EXPANSIONS_PER_TERM = 5

#: Trigram similarity a non-prefix candidate must reach. 0.35 keeps
#: `mountian`->`mountain` (0.38) and `archtecture`->`architecture` (0.67) while
#: dropping most unrelated words that merely share a few trigrams.
SIMILARITY_FLOOR = 0.35

#: Added to the similarity of a candidate the typed word is a prefix of, which is
#: the commonest near-miss: a word typed only part-way (`vien` for `vienna`).
PREFIX_BONUS = 0.25

#: How much a candidate's frequency counts. Ranking on similarity alone buries
#: the intended word: for `dogg`, `doggystyle` (7 images) outscores `dog`
#: (8 939). Weighting by log frequency fixes that. The coefficient is deliberately
#: small -- at 1.0 a very common but weakly similar word wins on frequency alone.
FREQUENCY_WEIGHT = 0.3

#: A candidate is only offered if it is at least this share as common as the word
#: actually typed. Without it, expansion damages the queries that were already
#: right: `flower` (22 214 images) picked up `flowerwatcher`, `flowerotica` and
#: `floweria`, adding nothing but noise. With it, a confident word expands only to
#: comparably common relatives (`flower` -> `flowers`) while a word that is barely
#: in the base -- `vien`, 7 images -- still reaches everything plausible, because
#: a fifth of 7 excludes nothing. A word absent from the vocabulary counts as 0
#: and is not filtered at all, which is the pure-typo case.
MIN_RELATIVE_FREQUENCY = 0.2


def has_trigrams(conn):
    """Whether pg_trgm is installed, so fuzzy matching can work at all."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
        return cur.fetchone() is not None


def ensure_trigrams(conn):
    """Install pg_trgm if it is not there yet; report rather than fail if not.

    Creating an extension needs privileges the owner of a managed database may
    not have. Fuzzy matching is a feature, not a requirement, so a refusal is
    reported once and exact search carries on. The failed statement aborts the
    transaction, hence the rollback.
    """
    if has_trigrams(conn):
        return True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION pg_trgm")
        conn.commit()
        return True
    except psycopg.Error as exc:
        conn.rollback()
        print(
            f"note: could not install the pg_trgm extension ({exc.__class__.__name__}), "
            "so fuzzy matching is off and keyword search needs exact words.\n"
            "      Install it once as a superuser:  CREATE EXTENSION pg_trgm;"
        )
        return False


def build_tag_vocab(conn, dataset):
    """(Re)build one dataset's fuzzy-matching vocabulary; returns how many words.

    Read straight out of the tsvector column with `ts_stat`, so the vocabulary is
    exactly the set of words the full-text search can match and cannot drift from
    the index it describes. Costs one pass over the tsvectors.

    Rebuilt by `build_index.py` and by `import_mirflickr.py --tags`, since both
    change which words exist.
    """
    with conn.cursor() as cur:
        cur.execute(TAG_VOCAB)
    conn.commit()

    if not ensure_trigrams(conn):
        return 0

    with conn.cursor() as cur:
        cur.execute("DELETE FROM tag_vocab WHERE dataset = %s", (dataset,))
        cur.execute(
            """
            INSERT INTO tag_vocab (dataset, word, ndoc)
            SELECT %s, word, ndoc
            -- ts_stat takes its inner query as a string, so the dataset name is
            -- quoted into it by Postgres' own format(). %%L is doubled because
            -- psycopg would otherwise read it as a parameter placeholder, and
            -- the ::text cast is required because format() takes "any", which
            -- leaves the parameter's type otherwise uninferable.
            FROM ts_stat(
                format('SELECT keywords FROM images WHERE dataset = %%L', %s::text)
            )
            WHERE ndoc >= %s AND word !~ %s
            """,
            (dataset, dataset, VOCAB_MIN_NDOC, VOCAB_SKIP),
        )
        written = cur.rowcount
        cur.execute(TAG_VOCAB_INDEX)
        cur.execute("ANALYZE tag_vocab")
    conn.commit()
    return written


def vocab_size(conn, dataset):
    """How many words the fuzzy vocabulary holds for this base, 0 if none.

    The table itself is guaranteed: `init_schema` creates it, and every caller
    runs that first -- the server in its lifespan, the scripts on entry.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM tag_vocab WHERE dataset = %s", (dataset,))
        return cur.fetchone()["n"]


def _like_prefix(term):
    """`term` as a LIKE pattern matching anything starting with it.

    The wildcards have to be escaped or a typed `%` would match the whole
    vocabulary and a typed `_` any character.
    """
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


#: Every near-miss candidate of one typed word, with its score and whether it is
#: too rare beside the word typed. `expand_term` keeps the best few that are not;
#: `scripts/build_tagvocab.py --show` prints all of them, so the two cannot drift.
EXPANSION_CANDIDATES = """
    WITH typed AS (
        SELECT coalesce(max(ndoc), 0) AS ndoc
        FROM tag_vocab
        WHERE dataset = %(dataset)s AND word = %(term)s
    ),
    candidate AS (
        SELECT word, ndoc,
               word LIKE %(prefix)s ESCAPE '\\' AS is_prefix,
               similarity(word, %(term)s)       AS sim
        FROM tag_vocab
        WHERE dataset = %(dataset)s
          AND (word LIKE %(prefix)s ESCAPE '\\' OR word %% %(term)s)
    )
    SELECT candidate.word, candidate.ndoc, candidate.is_prefix, candidate.sim,
           typed.ndoc AS typed_ndoc,
           (candidate.sim
            + CASE WHEN candidate.is_prefix THEN %(bonus)s ELSE 0 END)
           * (1 + %(weight)s * ln(candidate.ndoc + 2)) AS score,
           (candidate.word <> %(term)s
            AND candidate.ndoc < typed.ndoc * %(relative)s) AS too_rare
    FROM candidate, typed
    WHERE candidate.is_prefix OR candidate.sim >= %(floor)s
"""


def expansion_params(dataset, term):
    """The parameters EXPANSION_CANDIDATES takes, from the tuning constants above."""
    return {
        "term": term,
        "prefix": _like_prefix(term),
        "dataset": dataset,
        "floor": SIMILARITY_FLOOR,
        "bonus": PREFIX_BONUS,
        "weight": FREQUENCY_WEIGHT,
        "relative": MIN_RELATIVE_FREQUENCY,
    }


def expand_term(conn, dataset, term, limit=EXPANSIONS_PER_TERM):
    """The vocabulary words a typed word plausibly meant, best first.

    Two kinds of near-miss, answered by the one trigram index:

    * a prefix -- `vien` for `vienna`, by far the commonest case, and a strong
      enough signal to earn PREFIX_BONUS;
    * a typo anywhere -- `archtecture` for `architecture`, caught by trigram
      similarity above SIMILARITY_FLOOR.

    Candidates are ranked by similarity *and* by how common they are, then
    anything much rarer than the word typed is dropped. See FREQUENCY_WEIGHT and
    MIN_RELATIVE_FREQUENCY for why both halves of that are needed.
    """
    if len(term) < MIN_FUZZY_LENGTH:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT word FROM ({EXPANSION_CANDIDATES}) AS scored
            WHERE NOT too_rare
            ORDER BY score DESC, ndoc DESC, word
            LIMIT %(limit)s
            """,
            {**expansion_params(dataset, term), "limit": limit},
        )
        return [row["word"] for row in cur.fetchall()]


# --- queries -----------------------------------------------------------------

#: Ceiling on how many keyword matches are handed to the visual re-ranking. No
#: single-word query reaches it -- the commonest real tag, `nikon`, is 37 549
#: images -- but a query that did would be silently ranking part of its own match
#: set, so app/main.py reports when the ceiling is hit.
CANDIDATE_LIMIT = 50_000


def _quote_lexeme(word):
    """A lexeme as a quoted tsquery term.

    Inside a tsquery's single quotes a backslash escapes and a single quote is
    doubled, so both have to be handled -- real Flickr tags contain apostrophes
    and the vocabulary is UTF-8 throughout.
    """
    return "'" + word.replace("\\", "\\\\").replace("'", "''") + "'"


def lexemes_of(conn, text):
    """Tokenise text exactly as the index did, by asking Postgres.

    Splitting on whitespace here would drift from `to_tsvector`: the generated
    column decides what a word is, so the same function has to decide what the
    query's words are.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT lexeme FROM unnest(to_tsvector('simple', %s)) ORDER BY lexeme",
            (text,),
        )
        return [row["lexeme"] for row in cur.fetchall()]


def build_query(conn, dataset, text, fuzzy=False):
    """Turn typed text into a tsquery: (query string, expansions).

    Words are ANDed, as `plainto_tsquery` would do, but each one may become an
    OR-group of what it plausibly meant. The word actually typed is always in its
    own group, so turning fuzzy matching on can only ever add results -- never
    take away the exact match, even for a word too rare to be in the vocabulary.

    `expansions` is [(typed word, [words searched])] for the ones that grew, so
    the client can show what it did instead of silently searching something else.
    """
    terms = lexemes_of(conn, text)
    if not terms:
        return None, []

    groups, expansions = [], []
    for term in terms:
        words = [term]
        if fuzzy:
            words += [w for w in expand_term(conn, dataset, term) if w != term]
            if len(words) > 1:
                expansions.append((term, words))
        groups.append("(" + " | ".join(_quote_lexeme(w) for w in words) + ")")

    return " & ".join(groups), expansions


def keyword_search(conn, dataset, query, limit=None, ranked=True):
    """Full-text search inside one dataset, for a tsquery from `build_query`.

    `ranked` orders by text relevance, which makes Postgres score every match
    before it can pick the best few. That is worth paying for here because real
    Flickr tags are Zipf-distributed, so the worst case is the 37 549 rows of
    `nikon`. It was switched off while the derived visual words were the only
    text: each of those matched a third of the base and they all scored alike, so
    ranking meant scoring 600 000 rows to choose 24. A base sets this through
    `Dataset.rank_keywords`.
    """
    if not query:
        return []
    order = "ts_rank(keywords, q) DESC, id" if ranked else "id"
    sql = f"""
        SELECT id, url, class_label, tags,
               {"ts_rank(keywords, q)" if ranked else "0"} AS rank
        FROM images, to_tsquery('simple', %s) AS q
        WHERE dataset = %s AND keywords @@ q
        ORDER BY {order}
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (query, dataset, CANDIDATE_LIMIT if limit is None else limit))
        return cur.fetchall()


def keyword_ids(conn, dataset, query):
    """Ids of the images matching a tsquery, as an int64 array, at most CANDIDATE_LIMIT.

    This is the candidate filter in front of a visual ranking, which needs nothing
    but the ids. Selecting the whole row there meant carrying every match's URL and
    tags back as a dict -- 22 214 of them for `flower` -- only to keep the id.
    """
    if not query:
        return np.zeros(0, dtype=np.int64)
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            "SELECT id FROM images WHERE dataset = %s "
            "AND keywords @@ to_tsquery('simple', %s) ORDER BY id LIMIT %s",
            (dataset, query, CANDIDATE_LIMIT),
        )
        return np.fromiter((row[0] for row in cur), dtype=np.int64)


def get_image(conn, image_id):
    """One image's row, or None. Carries `dataset` so the caller can check it."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, dataset, url, class_label, tags FROM images WHERE id = %s",
            (image_id,),
        )
        return cur.fetchone()


def get_images(conn, ids):
    """{id: row} for the given ids, for attaching fields to a list of hits."""
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, dataset, url, class_label, tags FROM images "
            "WHERE id = ANY(%s)",
            (list(ids),),
        )
        return {row["id"]: row for row in cur.fetchall()}


def count(conn, dataset):
    """How many images one base has."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM images WHERE dataset = %s", (dataset,))
        return cur.fetchone()["n"]


def browse(conn, dataset, limit, offset=0):
    """A page of a dataset, so the UI has something to click on to start a search."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, url, class_label, tags FROM images WHERE dataset = %s "
            "ORDER BY id LIMIT %s OFFSET %s",
            (dataset, limit, offset),
        )
        return cur.fetchall()
