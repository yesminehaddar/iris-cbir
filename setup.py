"""One-command setup: dependencies, database, index, self-test.

    python setup.py                 # everything, full 1M base (~12 min)
    python setup.py --quick         # the first 20 000 thumbnails instead (~1 min)
    python setup.py --serve         # ... and start the server when it is done
    python setup.py --rebuild       # redo the index even if one is already there

This exists because the pipeline has an order dependency that is easy to get
wrong and silent when you do: `build_index.py flickr1m` replaces that dataset's
rows and renumbers the ids the feature matrices are keyed to, so the four
`import_mirflickr.py` commands have to run *after* it. Forget them and the base
still works by image, while every text search quietly returns nothing.

Nothing here is magic -- it runs the same commands docs/DESIGN.md documents, in the
right order, skipping the ones already done. Only the standard library is used,
since it has to run before the dependencies are installed.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)

#: The distribution's archives, in the order they must be imported, with the
#: descriptor name each one is stored under. `binary` marks the GIST archive,
#: whose entries are raw little-endian float32 rather than one value per line.
ARCHIVES = (
    ("tags", r"^tags\.zip$", None, False),
    ("features", r"^features_edgehistogram.*\.zip$", "mp7_ehd_ref", False),
    ("features", r"^features_homogeneoustexture.*\.zip$", "mp7_htd_ref", False),
    ("features", r"^features_gist.*\.zip$", "gist", True),
)


# --- small helpers -----------------------------------------------------------

class Abort(Exception):
    """A failure with an explanation already written for the user."""


def say(message=""):
    print(message, flush=True)


def step(number, total, title):
    say(f"\n[{number}/{total}] {title}")
    say("-" * (len(title) + 8))


def run(command, what, cwd=ROOT, check=True):
    """Run a command, streaming its output; raise Abort with context on failure."""
    say(f"  $ {' '.join(str(c) for c in command)}")
    result = subprocess.run(command, cwd=cwd)
    if check and result.returncode != 0:
        raise Abort(f"{what} failed (exit {result.returncode})")
    return result.returncode


def capture(command, cwd=ROOT):
    """Run a command quietly; return (returncode, stdout+stderr)."""
    done = subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return done.returncode, (done.stdout or "").strip()


# --- the steps ---------------------------------------------------------------

def check_python():
    if sys.version_info < MIN_PYTHON:
        raise Abort(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required; "
            f"this is {sys.version.split()[0]}."
        )
    say(f"  Python {sys.version.split()[0]} at {sys.executable}")


def install_dependencies():
    run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "-r", "requirements.txt"],
        "installing the dependencies",
    )


def write_env():
    env, example = ROOT / ".env", ROOT / ".env.example"
    if env.exists():
        say("  .env already exists, left alone")
    elif example.exists():
        shutil.copyfile(example, env)
        say("  .env created from .env.example")
    else:
        raise Abort("neither .env nor .env.example is present")


def compose_command():
    """`docker compose` on modern Docker, `docker-compose` on the old one."""
    if shutil.which("docker"):
        code, _ = capture(["docker", "compose", "version"])
        if code == 0:
            return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise Abort(
        "Docker was not found on the PATH.\n"
        "  It is only needed to run Postgres. Install Docker Desktop and start\n"
        "  it, then run this script again.\n"
        "  Already have a Postgres of your own? Put its URL in .env as\n"
        "  DATABASE_URL=... and re-run with --skip-docker."
    )


def start_database(timeout=120):
    """Bring the Postgres container up and wait until it really accepts queries."""
    compose = compose_command()
    code, output = capture(compose + ["ps", "-q", "db"])
    if code != 0:
        raise Abort(
            "Docker is installed but not responding. Start Docker Desktop and\n"
            f"  wait for it to finish starting, then try again.\n  ({output})"
        )

    run(compose + ["up", "-d", "db"], "starting Postgres")

    # The container reports "Up" a few seconds before the server accepts
    # connections, so poll pg_isready rather than trusting the container state.
    say("  waiting for Postgres to accept connections")
    deadline = time.time() + timeout
    while time.time() < deadline:
        code, _ = capture(compose + ["exec", "-T", "db", "pg_isready", "-U", "cbir", "-d", "cbir"])
        if code == 0:
            say("  Postgres is ready")
            return
        time.sleep(1.5)
    raise Abort(
        f"Postgres did not become ready within {timeout}s.\n"
        f"  Look at why with:  {' '.join(compose)} logs db"
    )


def find_archives(extra_dir=None):
    """Locate the distribution's four zips. Returns {pattern: path}."""
    places = [ROOT.parent, ROOT]
    if extra_dir:
        places.insert(0, Path(extra_dir).expanduser().resolve())

    found = {}
    for place in places:
        if not place.is_dir():
            continue
        for entry in place.iterdir():
            if not entry.is_file():
                continue
            for _, pattern, _, _ in ARCHIVES:
                if pattern not in found and re.match(pattern, entry.name, re.IGNORECASE):
                    found[pattern] = entry
    return found


def dataset_state():
    """What is already built, asked of the app's own modules and the database.

    Imported lazily: these modules need the dependencies, which are installed by
    an earlier step of this very script.
    """
    sys.path.insert(0, str(ROOT))
    from app import datasets, db

    flickr = datasets.FLICKR
    thumbnails = flickr.images_dir
    has_thumbnails = thumbnails.is_dir() and any(thumbnails.glob("*/*.jpg"))

    state = {
        "thumbnails": thumbnails,
        "has_thumbnails": has_thumbnails,
        "flickr_indexed": (flickr.features_dir / "ids.npy").exists(),
        "features_present": {
            name
            for name in ("mp7_ehd_ref", "mp7_htd_ref", "gist")
            if (flickr.features_dir / f"{name}.npy").exists()
        },
        "tagged": 0,
        "vocab": {},
        "rows": {},
    }

    db.init_schema()
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM images "
                "WHERE dataset = %s AND user_tags <> ''",
                (flickr.name,),
            )
            state["tagged"] = cur.fetchone()["n"]
        for name in datasets.DATASETS:
            state["rows"][name] = db.count(conn, name)
            state["vocab"][name] = db.vocab_size(conn, name)
    return state


def build_flickr(state, rebuild, limit):
    """The 1M thumbnail base, then the distribution's own tags and features.

    Returns True when the index was (re)built, which is what makes the imports
    mandatory: a rebuild renumbers the ids the imported matrices are keyed to.
    """
    if not state["has_thumbnails"]:
        say(f"  no thumbnails found at {state['thumbnails']}")
        say("  Nothing can be indexed without them. Put the distribution's")
        say("  `thumbnails/` directory there and run this script again.")
        return False

    if state["flickr_indexed"] and not rebuild:
        if state["rows"].get("flickr1m"):
            say("  already indexed, skipping (--rebuild to redo it)")
            return False
        # The index came with the project but the database did not -- it lives
        # in a Docker volume on whichever machine built it. The rows can be put
        # back from the index without recomputing a single descriptor.
        say("  index found on disk but the database is empty; restoring the rows")
        run(
            [sys.executable, "scripts/build_index.py", "flickr1m", "--restore-rows"],
            "restoring the database rows",
        )
        return False

    command = [sys.executable, "scripts/build_index.py", "flickr1m"]
    if limit:
        command += ["--limit", str(limit)]
        say(f"  quick mode: indexing the first {limit:,} thumbnails")
    else:
        say("  indexing the full base -- this is the slow step, around 8-12 minutes")
    run(command, "indexing the 1M base")
    return True


def import_archives(state, archives, rebuilt):
    """Load the distribution's tag and feature archives.

    Always run after a rebuild: `build_index.py` replaces the rows, which clears
    the imported tags and invalidates the ids the feature matrices line up with.
    Nothing warns you if this is skipped -- the tags simply come back empty.
    """
    if not archives:
        say("  none of the distribution's archives were found, so there are no")
        say("  real tags and no provided descriptors. The base still works by")
        say("  image and by visual property. Looked in:")
        say(f"    {ROOT.parent}")
        say(f"    {ROOT}")
        say("  Point at them with:  python setup.py --data <directory>")
        return

    for kind, pattern, name, binary in ARCHIVES:
        path = archives.get(pattern)
        if path is None:
            say(f"  (not found: {pattern}) -- skipped")
            continue

        # Skip only what is demonstrably already loaded, and never after a
        # rebuild, which invalidates all of it.
        if not rebuilt:
            if kind == "tags" and state["tagged"]:
                say(f"  tags already loaded ({state['tagged']:,} images), skipping")
                continue
            if kind == "features" and name in state["features_present"]:
                say(f"  {name} already present, skipping")
                continue

        command = [sys.executable, "scripts/import_mirflickr.py"]
        if kind == "tags":
            command += ["--tags", str(path)]
        else:
            command += ["--features", str(path), "--name", name]
            if binary:
                command.append("--binary")
        run(command, f"importing {path.name}")


def build_vocabulary(state):
    """Build the fuzzy-matching vocabulary for any base that has rows but none.

    build_index.py and import_mirflickr.py both build it as they go, so this
    only has work to do for an index built before fuzzy matching existed -- and
    it does that without redoing the descriptor pass.
    """
    todo = [
        name
        for name, rows in state["rows"].items()
        if rows and not state["vocab"].get(name)
    ]
    if not todo:
        built = ", ".join(
            f"{name} {words:,} words"
            for name, words in state["vocab"].items()
            if words
        )
        say(f"  already built ({built or 'nothing to build'})")
        return
    for name in todo:
        run(
            [sys.executable, "scripts/build_tagvocab.py", name],
            f"building the vocabulary for {name}",
            check=False,
        )


def self_test(quick=True):
    command = [sys.executable, "scripts/selftest.py", "flickr1m"]
    if quick:
        command.append("--quick")
    # A base that was never built reports it and exits non-zero; that is
    # information, not a reason to fail the whole setup.
    run(command, "self-test", check=False)


def serve():
    say("\nStarting the server. Open http://127.0.0.1:8000  (Ctrl+C to stop)\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"], cwd=ROOT
    )


# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quick", nargs="?", type=int, const=20_000, metavar="N",
        help="index only the first N thumbnails (default 20000) instead of all 805k",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="rebuild the index even if one is already on disk",
    )
    parser.add_argument(
        "--data", metavar="DIR",
        help="where the distribution's zips are, if not beside the project",
    )
    parser.add_argument(
        "--skip-docker", action="store_true",
        help="do not touch Docker; use the DATABASE_URL already in .env",
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="do not run pip install"
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="do not run the self-test at the end"
    )
    parser.add_argument(
        "--serve", action="store_true", help="start the server when setup finishes"
    )
    args = parser.parse_args()

    total = 7
    started = time.time()
    say("=" * 64)
    say("  CBIR -- setup")
    say("=" * 64)

    step(1, total, "Checking Python")
    check_python()

    step(2, total, "Installing dependencies")
    if args.skip_deps:
        say("  skipped (--skip-deps)")
    else:
        install_dependencies()

    step(3, total, "Configuration")
    write_env()

    step(4, total, "Starting Postgres")
    if args.skip_docker:
        say("  skipped (--skip-docker); using DATABASE_URL from .env")
    else:
        start_database()

    # Everything below needs the dependencies and a reachable database.
    try:
        import psycopg  # noqa: F401  (imported to recognise its errors below)
    except ImportError:
        raise Abort(
            "the dependencies are not installed.\n"
            "  Re-run without --skip-deps, or install them yourself:\n"
            f"      {sys.executable} -m pip install -r requirements.txt"
        )

    try:
        state = dataset_state()
    except psycopg.OperationalError as exc:
        sys.path.insert(0, str(ROOT))
        from app import db
        raise Abort(f"could not reach the database.{db.unreachable_message(exc)}")

    archives = find_archives(args.data)

    step(5, total, "Flickr 1M base")
    rebuilt = build_flickr(state, args.rebuild, args.quick)
    if state["has_thumbnails"] and (rebuilt or state["flickr_indexed"]):
        say("")
        say("  Importing the distribution's tags and provided descriptors")
        import_archives(state, archives, rebuilt)

    step(6, total, "Fuzzy-matching vocabulary")
    build_vocabulary(dataset_state())

    step(7, total, "Self-test")
    if args.skip_tests:
        say("  skipped (--skip-tests)")
    else:
        self_test()

    minutes, seconds = divmod(int(time.time() - started), 60)
    say("\n" + "=" * 64)
    say(f"  Setup finished in {minutes}m {seconds}s")
    say("=" * 64)
    if args.serve:
        serve()
    else:
        say("\nStart the server with:")
        say("    python -m uvicorn app.main:app --port 8000")
        say("\nThen open http://127.0.0.1:8000")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Abort as exc:
        say(f"\nSETUP FAILED: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        say("\ninterrupted")
        sys.exit(130)
