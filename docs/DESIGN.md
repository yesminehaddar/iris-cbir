# Iris: content-based indexing and retrieval

A textual/visual search engine over the MIRFLICKR thumbnail base, in the
client/server shape the course asks for: a web client, a FastAPI service
exchanging JSON, images on disk with their URLs in Postgres, and the feature
matrices in a file-backed index.

| base | images | descriptors | text search over |
| --- | --- | --- | --- |
| **Flickr 1M thumbnails** | 805 842 | 5 MPEG-7 computed from the pixels, plus 3 provided with the distribution | 761 603 images' real Flickr tags, and 25 measured visual properties |

Indexing the 1M base takes about 8 minutes on 12 cores. The eight descriptors come
to 3.0 GB of feature matrices; a query over all 805 842 images is 16–27 ms for one
descriptor, 161 ms for the 480-dimensional GIST, and 2 ms to narrow the whole base
by visual properties.

## Run

Needs Python 3.10+ and Docker (for Postgres; nothing else uses it).

```bash
python setup.py             # everything, in order, ~10-15 min
uvicorn app.main:app        # or: python -m uvicorn app.main:app
```

`setup.py --quick` indexes the first 20 000 thumbnails instead of all 805 842,
which takes about a minute and still exercises every path; `--serve` starts the
server when it finishes; `--rebuild` redoes an index that is already there. It is
re-runnable and skips whatever is already done, so a failed step can be fixed and
the script run again. `--help` lists the rest.

<details>
<summary>The same pipeline by hand, which is what <code>setup.py</code> runs</summary>

```bash
docker compose up -d db                    # Postgres
pip install -r requirements.txt
cp .env.example .env                       # Windows: copy .env.example .env

python scripts/build_index.py flickr1m     # ~8 min on 12 cores

# The archives the distribution ships alongside the thumbnails. Read in place,
# straight out of the zips -- no need to unpack four million small files.
python scripts/import_mirflickr.py --tags ../tags.zip
python scripts/import_mirflickr.py --features ../features_edgehistogram.zip --name mp7_ehd_ref
python scripts/import_mirflickr.py --features ../features_homogeneoustexture.zip --name mp7_htd_ref
python scripts/import_mirflickr.py --features ../features_gist.zip --name gist --binary

python scripts/selftest.py flickr1m        # optional: check the index is sane
```

The four imports must come **after** `build_index.py flickr1m`; see the order
dependency noted below. Running `setup.py` is the way not to get that wrong.
</details>

Then open <http://127.0.0.1:8000> and query the base three ways from the same
form:

- **by text** — tags people typed (`flower`, `sunset`, `dog portrait`) go to
  Postgres full-text search;
- **by visual property** — the dropdowns (lightness, saturation, texture, dominant
  hue, dominant edges) are measured from the pixels and answered by a bitmask;
- **by image** — either "browse the base" and click a thumbnail, or choose a file
  under "Query image file". An uploaded image does *not* have to be in the base;
  its descriptors are computed on the fly and compared to the indexed ones.
  Clicking any result re-runs the search with that image, which is how you walk
  through the base;
- **any combination** — tags and properties intersect to a candidate set, then the
  visual ranking happens inside it.

The "Index" selector picks the search strategy, and the status line under the form
reports which one ran, how many distances it computed and how long it took.

### Coming back to it

The index is already built. **What survives a reboot and what does not:**

| | |
| --- | --- |
| Postgres rows, tags and `tag_vocab` | persist — Docker named volume `cbir_pgdata` |
| `data/features/` matrices, bitmask, trees | persist — ordinary files on disk |
| the server | does not; it is just a process |

So picking the project up again is two commands, not the whole pipeline above:

```bash
docker compose up -d db       # if Docker was restarted
uvicorn app.main:app
```

`python scripts/selftest.py flickr1m --quick` confirms in a few seconds that the
index on disk still lines up with the database.

**On another machine** the index comes with the folder but the database does
not — it is in a Docker volume. `python setup.py` notices (index present, no
rows) and runs `python scripts/build_index.py flickr1m --restore-rows`, which
writes the rows back from `ids.npy`, the file listing and `tags.npy` in about
two minutes instead of recomputing descriptors, after checking a spread of rows
against their files. The ids are kept, so the imported descriptors stay valid;
only the tags are re-imported.

**The state as last built and verified:**

- 805 842 thumbnails indexed, 8 descriptors, 3.0 GB of feature matrices
- 761 603 images carry real Flickr tags (94.5%), 25 visual properties on all of them
- 295 177 words in the fuzzy-matching vocabulary
- `scripts/selftest.py` — 37/37 checks on `flickr1m`
- `flower` → 22 214 matches; `flower` + `vivid` → 13 339; `vien` → `vienna`

**The one order dependency worth remembering:** `build_index.py flickr1m` replaces
the dataset's rows, which is where the imported tags live, and renumbers the ids
the feature matrices are keyed to. So if you ever rebuild it, re-run all four
`import_mirflickr.py` commands afterwards — the tags take 19 seconds, the three
feature sets 35–60 s each. Nothing warns you if you forget; the tags simply come
back empty and a text search returns nothing.

### Why tags and visual properties are two separate controls

The form has a **Tags** text box and a row of **visual property** dropdowns, and
they are deliberately not the same input. The two vocabularies collide almost
completely:

| | Tags box | Visual property dropdowns |
| --- | --- | --- |
| vocabulary | open — 761 603 images' worth of real Flickr tags | closed — 25 words measured from the pixels |
| answered by | Postgres full-text (GIN) | the `uint32` tag bitmask |
| cost | 11–128 ms | ~2 ms over the whole base |
| `blue` means | a word 25 472 people typed | the dominant hue the indexer measured |

**23 of the 25 visual words also occur as real tags.** `blue` is 25 472 user-tagged
images *and* a measured hue; `sky` is 26 700 and a hue band; `red`, `green`,
`yellow`, `dark`, `bright` likewise. Had both gone in one text box, `blue` would
mean one thing or the other depending on which code path answered it — and the
bitmask path would have silently returned a subset, since it knows nothing about
what anyone typed.

Splitting them also fits what each one *is*. A closed set of five mutually
exclusive facets belongs in dropdowns, where it is discoverable without having to
guess; an open vocabulary of hundreds of thousands of tags belongs in a text box.
Give both and the engine intersects them, so `flower` + `vivid` narrows 805 842
images to 13 339 before any visual ranking starts.

## The 1M base

The base arrives as `thumbnails/<shard>/im<n>.jpg` — 805 842 JPEG thumbnails of
64×64, in 81 shards of 10 000. The distribution's tag and feature archives come
separately, as four zips totalling 3 GB.

**The five computed descriptors.** `app/mpeg7.py` computes these from the
thumbnails, following the standard's bin layouts, filter kernels and thresholds.
They are what make the engine work on any image, including one the user uploads:

| descriptor | attribute | dim | what it measures |
| --- | --- | --- | --- |
| Scalable Colour (SCD) | colour | 64 | 256-bin HSV histogram, Haar-transformed, coarsest 64 coefficients |
| Colour Layout (CLD) | colour | 12 | 8×8 DCT of the block-averaged YCbCr image, 6 Y + 3 Cb + 3 Cr |
| Colour Structure (CSD) | colour | 64 | how often each of 64 HMMD colours appears inside an 8×8 structuring element |
| Edge Histogram (EHD) | texture | 80 | dominant edge type among 5 per image-block, over 16 sub-images |
| Homogeneous Texture (HTD) | texture | 62 | log energy mean and deviation in 30 Gabor frequency channels |

What is deliberately left out is the bitstream half of the standard — the 4/5/11-bit
requantisation of the coefficients. The engine compares float vectors with a
Euclidean distance directly, so that coding step would only cost precision.

**The three provided descriptors.** `scripts/import_mirflickr.py` loads the
distribution's own feature archives, computed on the full-resolution originals
that the thumbnail base does not include:

| descriptor | dim | layout |
| --- | --- | --- |
| `mp7_ehd_ref` — Edge Histogram | 150 | 80 local (16 sub-images × 5) + 5 global + 65 semi-global |
| `mp7_htd_ref` — Homogeneous Texture | 43 | DC, SD, 30 scale × orientation energies, 5 scale + 6 orientation marginals |
| `gist` | 480 | GIST global scene descriptor, raw float32 |

Because they describe originals we do not have, **they can only compare images
already in the base** — there is no way to compute them for an uploaded file. The
API says so rather than failing on a missing key, and the form greys them out as
soon as a file is chosen.

Note that both provided MPEG-7 descriptors are *variants* of what we compute, not
the same thing: the provided EHD adds global and semi-global histograms to the 80
local bins, and the provided HTD swaps the standard's 30 deviation values for 11
scale and orientation marginals. Only the first 80 dimensions of the provided EHD
line up with ours — which is what made the mapping check below possible.

### Lining the archives up with the thumbnails

The thumbnails are 1-based (`im1.jpg` … `im1000000.jpg`, in shards of 10 000) and
every archive is 0-based (`0.txt` … `999999.txt`, in directories of 10 000), so
image `im<n>` is archive entry `<n-1>` in directory `(n-1) // 10000`.

That is the obvious guess, and an off-by-one would have silently attached the
wrong tags and features to every image in the base — so it was measured rather
than assumed. Computing our own Edge Histogram for a sample of thumbnails and
correlating it against the first 80 dimensions of the provided one:

| offset | correlation | cosine |
| --- | --- | --- |
| `n-1` | **0.836** | **0.936** |
| `n` | 0.222 | 0.676 |
| `n+1` | 0.240 | 0.673 |
| a random pair, for reference | 0.255 | — |

Both neighbours sit exactly at the random-pair floor, so the alignment is not a
coincidence. The same check is what validates the Edge Histogram implementation:
0.94 cosine against the MPEG-7 reference, computed from 64×64 thumbnails rather
than the originals.

### The derived visual properties

Alongside the real tags, the indexer describes each image by what the descriptors
already looked at — lightness, saturation, dominant hue, edge density and dominant
edge direction:

```
dark vivid blue textured vertical
bright greyscale smooth horizontal
midtone muted green detailed diagonal
```

The three continuous scales are cut at their **terciles over the whole base**,
which is why the measurement happens in the worker and the naming in the parent.
Fixed thresholds were tried first and were useless: a cut at V < 0.3 labelled
78.7% of this base `midtone` and 74.7% `textured`, so neither word narrowed
anything. Terciles give three words that each select a third of the base:

| word group | share of the base each |
| --- | --- |
| `dark` / `midtone` / `bright` | 33.3% each, by construction |
| `greyscale` / `muted` / `vivid` | 33.3% each |
| `smooth` / `detailed` / `textured` | 33.3% each |
| 12 hue names | as they fall — `red` and `orange` 27% each, `blue` 0.9%, `purple` 0.4%; photographs are warm |
| `vertical` / `horizontal` / `diagonal` / `busy` | as they fall — `busy` 58%, `diagonal` 0.6% |

Conjunctions are what make it usable: `dark` selects 268 613 images, `dark vivid`
112 954, `dark vivid blue` 1 557.

### The real tags

`scripts/import_mirflickr.py --tags` reads all 805 842 tag files out of the zip in
19 seconds and writes them to `user_tags`. **761 603 images are tagged (94.5%)**;
the remaining 44 239 carry none, so a text search reaches 94.5% of the base while
a visual search reaches all of it. Mean 11.4 tags per image, already single tokens
(`blackandwhite`, `governorsisland`), UTF-8 throughout — Japanese and accented tags
survive intact.

Two things the data forced:

**`jpg` was the commonest word in the index.** The filename goes into the tsvector
so that an image can be found by name, but `im23.jpg` tokenises to `im23` + `jpg`,
which put `jpg` in all 805 842 documents. The expression now strips the extension
first. After that the commonest real token is `nikon` at 37 549 — 4.7% of the base.

**Ranking became worth paying for again.** `ORDER BY ts_rank` was switched off
while the derived words were the only text: each matched a third of the base and
they all scored alike, so ranking meant scoring 600 000 rows to choose 24. Real
tags are Zipf-distributed, so the same query shape now costs 11–128 ms and orders
by something meaningful. It is back on — and since the worst case is 37 549 rows,
**no query reaches the 50 000 candidate ceiling at all**, which is the other half
of the bias problem the bitmask was built to solve.

### Near-miss matching on the tags

`to_tsvector('simple')` does not stem, which is deliberate — what goes in is what
has to be typed — but it makes the text search unforgiving. Typing `vien` for
Vienna found 7 images, because `vien` happens to be a real tag; `vienna`'s
1 353 were not among them. The engine now expands each typed word to the words it
plausibly meant.

**The expansion happens in the vocabulary, not over the images.** `tag_vocab`
holds every word the full-text index can match with how many images carry it, read
straight out of the tsvector column by `ts_stat`, so it cannot drift from the
index it describes. A near-miss lookup therefore compares one typed word against
**295 177 known words** rather than against 805 842 documents, and costs 4–12 ms.
One GIN trigram index answers both kinds of near-miss:

| | |
| --- | --- |
| a word typed part-way | `word LIKE 'vien%'` — the commonest case by far |
| a typo anywhere in it | `word % 'archtecture'` — trigram similarity ≥ 0.35 |

Of the base's 734 330 distinct tag words, 439 153 occur in exactly one image —
somebody's private string, almost never what a searcher meant. Excluding them
halves the index and improves what it suggests. They stay findable: **the word
actually typed is always searched**, in or out of the vocabulary, so turning
fuzzy matching on can only add results, never take the exact match away.

**Ranking on similarity alone does not work.** `dogg` scores `doggystyle`
(7 images) above `dog` (8 939), and `sunsets` buries `sunset` (17 295) under
`sunsetsofmanspic` (42). The intended word is almost always the common one, so a
candidate is scored

```
(similarity + 0.25 if the typed word is a prefix) × (1 + 0.3 · ln(ndoc))
```

The frequency coefficient is deliberately small: at 1.0 a very common but weakly
similar word wins on frequency alone.

**And expansion has to leave the already-correct queries alone.** `flower` is
22 214 images and needs no help, yet it picked up `flowerwatcher`,
`flowerotica` and `floweria` — noise added to a query that was right. So a
candidate is only offered if it is at least a fifth as common as the word typed.
That is what separates the two cases:

| typed | in | expands to | because |
| --- | --- | --- | --- |
| `vien` | 7 images | `vienna`, `viena`, `viento`, `vienne` | a fifth of 7 excludes nothing |
| `flowr` | 3 images | `flower`, `flow`, `flowers`, `flowrs` | likewise |
| `archtecture` | 23 images | `architecture`, `archetecture`, `architectur` | likewise |
| `flower` | 22 214 images | `flowers` only | the rest are 2% as common |
| `architecture` | 14 725 images | nothing at all | no comparably common relative |

Words under three characters are not expanded: trigram similarity stops
discriminating, and a two-letter prefix matches much of the vocabulary.

**What it did is always reported.** The status line reads
`also searched vien → vien, vienna, viena, viento, vienne`, because a search
engine that silently looks for something other than what was typed is worse than
one that finds nothing. `scripts/build_tagvocab.py --show <word>` prints the
same decision with the score behind every candidate, including the ones dropped
for being too rare.

**What it costs**, measured warm, same query with the expansion off and on:

| query | exact | fuzzy | |
| --- | --- | --- | --- |
| `vien` | 14 ms | 26 ms | 7 matches → 1 353 |
| `sunset` | 90 ms | 123 ms | also searches `sun` |
| `portrait` | 141 ms | 171 ms | also searches `selfportrait` |
| `flower` | 142 ms | 179 ms | also searches `flowers` |
| `nikon` | 180 ms | 172 ms | nothing comparably common; no expansion |

So 12–37 ms, for finding what was meant rather than what was typed. The
vocabulary lookup itself is 4–12 ms of that; the rest is the extra lexemes in the
tsquery, which is why the relative-frequency rule matters twice over — it keeps
precision *and* keeps the OR-group small.

Fuzzy matching is on by default and the form has a checkbox to turn it off. It
needs the `pg_trgm` extension; where that cannot be installed the vocabulary is
skipped, the checkbox greys out and exact search carries on.

## Making the search fast

The eight descriptors come to 3.0 GB of feature matrices — 1.55 GB of that is GIST
alone, at 480 dimensions. Three things keep a query interactive:

**The scan is a matrix product.** `||a−b||² = ||a||² − 2a·b + ||b||²` with the row
norms precomputed at index time, so a full 806k-row scan is one BLAS `gemv`
instead of an 806k × D array of differences. Loading the matrices with
`mmap_mode="r"` keeps startup instant and lets the page cache decide what stays
resident.

**A VP-tree, for the queries it can answer.** `app/vptree.py` is the metric tree
the course asks for: each node splits its points by distance to a vantage point,
and a query prunes a subtree whenever the triangle inequality proves nothing in it
can beat the k-th best distance so far. It is built with one vectorised numpy pass
per node — 806k points node-by-node in Python would not finish — and saved as flat
arrays in a single `.npz`, so the server loads it rather than rebuilds it.

The VP-tree is validated by `scripts/selftest.py`; standalone benchmarking
utilities were removed because they are not part of setup or normal operation.

**The tree is exact and it loses almost everywhere.** That is the measurement, not
a bug: it returns the same neighbours on every query, and pruning does improve
with scale — SCD touched 26% of a 20k index and 14% of this one, which is enough
to draw level there. Two things beat it elsewhere.

The dimension, which is the whole story. Look down the "base touched" column
against dim: 12 dimensions prunes to 25%, 64 to 14–44%, 80 to 94%, and **480
dimensions to 95.3% — the tree visits essentially the entire base and takes 15×
longer than simply reading it.** Once every point is roughly equidistant from the
vantage point, the triangle inequality excludes nothing. That is the curse of
dimensionality doing exactly what it says, and no leaf size fixes it: sweeping
96 → 2048 moved `mp7_ehd` between 374 ms and 181 ms, never near 24 ms.

And the scan is already at the hardware limit. `gist` is
805 842 × 480 × 4 B = **1.55 GB, read in 85.7 ms — 18 GB/s**; `mp7_ehd` is 258 MB
in 23.5 ms, 11 GB/s. The scan is not slow code the tree could beat; it is a
sequential read at main-memory bandwidth, while the tree does scattered reads plus
per-node Python. **No exact index can win here without reading less data** — the
real levers are reducing the dimension (PCA before the tree) or compressing the
vectors (float16, product quantisation), both of which trade away exactness or
precision. So `index=auto` is the scan, and the tree stays selectable so the
comparison can be rerun rather than taken on trust.

What the engine does deliver on the full base, end to end through the API:

| query | candidates | time |
| --- | --- | --- |
| one computed descriptor, no filter | 805 842 | 62–82 ms |
| `gist` (480-d), no filter | 805 842 | 161 ms |
| tags `flower` | 22 214 match | 257–615 ms |
| tags `dog portrait` | 8 939 match | 26 ms |
| properties `dark vivid blue` | 1 557 | 41 ms |
| properties `bright smooth` | 101 827 | 25 ms |
| `flower` + `vivid` | 13 339 | 338 ms |
| `flower` + `vivid` + image + `gist` | 13 339 | 287 ms |

The ranked full-text query is the slowest part, not the visual search — scoring
22 214 matches to order the top 24 costs more than scanning 805 842 feature
vectors.

**A bitmask for the visual-property filter.** This is the part that needed a real
index structure rather than a bigger `LIMIT`. A visual property matches a third of
the base — `dark` is 268 613 rows — and a combined query has to rank *all* of them,
not a slice. Three approaches, measured:

| | cost |
| --- | --- |
| `ORDER BY ts_rank(...) LIMIT 24` | 79 ms mean, 255 ms worst — Postgres scores every match to pick 24 |
| `ORDER BY id LIMIT 50000` | 46 ms, and it silently restricts the search to the lowest-id 6% of the base |
| tag bitmask in the feature store | **~2 ms**, whole base, no ceiling |

The visual vocabulary is closed — 25 words — so every image's properties fit in one
`uint32`. The whole index is 3.2 MB (`tags.npy`), and a conjunctive filter is one
masked comparison:

```python
return np.flatnonzero((self.tag_masks & wanted) == wanted)
```

Real tags are the opposite shape — an open vocabulary, Zipf-distributed, selective
— so they stay in Postgres' GIN index where they belong. When a query gives both,
the two candidate sets are intersected on rows.

Once a candidate set is that large, gathering those rows out of the memory map
costs more than reading the matrix straight through, so above
`FULL_SCAN_FRACTION` (10%) the engine scans everything and masks the result
instead. `scripts/selftest.py` checks the two paths give identical answers.


## Layout

To read the code rather than the prose, the short way through is
`app/datasets.py` (the registry everything else reads) → `app/main.py` (whose
module docstring describes how a query is answered in two stages) →
`app/search.py` (the stage that does the work). `scripts/build_index.py` is the
offline half. Every module opens with a docstring saying what it is for and why
it is shaped that way.

| path | role |
| --- | --- |
| `setup.py` | one-command setup: deps, Postgres, the index, the imports, the vocabulary, self-test |
| `app/datasets.py` | the registry: `DATASETS` is what the server serves, `ALL_DATASETS` adds the offline-only base |
| `app/db.py` | connection, migrating schema, the SQL queries |
| `app/imagebase.py` | image loading and `ImageContext`, the conversions descriptors share |
| `app/mpeg7.py` | the five computed MPEG-7 descriptors |
| `app/descriptors.py` | `color_hist` / `lbp` / `hog`, merged with MPEG-7 and the provided ones into `DESCRIPTORS` |
| `app/visualtags.py` | the closed visual vocabulary and its facets, shared by indexer and API |
| `app/vptree.py` | the metric tree: build, save, exact k-NN |
| `app/search.py` | `BruteForceIndex`, `FeatureStore` (memory maps + tag bitmask), `search()`, and `save_matrix()` which writes the files it reads |
| `app/main.py` | endpoints, `/static`, `/images/<dataset>` |
| `scripts/build_index.py` | offline indexer, parallel over the cores; `--restore-rows` refills an empty database from the index |
| `scripts/import_mirflickr.py` | loads the distribution's tag and feature archives, read in place |
| `scripts/build_tagvocab.py` | the fuzzy-matching vocabulary; `--show <word>` explains one expansion |
| `scripts/make_dataset.py` | regenerates the synthetic labelled base `evaluate.py` scores |
| `scripts/evaluate.py` | precision and recall at k on a labelled base |
| `scripts/selftest.py` | correctness checks on a built index; run after touching the search path |
| `data/features/<dataset>/` | `ids.npy`, `tags.npy`, `tagvocab.json`, `<descriptor>.npy`, `.sqnorm.npy`, `.scale.npy`, `.vptree.npz` |

## API

- `GET /api/datasets` → the bases, how many images each has indexed, their
  visual-property facets, how many words are available for fuzzy matching
  (`fuzzy_words`), and each descriptor's dimension, whether a tree was built, and
  whether it can be computed for an uploaded image (`computable`).
- `GET /api/browse?dataset=&limit=&offset=` → a page of a base, to start from.
- `POST /api/search` (JSON) with
  `{dataset, keywords, visual, image_id, descriptors, k, index, fuzzy}` — query by
  tags and/or visual properties and/or an image already in the base. `fuzzy`
  defaults to true and expands each typed word to its near misses; what it
  searched comes back in `index.note`.
- `POST /api/search/image` (multipart) with `file`, `dataset`, `keywords`, repeated
  `visual`, repeated `descriptors`, `k`, `index`, `fuzzy` — query by an arbitrary
  image file:

  ```bash
  curl -X POST http://127.0.0.1:8000/api/search/image \
       -F "file=@my_photo.jpg" -F "dataset=flickr1m" \
       -F "descriptors=mp7_scd" -F "descriptors=mp7_ehd" -F "k=24"
  ```

  A provided descriptor (`mp7_ehd_ref`, `mp7_htd_ref`, `gist`) here returns 400 if
  it is the only one asked for, and is dropped with a note if others remain.

Both return

```json
{
  "dataset": "flickr1m",
  "query": {"id": 123, "url": "/images/flickr1m/1/im123.jpg", "tags": "dark blue"},
  "descriptors": ["mp7_scd"],
  "results": [{"id": 456, "url": "...", "class_label": "1", "tags": "...", "score": 0.0}],
  "index": {"method": "vptree", "scanned": 81234, "candidates": 805842},
  "elapsed_ms": 42.1,
  "score": "Euclidean distance, 0 is identical"
}
```

most similar first. `query.url` is null for an upload (the file is not under
`/images`; the page previews it locally). With one descriptor `score` is the
Euclidean distance; with several it is the mean of the min-max normalised
distances, which is comparable within one result list only.

## Checking it still works

`scripts/selftest.py <dataset>` asserts the invariants that actually caught bugs
while this was built, and exits non-zero on any failure:

```
PASS  self-query scores exactly 0                 (8 descriptors)
PASS  upload reproduces the index                 largest difference 0
PASS  scan matches float64 ground truth           (8 descriptors)
PASS  tree matches float64 ground truth           (8 descriptors)
PASS  vptree == scan                              (8 descriptors)
PASS  gather == scan+mask                         (1 and 8 descriptors)
PASS  filtered hits all satisfy the keywords
```

37 checks on the Flickr base. Two of them have each caught a real bug. *Upload reproduces the index* recomputes descriptors from the file on disk
and compares them to the indexed row — the only thing that would catch a mismatch
in the column scaling. *Scan matches float64 ground truth* is what exposed the
conditioning problem in the Notes below; `--quick` skips it, since it reads every
matrix end to end.

## Evaluation

`scripts/evaluate.py` scores a base that has real classes by using every image as
a query in turn and counting how many of the top k share its label.

The Flickr base has no classes, so this needs a labelled one. The only labelled
base here is a synthetic stand-in — 30 images, three shapes, colours drawn at
random — and it is **not served**: 30 generated pictures have no business sitting
next to 805 842 real ones in the interface. It is still defined
(`app/datasets.py`, in `ALL_DATASETS` but not `DATASETS`) so the measurement can
be reproduced in three commands:

```bash
python scripts/make_dataset.py             # regenerate the 30 images
python scripts/build_index.py objects      # index them
python scripts/evaluate.py objects --k 9   # score every combination
```

That is where the numbers below come from. Because the class is the shape and the
colour is random, the result is a check on the descriptors themselves — a shape
descriptor must find the class and a colour descriptor must not:

```
descriptors                                      P@k     R@k      ms
--------------------------------------------------------------------
color_hist                                     0.300   0.300    0.12
lbp                                            0.633   0.633    0.02
hog                                            0.978   0.978    0.04
color_hist + lbp                               0.567   0.567    0.05
color_hist + hog                               0.859   0.859    0.06
lbp + hog                                      0.915   0.915    0.05
color_hist + lbp + hog                         0.878   0.878    0.07
```

HOG, a shape descriptor, finds the class; the colour histogram sits at chance
(1/3) because the colours were drawn at random; and averaging a good descriptor
with a useless one lands in between. Exactly what should happen, which is what
makes it worth keeping as a result.

The Flickr base has no class labels, so precision/recall evaluation applies only
to the synthetic `objects` dataset.

**At a million images, descriptors that correlate strongly still return almost
disjoint neighbour sets.** Our Edge Histogram matches the provided one at 0.94
cosine, yet they share only 4% of their top 24 — because the top 24 of 805 842 is
the top 0.003%, a selection fine enough that a small difference in the metric
reorders it completely. Widen the window and they converge (17.2% at k = 2000)
while genuinely unrelated pairs stay flat (1.3%), which is what tells you the 4%
is real agreement and not noise: chance would be 0.003%.

Two practical consequences. Which descriptor you pick matters more than anything
else in the engine — more than the index structure, more than k. And combining
descriptors is not redundant: they are mostly measuring different things, which is
exactly why the averaged-distance mode is worth having.

## Adding to it

- **A descriptor**: write a function taking an `ImageContext` in `app/mpeg7.py` or
  `app/descriptors.py`, register it in `DESCRIPTORS` and `DESCRIPTOR_META`, list it
  on a dataset in `app/datasets.py`, and rebuild. If its components live on
  different scales, add it to `SCALED_DESCRIPTORS` and the indexer will divide each
  column by its deviation over the base.
- **A base**: one `Dataset` entry in `app/datasets.py`, added to `DATASETS`, plus
  `python scripts/build_index.py <name>`. Schema, mounts, API and UI all read
  that registry — and the UI's base selector, hidden while there is only one
  base, reappears by itself. Listing it in `ALL_DATASETS` only, as the synthetic
  `objects` base is, makes it buildable and measurable offline without serving
  it.
- **Another index structure**: `FeatureStore.search()` picks between
  `BruteForceIndex` and `VPTree`, which share a `knn(query, k)` signature. An
  R-tree or an LSH table exposing the same method drops in there.

## Notes

- **Constant columns are not scaled.** Four of `mp7_htd`'s 62 columns — the
  deviations of high-frequency bands that hold a single symmetric pair of
  coefficients — are zero up to float rounding (std ~1e-7). The old guard only
  skipped deviations below 1e-9, so those columns were divided by ~1e-7 and the
  rounding noise that 88% of images carry there became values of 0.35 to 56,
  enough to reorder the nearest neighbours: on ten random HTD queries the
  corrected top 24 kept between 4 and 23 of the old ones. `MIN_COLUMN_SCALE` in
  `app/search.py` now leaves any column with a deviation under 1e-4 unscaled.
- The server keeps a connection pool (`db.open_pool`) instead of opening a
  Postgres connection per request, and the keyword filter in front of a visual
  ranking fetches ids only (`db.keyword_ids`) rather than whole rows.

- Separators in filenames are blanked out before `to_tsvector`, so
  `circle_blue_06.png` indexes as `circle blue 06 png`; Postgres' parser would
  otherwise keep the whole filename as a single token and `blue` would match
  nothing.
- The VP-tree answers "k nearest under one metric". As soon as several descriptors
  are combined, the ranking is an average of min-max normalised distances — not a
  metric — so the scan is the honest way to compute it, and `index=auto` falls
  back to it. Same when keywords restrict the candidates.
- `build_index.py` replaces only the rows of the dataset it is given, so an
  unserved base built for evaluation is untouched by a Flickr rebuild. It drops
  the GIN index around the bulk load and recreates it after; maintaining it
  across 800k inserts costs far more.
- Re-running `build_index.py flickr1m` clears the imported tags and features; see
  **Coming back to it** above.
- `init_schema()` migrates rather than destroys where it can. Adding `user_tags`
  and changing the `keywords` expression both need that generated column rebuilt,
  which rewrites the table — a couple of minutes on 805k rows, against 8 minutes
  to redo the descriptor pass. `cbir_meta.schema_version` is what tells it the
  expression changed; bump `SCHEMA_VERSION` when editing `KEYWORDS_EXPRESSION`.
  A table missing a *core* column still gets rebuilt, and nothing is lost there
  either — the indexer is the only thing that writes rows.
- The archives are read in place, never unpacked. Four million files of a few
  hundred bytes each would cost several GB and far more time on the filesystem
  than streaming the zip entries once — 19 s for the tags, 35–60 s per feature set.
- MPEG-7's Haar transform for the SCD keeps sums and differences unnormalised, so
  the coarse coefficients are larger than the fine ones and weigh more in the
  distance. That is the standard's own behaviour; it allocates fewer bits to the
  fine bands for the same reason.
- **The SCD's bin order is not a free choice.** Keeping 64 of 256 Haar
  coefficients resolves the histogram down to blocks of four adjacent bins, so
  whichever HSV axis varies fastest is the one that gets summed away. Ordering the
  bins V-fastest put all four bins a greyscale image can occupy into one block,
  and the 7.4% of this base that is black and white collapsed onto a single
  identical vector — `im7.jpg` at mean brightness 0.23 and `im49.jpg` at 0.72
  scored distance 0 from each other. Ordering H-fastest instead keeps brightness,
  and the base went from 18 529 distinct SCD vectors to 19 999 of 20 000 in the
  sample. The remaining tie is one genuinely duplicated photograph, and it shows
  up in all five descriptors. Worth noting that the fix made the VP-tree *slower*
  on SCD — 6% of the base touched before, 18% after — because a 1 471-way tie is a
  degenerate cluster that prunes trivially. Faster search over a descriptor that
  cannot tell black from white is not a trade worth having.
- Identical images tie at distance 0, so a query is not guaranteed to be rank 1 in
  its own result list — only to score 0. `scripts/selftest.py` asserts the latter.
- **The fast scan is ill-conditioned, and the fix is to over-select.** The
  `‖a‖² − 2a·b + ‖b‖²` identity is what makes a full scan one BLAS call, but it
  computes a small number by subtracting large ones. On `mp7_htd_ref` the squared
  norms are ~19 018 and the squared distance ~4.19 — a cancellation of **3939×**,
  which leaves about 1e-3 of float32 slack sitting on the answer. In a measured
  case that was 8.1e-4 of error against a 5.7e-4 gap between the 24th and 25th
  neighbour, i.e. **1.4× bigger than the difference it had to resolve**, and the
  top k came back mis-ordered. It showed up as the VP-tree and the scan
  disagreeing on one query in twenty — where the *tree* was right.

  The scan now pre-selects `max(4k, k+64)` candidates on the approximate
  distances and ranks that pool in float64, which is a few dozen extra rows and
  makes the returned k exact. The cancellation tracks column scaling almost
  perfectly — `mp7_htd_ref` 3939×, `mp7_htd` 892×, `mp7_cld` 274× are the three
  scaled descriptors, against 5–18× for the unscaled ones — because dividing by
  each column's deviation pushes the cloud away from the origin. Centring the
  matrices at index time would cure it at the source, since L2 is
  translation-invariant; over-selecting fixes it without rewriting 3 GB.
  `scripts/selftest.py` now compares both search paths against float64 ground
  truth for every descriptor, which is the check that caught this.
