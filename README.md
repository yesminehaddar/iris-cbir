# Iris: content-based image retrieval over 805,842 images

Iris is a search engine that finds images by what they look like, over the 805,842 thumbnails of the MIRFLICKR-1M collection. You can search by **tags**, by **visual properties** (for example `dark vivid blue`) or by **example image**, from the base or uploaded, and combine all three. Every query answers in tens of milliseconds on a laptop, and the results are exact.

![The Iris search interface](docs/img/interface.png)

## Results

| | |
| --- | --- |
| Images indexed | 805,842 thumbnails, 8 descriptors, 3.0 GB of feature matrices |
| Image query, one descriptor | 18 to 67 ms, exact scan of the whole base |
| Visual-property filter | about 2 ms over the whole base |
| Text query, fuzzy matching included | 3 to 100 ms |
| Correctness | 37/37 self-test checks pass against float64 ground truth |

Measured on an Intel Core i7-13650HX laptop (16 GB RAM, Windows 11), medians of 10 to 20 queries after warm-up, k = 24.

### The main finding: the VP-tree loses to a linear scan

Iris implements the vantage-point tree, the metric index structure from the course, and compares it with a well-implemented exact linear scan on all 8 descriptors. Both return identical neighbours on all 160 test queries. The tree is faster on only one descriptor.

![Query time of the linear scan and the VP-tree for each descriptor](docs/img/scan-vs-vptree.png)

At 12 to 64 dimensions the tree visits 17% to 49% of the base. At 80 and 480 dimensions it visits 98% of it: distances concentrate, so the triangle inequality excludes almost nothing (the curse of dimensionality). The tree then reads nearly the whole base in scattered order, while the scan reads it sequentially at memory bandwidth (GIST, 1.55 GB, in 67 ms). That is why the scan is the default, and the tree stays selectable in the interface so the comparison can be rerun.

![Share of the base visited by the VP-tree, by descriptor dimension](docs/img/vptree-dimension.png)

## How it works

![Architecture of Iris](docs/img/architecture.png)

A single web page talks JSON to a FastAPI server. Image URLs and tags live in PostgreSQL 16 (in Docker), and the feature matrices live in memory-mapped NumPy files. A query runs in two stages: an optional filter (tags and visual properties) builds a candidate set, then the image query ranks the candidates.

### Search methods

1. **Text search.** The Flickr tags of 761,603 images are a `tsvector` column with a GIN full-text index. Words are ANDed and ranked with `ts_rank`.
2. **Fuzzy matching.** Each typed word is expanded against a 295,177-word vocabulary with a `pg_trgm` GIN index: prefixes (`vien` finds `vienna`) and typos (`archtecture` finds `architecture`). Candidates are scored by similarity weighted by frequency, and the exact word is always searched too, so fuzzy matching can only add results.
3. **Visual-property filter.** Lightness, saturation, texture, dominant hue and dominant edge direction are measured from the pixels and cut at terciles of the base. The result is a closed vocabulary of 25 words that fits in one `uint32` per image, so a filter is a single vectorised comparison over a 3 MB array.
4. **Image similarity.** Euclidean distance over up to 8 descriptors, with exact k nearest neighbours by linear scan (default) or VP-tree.
5. **Combinations.** Several descriptors are min-max normalised and averaged. Filters and an image query are combined by ranking only inside the candidate set, gathering the rows when the set is under 10% of the base and masking a full scan otherwise.

### Descriptors

| Descriptor | Kind | Dim | Source |
| --- | --- | --- | --- |
| Scalable Colour (`mp7_scd`) | colour: HSV histogram, Haar transform | 64 | computed from the pixels |
| Colour Layout (`mp7_cld`) | colour: 8x8 DCT of YCbCr | 12 | computed from the pixels |
| Colour Structure (`mp7_csd`) | colour: local colour structure (HMMD) | 64 | computed from the pixels |
| Edge Histogram (`mp7_ehd`) | texture: 5 edge types x 16 sub-images | 80 | computed from the pixels |
| Homogeneous Texture (`mp7_htd`) | texture: energy in 30 frequency channels | 62 | computed from the pixels |
| `mp7_ehd_ref`, `mp7_htd_ref`, `gist` | texture and scene | 150 / 43 / 480 | provided with the dataset |

The five MPEG-7 descriptors are implemented in `app/mpeg7.py` and also work on an uploaded image. The three provided ones only exist for images already in the base. Our Edge Histogram reaches a cosine of 0.94 against the provided reference, even though it is computed from 64x64 thumbnails.

### Fast exact scan

The distance is expanded as `‖q − x‖² = ‖x‖² − 2 q·x + ‖q‖²` with the row norms precomputed, so scanning all 805,842 rows is one BLAS matrix-vector product over the memory-mapped matrix. This identity loses precision in float32 when the norms are large, so the scan pre-selects `max(4k, k + 64)` candidates and re-ranks them in float64. The returned top k is exact.

## Problems we hit and how we fixed them

- **Black-and-white images all looked identical.** Keeping 64 of the 256 Scalable Colour coefficients merges adjacent bins, and with brightness varying fastest, every black-and-white image (7.4% of the base) collapsed onto one vector. Ordering the bins hue-fastest fixed it: in a 20,000-image sample, distinct vectors went from 18,529 to 19,999.
- **Wrong ranking on one descriptor.** On `mp7_htd_ref` the float32 error of the norm identity was larger than the gap between the 24th and 25th neighbour. Re-ranking a candidate pool in float64 made the top k exact, and the self-test now checks it.
- **Off-by-one between images and archives.** Thumbnails start at `im1.jpg` and archive entries at `0.txt`. Instead of assuming, we correlated our Edge Histogram with the provided one at three offsets: only `n − 1` matched (correlation 0.836, against about 0.24 for the others).
- **Every image matched "jpg".** The file extension was indexed as a tag. Stripping it fixed text ranking.
- **A filter that filtered nothing.** Fixed lightness thresholds labelled 79% of the base "midtone". Cutting at terciles makes each word select a third of the base.
- **Slow Colour Structure.** Testing whether a window contains a colour is done with bitwise ORs of 64-bit masks, in six array operations instead of a loop per colour.

## Run it

Requirements: Python 3.10 or newer, and Docker Desktop running (it only hosts PostgreSQL).

**1. Get the data.** Download from the [MIRFLICKR website](https://press.liacs.nl/mirflickr/) the 1M thumbnails and four archives: `tags.zip`, `features_edgehistogram.zip`, `features_homogeneoustexture.zip` and `features_gist.zip`. Place them next to this repository:

```
some-folder/
├── iris-cbir/          this repository
├── thumbnails/         1/, 2/, ... with im1.jpg, im2.jpg, ...
├── tags.zip
├── features_edgehistogram.zip
├── features_homogeneoustexture.zip
└── features_gist.zip
```

The zips are read in place and never unpacked. They can live elsewhere with `python setup.py --data <folder>`, and the thumbnails with `FLICKR_DIR=<path>` in `.env`.

**2. Create a virtual environment.**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
```

**3. Build and serve.**

```bash
python setup.py                  # full base, about 15 minutes the first time
python -m uvicorn app.main:app --port 8000
```

Then open <http://127.0.0.1:8000>. To try it quickly, `python setup.py --quick --serve` indexes only 20,000 images (about 1 minute) and starts the server.

`setup.py` checks Python, installs `requirements.txt`, creates `.env` from `.env.example`, starts PostgreSQL, computes the descriptors in parallel, imports the archives, builds the fuzzy vocabulary and runs the self-test. It skips any step that is already done, so it can be rerun safely.

### Useful commands

```bash
python scripts/selftest.py flickr1m              # 37 correctness checks
python scripts/build_tagvocab.py --show vien     # explain one fuzzy expansion
python scripts/evaluate.py objects --k 9         # precision@k on the synthetic labelled base
```

### If something goes wrong

| Problem | Fix |
| --- | --- |
| `Docker was not found` or `not responding` | Start Docker Desktop and wait for *Engine running* |
| Port 5432 already in use | Stop the other PostgreSQL, or change the port in `docker-compose.yml` and in `DATABASE_URL` in `.env` |
| `ModuleNotFoundError` when starting the server | The virtual environment is not active |
| Text search returns nothing | The tags were not imported: run `python setup.py` again |
| `no thumbnails found` | `thumbnails/` is not next to the repository: move it or set `FLICKR_DIR` in `.env` |

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/datasets` | Bases, image counts, visual facets, descriptors |
| `GET /api/browse` | A page of the base, to start browsing |
| `POST /api/search` | JSON body: keywords, visual properties, image id, descriptors, k, index, fuzzy |
| `POST /api/search/image` | Same as a multipart form, with an uploaded file as the query |

Every response returns the results (id, URL, tags, score) plus the strategy used, the number of vectors compared and the time taken.

## Project layout

| Path | Content |
| --- | --- |
| `app/` | The server: `main.py` (API), `search.py` (scan, bitmask, fusion), `vptree.py`, `mpeg7.py` (descriptors), `db.py` (SQL, fuzzy matching), `static/index.html` (web client) |
| `scripts/` | Offline tools: `build_index.py`, `import_mirflickr.py`, `build_tagvocab.py`, `selftest.py`, `evaluate.py`, `make_dataset.py` |
| `data/objects/` | Small synthetic labelled base (3 shapes x 10 images) for the precision evaluation |
| `docs/DESIGN.md` | Full design notes and measurements |
| `setup.py` | One-command setup, re-runnable |

The feature index (`data/features/`, 3 GB) is not in the repository. `setup.py` builds it.

## Limits and next steps

- Precision is measured only on the small synthetic base: the Flickr base has no class labels.
- MPEG-7 descriptors capture colour, texture and layout, not meaning. Learned embeddings such as CLIP would capture semantics and allow text-to-image search without tags.
- To go faster in high dimension, the next levers are reading less data (PCA, float16, product quantisation) or approximate indexes such as HNSW or IVF.
- Relevance feedback, so the user can mark good and bad results and re-weight the descriptors.

## Credits

Built by Yesmine Haddar and Amir Triki at SUP'COM (École Supérieure des Communications de Tunis), academic year 2026 to 2027, supervised by Pr. Riadh Tebourbi.

The images, tags and provided descriptors come from the MIRFLICKR-1M collection (Mark J. Huiskes, Bart Thomee and Michael S. Lew, LIACS Medialab, Leiden University). They are not included in this repository, and each image keeps its own Flickr licence.

The code is released under the MIT licence, see [LICENSE](LICENSE).
