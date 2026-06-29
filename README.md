# Document Similarity Analyzer (DocScan)

**DocScan** is a plagiarism-detection tool that compares a set of text documents and ranks how similar they are to each other, using **shingling** and **Jaccard similarity**. The core data structures — a hash table, a binary search tree, and merge sort — are implemented from scratch in Python rather than imported from libraries, to demonstrate the underlying algorithms.

**▶ Live demo:** https://karimdayekhh.github.io/document-similarity-analyzer/
*(The demo's backend runs on a free tier that sleeps when idle — the first analysis may take up to a minute to wake the server, then it's fast.)*

It ships with both a command-line interface and a Flask-backed web dashboard.

## What it demonstrates

- **Hash table (from scratch):** separate-chaining collision handling, dynamic resizing on load factor, polynomial hashing — used to store token frequencies and shingle sets.
- **Binary search tree (from scratch):** stores only "suspicious" pairs, ordered by similarity score with shared-shingle and document-ID tie-breakers, so an in-order traversal returns results highest-similarity first.
- **Merge sort (from scratch):** ranks every document pair in descending order of similarity.
- **Shingling + Jaccard similarity:** documents are broken into overlapping k-word shingles; similarity is the size of the shingle intersection over the union.

## How it works

1. Each document is tokenized and normalized.
2. Token frequencies and overlapping k-word shingles are stored in the custom hash table.
3. Every document pair is compared via Jaccard similarity of their shingle sets.
4. Pairs are classified as High (≥75%), Moderate (≥40%), or Low.
5. Results are ranked with merge sort; suspicious pairs are stored in the BST for ordered retrieval.

## Tech stack

**Backend:** Python, Flask, Flask-CORS, Gunicorn
**Frontend:** HTML, CSS, JavaScript, drag-and-drop upload, dark/light mode
**Core algorithms:** custom hash table, BST, merge sort
**Deployment:** backend on Render, frontend on GitHub Pages

## Running it locally

### Web dashboard
```bash
pip install -r requirements.txt
python app.py
```
The API runs on `http://localhost:5000`. In `index.html`, set `API_BASE` (near the top of the `<script>`) to `http://localhost:5000`, then open `index.html` in your browser. If you hit CORS issues, serve the frontend with `python -m http.server 8080` and visit `http://localhost:8080`.

Upload at least two `.txt` files and adjust the shingle size (1–10, default 3) to compare.

### Command line
```bash
python plagiarism_checker.py
```
Enter the path to a folder containing `.txt` files when prompted. It prints a ranked similarity report.

## Deployment

The backend is production-ready (configurable port, debug off by default, Gunicorn included).

**Backend (Render / Railway):**
1. Connect this repo to your host.
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn app:app` (uses the included `Procfile`).
4. The host's `PORT` environment variable is picked up automatically.

**Frontend:**
Set `API_BASE` in `index.html` to your deployed backend URL, then host `index.html` on any static host (GitHub Pages, Netlify, Vercel).

## API

`POST /api/analyze` (multipart/form-data)
- `files` — two or more `.txt` files
- `shingle_size` — integer 1–10 (optional, default 3)

Returns JSON: per-document stats, all ranked pairs, and suspicious pairs.

`GET /api/health` — returns `{"status": "ok"}`.

## Author

Karim Dayekh — Computer Science, Lebanese American University
