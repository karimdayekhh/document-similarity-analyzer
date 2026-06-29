import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import all logic from the original module
from plagiarism_checker import (
    Document,
    preprocess_text,
    build_token_frequency_table,
    build_shingle_table,
    compare_all_document_pairs,
    classify_similarity_results,
    build_suspicious_pair_tree,
    merge_sort_similarity_results,
    DEFAULT_SHINGLE_SIZE,
    HIGH_SIMILARITY_THRESHOLD,
    MODERATE_SIMILARITY_THRESHOLD,
)

app = Flask(__name__)

# Restrict cross-origin requests to the known frontends.
# Override in production by setting ALLOWED_ORIGINS to a comma-separated list.
DEFAULT_ORIGINS = [
    "https://karimdayekhh.github.io",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
allowed_origins = os.environ.get("ALLOWED_ORIGINS")
if allowed_origins:
    origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
else:
    origins = DEFAULT_ORIGINS
CORS(app, origins=origins)

ALLOWED_EXTENSIONS = {"txt"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_documents_from_files(uploaded_files):
    """Build Document objects from uploaded file storage objects."""
    documents = []
    for idx, file in enumerate(uploaded_files, start=1):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            raw_text = file.read().decode("utf-8", errors="replace")
            doc = Document(idx, filename, filename, raw_text)
            documents.append(doc)
    return documents


def run_pipeline(documents, shingle_size=DEFAULT_SHINGLE_SIZE):
    """Run the full similarity pipeline and return structured results."""
    for doc in documents:
        doc.tokens = preprocess_text(doc.raw_text)
        doc.token_frequencies = build_token_frequency_table(doc.tokens)
        doc.shingles = build_shingle_table(doc.tokens, shingle_size)

    results = compare_all_document_pairs(documents)
    classify_similarity_results(results)
    suspicious_tree = build_suspicious_pair_tree(results)
    ranked = merge_sort_similarity_results(results)

    # Serialize documents
    docs_data = []
    for doc in documents:
        docs_data.append({
            "id": doc.document_id,
            "name": doc.file_name,
            "word_count": len(doc.tokens),
            "unique_words": doc.token_frequencies.count(),
            "unique_shingles": doc.shingles.count(),
        })

    # Serialize ranked pairs
    pairs_data = []
    for rank, result in enumerate(ranked, start=1):
        pairs_data.append({
            "rank": rank,
            "doc1_id": result.first_document.document_id,
            "doc1_name": result.first_document.file_name,
            "doc2_id": result.second_document.document_id,
            "doc2_name": result.second_document.file_name,
            "score": round(result.score * 100, 2),
            "score_raw": result.score,
            "label": result.label,
            "is_suspicious": result.is_suspicious,
            "shared_count": result.shared_count,
            "total_count": result.total_count,
        })

    # Serialize suspicious pairs from BST traversal
    suspicious_data = []
    for rank, result in enumerate(suspicious_tree.in_order_traversal(), start=1):
        suspicious_data.append({
            "rank": rank,
            "doc1_id": result.first_document.document_id,
            "doc1_name": result.first_document.file_name,
            "doc2_id": result.second_document.document_id,
            "doc2_name": result.second_document.file_name,
            "score": round(result.score * 100, 2),
            "label": result.label,
            "shared_count": result.shared_count,
            "total_count": result.total_count,
        })

    return {
        "config": {
            "shingle_size": shingle_size,
            "high_threshold": round(HIGH_SIMILARITY_THRESHOLD * 100, 2),
            "moderate_threshold": round(MODERATE_SIMILARITY_THRESHOLD * 100, 2),
        },
        "documents": docs_data,
        "ranked_pairs": pairs_data,
        "suspicious_pairs": suspicious_data,
        "total_pairs": len(ranked),
        "suspicious_count": suspicious_tree.count(),
    }


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    txt_files = [f for f in files if f and allowed_file(f.filename)]

    if len(txt_files) < 2:
        return jsonify({"error": "Please upload at least 2 .txt files"}), 400

    shingle_size = int(request.form.get("shingle_size", DEFAULT_SHINGLE_SIZE))
    shingle_size = max(1, min(shingle_size, 10))

    try:
        documents = process_documents_from_files(txt_files)
        result = run_pipeline(documents, shingle_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Port and debug come from environment so the same code runs locally
    # and on hosts like Render/Railway that inject a PORT variable.
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
