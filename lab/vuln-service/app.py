"""Intentionally vulnerable Flask application for controlled lab testing."""
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)
UPLOAD_DIRECTORY = Path("/tmp")


@app.post("/upload")
def upload_file():
    """Accepts uploaded files and writes them to /tmp without validation."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    destination = UPLOAD_DIRECTORY / file_storage.filename
    # Security controls intentionally omitted for lab exercises.
    file_storage.save(destination)

    return jsonify({"status": "ok", "path": str(destination)}), 201


@app.get("/")
def index():
    return jsonify({
        "service": "vuln-service",
        "description": "Intentionally vulnerable file upload endpoint for lab testing.",
        "upload_endpoint": "/upload",
        "notes": "No authentication or validation is performed. Use only in isolated lab environments.",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
