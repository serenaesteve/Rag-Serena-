import os
import re
import hashlib
import requests
import chromadb
import fitz
from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)
app.secret_key = "rag-serena-secret-2024"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///rag_serena.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
CHROMA_DIR      = "chroma_db"
OLLAMA_EMBED    = "http://127.0.0.1:11434/api/embed"
OLLAMA_CHAT     = "http://127.0.0.1:11434/api/generate"
EMBED_MODEL     = "nomic-embed-text:v1.5"
CHAT_MODEL      = "llama3"
TOP_K           = 5
CONTEXT_BEFORE  = 1
CONTEXT_AFTER   = 2
MIN_WORDS       = 20
MAX_CHARS       = 4000

# ── MODELS ─────────────────────────────────────────────────────────────────────
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    documents     = db.relationship("Document", backref="owner", lazy=True)

class Document(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    filename       = db.Column(db.String(200), nullable=False)
    collection_name= db.Column(db.String(200), nullable=False)
    chunk_count    = db.Column(db.Integer, default=0)
    user_id        = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

class ChatMessage(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    doc_id      = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=True)
    role        = db.Column(db.String(10), nullable=False)
    content     = db.Column(db.Text, nullable=False)

with app.app_context():
    db.create_all()

# ── AUTH HELPERS ───────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "No autenticado"}), 401
        return f(*args, **kwargs)
    return decorated

# ── RAG HELPERS ────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_paragraphs(text):
    raw = re.split(r"\n\s*\n", text)
    result = []
    for p in raw:
        p = clean_text(p)
        if len(p.split()) >= MIN_WORDS:
            result.append(p[:MAX_CHARS])
    return result

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        data = page.get_text("dict")
        for block in data["blocks"]:
            if block["type"] != 0:
                continue
            lines = []
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if line_text:
                    lines.append(line_text)
            if lines:
                pages.append(" ".join(lines))
    return "\n\n".join(pages)

def get_embedding(text):
    r = requests.post(OLLAMA_EMBED, json={"model": EMBED_MODEL, "input": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"][0]

def make_id(text, index):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"p_{index}_{h}"

def index_document(pdf_path, collection_name):
    text = pdf_to_text(pdf_path)
    paragraphs = split_paragraphs(text)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=collection_name)
    count = 0
    for i, para in enumerate(paragraphs):
        emb = get_embedding(para)
        if emb is None:
            continue
        col.add(
            ids=[make_id(para, i)],
            embeddings=[emb],
            documents=[para],
            metadatas=[{"paragraph_index": i, "word_count": len(para.split())}]
        )
        count += 1
    return count

def search_context(query, collection_name):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(name=collection_name)
    all_data = col.get(include=["documents", "metadatas"])
    paragraphs = {m["paragraph_index"]: d for d, m in zip(all_data["documents"], all_data["metadatas"])}
    q_emb = get_embedding(query)
    results = col.query(query_embeddings=[q_emb], n_results=TOP_K,
                        include=["documents", "metadatas", "distances"])
    context_parts = []
    seen = set()
    for i in range(len(results["documents"][0])):
        idx = results["metadatas"][0][i]["paragraph_index"]
        for j in range(idx - CONTEXT_BEFORE, idx + CONTEXT_AFTER + 1):
            if j in paragraphs and j not in seen:
                context_parts.append(paragraphs[j])
                seen.add(j)
    return "\n\n".join(context_parts)

def generate_answer(query, context):
    prompt = f"""Eres un asistente experto. Usa SOLO el siguiente contexto para responder la pregunta.
Si la respuesta no está en el contexto, dilo claramente.

CONTEXTO:
{context}

PREGUNTA: {query}

RESPUESTA:"""
    r = requests.post(OLLAMA_CHAT, json={
        "model": CHAT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()

# ── AUTH ROUTES ────────────────────────────────────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Usuario ya existe"}), 400
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "Email ya registrado"}), 400
    user = User(
        username=data["username"],
        email=data["email"],
        password_hash=generate_password_hash(data["password"])
    )
    db.session.add(user)
    db.session.commit()
    session["user_id"] = user.id
    session["username"] = user.username
    return jsonify({"ok": True, "username": user.username})

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data["username"]).first()
    if not user or not check_password_hash(user.password_hash, data["password"]):
        return jsonify({"error": "Credenciales incorrectas"}), 401
    session["user_id"] = user.id
    session["username"] = user.username
    return jsonify({"ok": True, "username": user.username})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/me")
def me():
    if "user_id" not in session:
        return jsonify({"logged": False})
    return jsonify({"logged": True, "username": session["username"]})

# ── DOCUMENT ROUTES ────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Solo PDF"}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    collection_name = f"user_{session['user_id']}_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
    try:
        count = index_document(path, collection_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    doc = Document(
        filename=filename,
        collection_name=collection_name,
        chunk_count=count,
        user_id=session["user_id"]
    )
    db.session.add(doc)
    db.session.commit()
    return jsonify({"ok": True, "doc_id": doc.id, "filename": filename, "chunks": count})

@app.route("/api/documents")
@login_required
def documents():
    docs = Document.query.filter_by(user_id=session["user_id"]).all()
    return jsonify([{"id": d.id, "filename": d.filename, "chunks": d.chunk_count} for d in docs])

@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
@login_required
def delete_document(doc_id):
    doc = Document.query.filter_by(id=doc_id, user_id=session["user_id"]).first_or_404()
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(doc.collection_name)
    except Exception:
        pass
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"ok": True})

# ── CHAT ROUTES ────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    doc_id = data.get("doc_id")
    if not query:
        return jsonify({"error": "Consulta vacía"}), 400
    if not doc_id:
        return jsonify({"error": "Selecciona un documento"}), 400
    doc = Document.query.filter_by(id=doc_id, user_id=session["user_id"]).first_or_404()
    try:
        context = search_context(query, doc.collection_name)
        answer = generate_answer(query, context)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    db.session.add(ChatMessage(user_id=session["user_id"], doc_id=doc_id, role="user", content=query))
    db.session.add(ChatMessage(user_id=session["user_id"], doc_id=doc_id, role="assistant", content=answer))
    db.session.commit()
    return jsonify({"answer": answer})

@app.route("/api/history/<int:doc_id>")
@login_required
def history(doc_id):
    msgs = ChatMessage.query.filter_by(user_id=session["user_id"], doc_id=doc_id).all()
    return jsonify([{"role": m.role, "content": m.content} for m in msgs])

# ── MAIN ───────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
