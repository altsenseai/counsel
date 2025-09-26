import os
import json
import time
import shutil
import pickle
import logging
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename

# -------- .env (optional) ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- Gemini SDK ---------------
import google.generativeai as genai
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# -------- Flask --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
BASE_DIR = os.path.dirname(__file__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.permanent_session_lifetime = timedelta(days=7)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -------- Logging ------------------
logging.basicConfig(level=logging.INFO)

# -------- In-memory stores (demo) --
USERS = {}   # {"username": {"password": "...", "name": "..."}}
CHATS = {}   # {"username": [{"id","title","messages":[{role,content}],...}]}

# =========================
# Embeddings (SentenceTransformers + FAISS)
# =========================
EMBEDDINGS_DIR = os.environ.get("EMBEDDINGS_DIR", os.path.join(BASE_DIR, "embeddings"))
PROCESSED_FOLDER = os.path.join(app.config["UPLOAD_FOLDER"], "_processed")
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(EMBEDDINGS_DIR, "chunks.pkl")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "metadata.json")

# Import heavy deps safely so the app can still boot if missing
EMBEDDING_DEPS_OK = True
EMBEDDING_IMPORT_ERROR = ""
try:
    import faiss
    from PyPDF2 import PdfReader
    from sentence_transformers import SentenceTransformer
except Exception as e:
    EMBEDDING_DEPS_OK = False
    EMBEDDING_IMPORT_ERROR = f"{type(e).__name__}: {e}"

_embed_model = None  # lazy init


def get_embed_model():
    """Load embedding model once."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _allowed_file(name: str) -> bool:
    return name.lower().endswith((".pdf", ".txt"))


def extract_text_from_file(file_path: str):
    """Extract text chunks from PDF/TXT by simple paragraph split."""
    chunks = []
    if file_path.lower().endswith(".pdf"):
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                        chunks.extend(parts)
        except Exception as e:
            app.logger.warning(f"PDF parse error for {file_path}: {e}")
    elif file_path.lower().endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                chunks.extend(parts)
        except Exception as e:
            app.logger.warning(f"TXT parse error for {file_path}: {e}")
    return chunks


def save_embeddings(chunks, embeddings_np, index):
    """Persist FAISS index, chunks, embeddings, and metadata."""
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    try:
        np.save(EMBEDDINGS_FILE, embeddings_np.astype("float32"))
    except Exception:
        with open(EMBEDDINGS_FILE.replace(".npy", ".pkl"), "wb") as f:
            pickle.dump(embeddings_np.astype("float32"), f)

    faiss.write_index(index, INDEX_FILE)

    meta = {
        "model_used": "all-MiniLM-L6-v2",
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings_np.shape[1]) if embeddings_np.size else 0,
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_embeddings():
    """Return (index, chunks, embeddings_np, metadata) or (None, [], None, None)."""
    try:
        if not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
            return None, [], None, None
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        embeddings_np = None
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                embeddings_np = np.load(EMBEDDINGS_FILE)
            except Exception:
                with open(EMBEDDINGS_FILE.replace(".npy", ".pkl"), "rb") as f:
                    embeddings_np = pickle.load(f)
        meta = None
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return index, chunks, embeddings_np, meta
    except Exception as e:
        app.logger.exception(f"Error loading embeddings: {e}")
        return None, [], None, None


def safe_archive_file(src_path: str, fname: str):
    """
    Move a file from uploads/ to uploads/_processed/ safely on Windows.
    Retries + fallback to copy+delete to avoid WinError 2 / file locks.
    """
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    name, ext = os.path.splitext(fname)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(PROCESSED_FOLDER, f"{name}{ext}")

    # ensure unique name
    counter = 0
    while os.path.exists(dst):
        counter += 1
        dst = os.path.join(PROCESSED_FOLDER, f"{name}_{ts}_{counter}{ext}")

    # try move a few times
    for _ in range(4):
        try:
            shutil.move(src_path, dst)    # uses rename then copy+remove
            return dst
        except FileNotFoundError:
            if not os.path.exists(src_path):
                break
            time.sleep(0.25)
        except PermissionError:
            time.sleep(0.5)

    # final fallback: copy then delete
    try:
        shutil.copy2(src_path, dst)
        os.remove(src_path)
        return dst
    except Exception as e:
        app.logger.exception(f"Archive fallback failed for {src_path} -> {dst}: {e}")
        return None


def process_new_documents():
    """
    Incremental build:
      - read new PDFs/TXTs from uploads/
      - extract chunks, embed, add to FAISS (or create)
      - archive processed files safely
    """
    pending = [
        f for f in os.listdir(app.config["UPLOAD_FOLDER"])
        if os.path.isfile(os.path.join(app.config["UPLOAD_FOLDER"], f)) and _allowed_file(f)
    ]

    if not pending:
        return {"message": "No new documents found in uploads."}

    new_chunks = []
    processed_files = []
    for fname in pending:
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        parts = extract_text_from_file(path)
        if parts:
            new_chunks.extend(parts)
            processed_files.append(fname)

    if not new_chunks:
        return {"message": "No valid content extracted from new documents."}

    # Load existing
    index, chunks, embeddings_np, _ = load_embeddings()

    # Encode new chunks
    model = get_embed_model()
    new_emb = model.encode(new_chunks)
    new_emb_np = np.array(new_emb, dtype="float32")

    # (Re)build index as needed
    if index is None:
        dim = int(new_emb_np.shape[1])
        index = faiss.IndexFlatL2(dim)
        index.add(new_emb_np)
        all_chunks = list(new_chunks)
        all_emb_np = new_emb_np
    else:
        # guard against dim mismatch if model ever changes
        dim = index.d
        if dim != int(new_emb_np.shape[1]):
            app.logger.warning("Embedding dimension changed; rebuilding FAISS index.")
            index = faiss.IndexFlatL2(int(new_emb_np.shape[1]))
            base = embeddings_np if embeddings_np is not None else np.zeros((0, new_emb_np.shape[1]), dtype="float32")
            if base.size:
                index.add(base.astype("float32"))
            index.add(new_emb_np)
            all_chunks = chunks + new_chunks
            all_emb_np = np.vstack([base, new_emb_np]) if base.size else new_emb_np
        else:
            index.add(new_emb_np)
            all_chunks = chunks + new_chunks
            all_emb_np = np.vstack([embeddings_np, new_emb_np]) if embeddings_np is not None else new_emb_np

    meta = save_embeddings(all_chunks, all_emb_np, index)

    # archive processed files
    for fname in processed_files:
        src = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        safe_archive_file(src, fname)

    return {
        "message": f"Processed {len(processed_files)} file(s). New chunks: {len(new_chunks)}.",
        "new_chunks": len(new_chunks),
        "total_chunks": meta["num_chunks"],
    }


def retrieve_context(query: str, top_k: int = 5):
    """Return top_k chunks for the query, or [] if not trained yet."""
    try:
        index, chunks, _, _ = load_embeddings()
        if index is None or not chunks:
            return []
        model = get_embed_model()
        q = model.encode([query]).astype("float32")
        k = min(top_k, len(chunks))
        D, I = index.search(q, k)
        return [chunks[i] for i in I[0] if i != -1]
    except Exception as e:
        app.logger.exception(f"retrieve_context error: {e}")
        return []


# ---------- POWERFUL PROMPTS ----------
def make_hybrid_prompt(user_input: str, context_chunks):
    """
    Hybrid prompt:
    - Prefer the user's document CONTEXT when it's clearly relevant.
    - Otherwise, answer normally with your general knowledge (greetings, small talk, etc.).
    - If any part of your answer uses the CONTEXT, add a small section headed 'Based on your documents:' with 1–3 bullet points quoting short phrases.
    - If the answer is NOT in the CONTEXT and you can't answer from it, just answer normally without hallucinating citations.
    - Be concise (3–8 sentences unless asked for more), use plain English, and keep legal tone when appropriate.
    - If information is insufficient, say: 'Information not available in the provided documents.'
    """
    if context_chunks:
        ctx = "\n\n---\n\n".join(context_chunks)[:8000]
        return (
            "You are a helpful legal assistant for Indian law (GST & Income Tax) who can also chat normally.\n"
            "You have TWO sources: (1) your general knowledge and (2) the user's uploaded documents below as CONTEXT.\n"
            "Decide which to use:\n"
            " • If the CONTEXT clearly contains the answer, ground your answer in it.\n"
            " • Otherwise, answer from your general knowledge in a friendly, concise way.\n"
            "When you used the CONTEXT, add at the end a short section headed exactly: 'Based on your documents:'\n"
            "with 1–3 bullet points, each quoting a brief phrase from the CONTEXT that supports your answer.\n"
            "If the answer is not present in the CONTEXT, do not fabricate support. It's okay to say the information is not there.\n\n"
            f"CONTEXT:\n{ctx}\n\nUSER:\n{user_input}\n\nASSISTANT:"
        )
    # No context found → just chat normally
    return user_input


def make_docs_only_prompt(user_input: str, context_chunks):
    """
    Strict docs-only prompt:
    - Answer ONLY from CONTEXT; if not present, say the fixed sentence.
    - Be concise, include section/tribunal references exactly as they appear in CONTEXT when available.
    """
    ctx = "\n\n---\n\n".join(context_chunks)[:8000] if context_chunks else ""
    return (
        "Answer ONLY using the following document context. "
        "If the answer is not present, reply exactly: 'Information not available in the provided documents.'\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION:\n{user_input}\n\nPROFESSIONAL ANSWER:"
    )
# --------------------------------------


# -------------- Helpers -----------------
def login_required(view):
    from functools import wraps
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapper


def gemini_reply_from_history(history_messages, user_input):
    """Call Gemini with chat history; return (reply_text, ok_bool)."""
    if not GEMINI_API_KEY:
        return ("[Gemini not configured] Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env", False)

    gemini_history = []
    for m in history_messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = "user" if m.get("role") == "user" else "model"
        gemini_history.append({"role": role, "parts": [content]})

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        chat_session = model.start_chat(history=gemini_history)
        resp = chat_session.send_message(user_input)
        text = getattr(resp, "text", "") or "[No text returned by Gemini]"
        return (text, True)
    except Exception as e:
        app.logger.exception("Gemini call failed:")
        return (f"[Gemini error] {type(e).__name__}: {e}", False)

# -------------- Routes ------------------
@app.route("/")
def root():
    return redirect(url_for("chat") if session.get("user") else url_for("login"))

# -- Auth --
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = USERS.get(username)
        if user and user["password"] == password:
            session["user"] = {"username": username, "name": user.get("name") or username}
            session.permanent = True
            return redirect(url_for("chat"))
        flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not (name and username and password):
            flash("All fields are required", "error")
        elif username in USERS:
            flash("Username already exists", "error")
        else:
            USERS[username] = {"password": password, "name": name}
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "success")
    return redirect(url_for("login"))

# -- Training --
@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    train_result = None
    if request.method == "POST":
        if "files" in request.files:
            files = request.files.getlist("files")
            uploaded = 0
            for f in files:
                if not f.filename:
                    continue
                if not _allowed_file(f.filename):
                    continue
                f.save(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)))
                uploaded += 1
            if uploaded:
                flash(f"Uploaded {uploaded} file(s).", "success")
            else:
                flash("No valid files uploaded (.pdf or .txt).", "error")
        elif request.form.get("action") == "train":
            if not EMBEDDING_DEPS_OK:
                flash(f"Embedding dependencies missing: {EMBEDDING_IMPORT_ERROR}", "error")
            else:
                train_result = process_new_documents()
                flash(train_result.get("message", "Training finished."), "success")

    # lists for UI
    pending_files = sorted(
        [f for f in os.listdir(app.config["UPLOAD_FOLDER"])
         if os.path.isfile(os.path.join(app.config["UPLOAD_FOLDER"], f)) and _allowed_file(f)]
    )
    processed_files = sorted(
        [f for f in os.listdir(PROCESSED_FOLDER)
         if os.path.isfile(os.path.join(PROCESSED_FOLDER, f)) and _allowed_file(f)]
    )

    _, _, _, embed_meta = (load_embeddings() if EMBEDDING_DEPS_OK else (None, [], None, None))

    return render_template(
        "train.html",
        pending_files=pending_files,
        processed_files=processed_files,
        embed_meta=embed_meta,
        train_result=train_result,
        embedding_deps_ok=EMBEDDING_DEPS_OK,
        embedding_import_error=EMBEDDING_IMPORT_ERROR,
    )

# -- Chat page --
@app.route("/chat")
@login_required
def chat():
    user = session["user"]
    chats = CHATS.get(user["username"], [])
    return render_template("chat.html", user=user, chats=chats)

# -- Chat APIs --
@app.post("/api/chats")
@login_required
def create_chat():
    username = session["user"]["username"]
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}
    initial_title = (data.get("title") or "New chat").strip() or "New chat"

    chat_id = str(uuid4())
    chat = {
        "id": chat_id,
        "title": initial_title,
        "messages": [],
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    CHATS.setdefault(username, []).insert(0, chat)
    return jsonify(chat)


@app.get("/api/chats")
@login_required
def list_chats():
    return jsonify(CHATS.get(session["user"]["username"], []))


@app.get("/api/chats/<chat_id>")
@login_required
def get_chat(chat_id):
    username = session["user"]["username"]
    for c in CHATS.get(username, []):
        if c["id"] == chat_id:
            return jsonify(c)
    return jsonify({"error": "not found"}), 404


def _title_from(text: str, max_len: int = 40) -> str:
    s = " ".join((text or "").strip().split())
    return s[:max_len] + ("…" if len(s) > max_len else "")


@app.post("/api/chats/<chat_id>/message")
@login_required
def send_message(chat_id):
    username = session["user"]["username"]
    data = request.get_json(force=True)
    content = (data or {}).get("content", "").strip()
    mode = (data or {}).get("mode", "hybrid")  # "hybrid" | "docs" | "general"
    if not content:
        return jsonify({"error": "empty message"}), 400

    user_chats = CHATS.get(username, [])
    chat = next((c for c in user_chats if c["id"] == chat_id), None)
    if not chat:
        return jsonify({"error": "chat not found"}), 404

    # user message
    chat["messages"].append({"role": "user", "content": content})

    # set a proper title if still default/new
    if chat.get("title") in (None, "", "New chat"):
        chat["title"] = _title_from(content)

    # Decide prompt based on mode
    if mode == "general":
        prompt = content  # normal Gemini
    elif mode == "docs":
        context_chunks = retrieve_context(content, top_k=5) if EMBEDDING_DEPS_OK else []
        prompt = make_docs_only_prompt(content, context_chunks)
    else:  # hybrid (default)
        context_chunks = retrieve_context(content, top_k=5) if EMBEDDING_DEPS_OK else []
        prompt = make_hybrid_prompt(content, context_chunks)

    # Gemini
    history = chat["messages"][:-1]
    reply, ok = gemini_reply_from_history(history, prompt)

    # assistant reply
    chat["messages"].append({"role": "assistant", "content": reply})
    chat["updated_at"] = datetime.utcnow().isoformat() + "Z"

    # move this chat to top
    try:
        user_chats.remove(chat)
    except ValueError:
        pass
    user_chats.insert(0, chat)

    return jsonify({"reply": reply, "chat": chat, "ok": ok}), (200 if ok else 502)

# -- Health check --
@app.get("/health/gemini")
def health_gemini():
    try:
        if not GEMINI_API_KEY:
            return {"ok": False, "detail": "GEMINI_API_KEY/GOOGLE_API_KEY not set"}, 500
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        r = model.generate_content("ping")
        return {"ok": True, "model": GEMINI_MODEL_NAME, "text": (r.text or "")[:120]}
    except Exception as e:
        app.logger.exception("Health check failed")
        return {"ok": False, "detail": str(e)}, 500

# -- Serve uploads/processed for preview --
@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

@app.route("/processed/<path:filename>")
@login_required
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=False)

# -- Main --
if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
