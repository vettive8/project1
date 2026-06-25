import gc  # noqa: F401 (kept for potential future use)
import html as _html
import io
import json
import os
import re
import chromadb
import mammoth
import markdown as _md
import pymupdf4llm
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DEFAULT_PROMPT = (
    "You are a helpful assistant. Answer the question using only the context below.\n"
    "If the answer is not in the context, say \"I don't know based on the provided documents.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

MODELS     = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploaded_docs"
REGISTRY_F = "doc_registry.json"
HISTORY_F  = "history.json"
STATE_F    = "app_state.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Testing Labs", layout="wide")

st.markdown("""<style>
[data-testid="stBaseButton-secondary"] {
    display: flex; align-items: center; justify-content: center;
    width: 100%; height: 100%;
}
[data-testid="stBaseButton-secondary"] p { line-height: 1; margin: 0; text-align: center; }
[data-testid="stHeaderActionElements"] { display: none; }
</style>""", unsafe_allow_html=True)

# ── Persistence helpers ───────────────────────────────────────────────────────
def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_registry():
    return load_json(REGISTRY_F, [])

def save_app_state():
    runs_data = [
        {
            "run_id": rid,
            "model":  st.session_state.get(f"model_{rid}", MODELS[0]),
            "mode":   st.session_state.get(f"mode_{rid}", "RAG"),
            "prompt": st.session_state.get(f"prompt_{rid}", DEFAULT_PROMPT),
        }
        for rid in st.session_state.get("run_ids", [])
    ]
    save_json(STATE_F, {
        "active_docs": st.session_state.get("active_docs", []),
        "next_id":     st.session_state.get("next_id", 1),
        "runs":        runs_data,
    })

# ── Converters ────────────────────────────────────────────────────────────────
def html_body_to_text(html_body: str) -> str:
    """Extract plain text from an HTML body fragment for RAG chunking.
    Adds newlines at block boundaries so the resulting text mirrors what the
    browser renders and what mark.js will find in the DOM text nodes."""
    text = re.sub(r'<br\s*/?>', '\n', html_body, flags=re.I)
    text = re.sub(r'</(p|h[1-6]|li|tr|blockquote)>', '\n', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)          # strip all remaining tags
    text = _html.unescape(text)                  # decode &amp; &lt; etc.
    text = re.sub(r'[ \t]+', ' ', text)          # collapse horizontal whitespace
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text) # clean line margins
    text = re.sub(r'\n{3,}', '\n\n', text)       # max one blank line
    return text.strip()


def convert_to_md(src_path: str):
    """Convert DOCX / PDF / TXT to plain text + HTML for viewing.
    Returns (md_path, html_path).
    md_path stores plain text derived from the HTML body so chunk text
    matches DOM text nodes and mark.js can highlight them.
    html_path stores the rendered HTML used in the viewer."""
    base      = os.path.splitext(os.path.basename(src_path))[0]
    md_path   = os.path.join(UPLOAD_DIR, base + ".md")
    html_path = os.path.join(UPLOAD_DIR, base + ".html")
    ext       = os.path.splitext(src_path)[1].lower()

    if ext == ".docx":
        with open(src_path, "rb") as f:
            raw = f.read()
        html_body = mammoth.convert_to_html(io.BytesIO(raw)).value
        # Derive plain text from HTML so chunks match the viewer's DOM text
        plain_text = html_body_to_text(html_body)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_body)

    elif ext == ".pdf":
        md_text   = pymupdf4llm.to_markdown(src_path)
        html_body = _md.markdown(md_text, extensions=["tables", "fenced_code"])
        plain_text = html_body_to_text(html_body)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_body)

    elif ext == ".txt":
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<pre style='white-space:pre-wrap'>{_html.escape(text)}</pre>")

    return md_path, html_path

# ── ChromaDB helpers ──────────────────────────────────────────────────────────
def collection_name_for(filename: str) -> str:
    """Sanitize a filename into a valid ChromaDB collection name (≤63 chars)."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name or "doc")[:63]

def ingest_document(src_path: str, filename: str, status_obj) -> dict:
    """Convert → chunk → embed into a named collection. Returns a registry entry dict."""
    status_obj.write("Converting document…")
    md_path, html_path = convert_to_md(src_path)
    status_obj.write(f"Saved `{os.path.basename(md_path)}` + HTML viewer")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    col_name = collection_name_for(filename)
    chunks   = [Document(page_content=c) for c in
                RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(md_text)]
    status_obj.write(f"Created {len(chunks)} chunks.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(col_name)   # re-ingest: drop old version
    except Exception:
        pass

    status_obj.write("Embedding and saving…")
    Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"), client=client, collection_name=col_name)

    return {
        "filename":    filename,
        "collection":  col_name,
        "md_path":     md_path,
        "html_path":   html_path,
        "chunks":      len(chunks),
        "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

def retrieve_docs(question: str, collection_names: list, k: int = 3) -> list:
    """Query one or more ChromaDB collections, merge by relevance score."""
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    all_scored = []
    for col in collection_names:
        try:
            vs = Chroma(client=client, collection_name=col, embedding_function=embeddings)
            for doc, score in vs.similarity_search_with_score(question, k=k):
                doc.metadata["_collection"] = col
                all_scored.append((score, doc))
        except Exception:
            pass

    all_scored.sort(key=lambda x: x[0])   # lower L2 = more relevant
    seen, unique = set(), []
    for _, doc in all_scored:
        key = doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    # Always return the global top-k, not k-per-doc.
    # If the answer only lives in one doc, all k chunks should come from there.
    return unique[:k]

# ── Text helpers ──────────────────────────────────────────────────────────────
def format_docs(docs: list) -> str:
    return "\n\n".join(d.page_content for d in docs)

def load_full_text(active_entries: list) -> str:
    """Read and concatenate MD files for all active documents."""
    parts = []
    for e in active_entries:
        p = e.get("md_path", "")
        if p and os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                content = f.read()
            if len(active_entries) > 1:
                parts.append(f"## Document: {e['filename']}\n\n{content}")
            else:
                parts.append(content)
    return "\n\n---\n\n".join(parts)

# ── Document viewer ───────────────────────────────────────────────────────────
def build_viewer_html(html_content: str, source_docs: list) -> str:
    """Two-panel viewer: source cards left, full document right.
    html_content is the rendered document body (from mammoth or markdown lib).
    mark.js highlights the retrieved passages directly in the HTML DOM."""
    chunks_json = json.dumps([doc.page_content for doc in source_docs])

    cards_html = ""
    for i, doc in enumerate(source_docs):
        coll    = doc.metadata.get("_collection", "")
        snippet = _html.escape(doc.page_content[:160].replace("\n", " ").strip())
        cards_html += f"""
        <div class="source-card" id="card-{i}" onclick="jumpTo({i})">
            <div class="source-num">Source {i + 1}{f" · {coll}" if coll else ""}</div>
            <div class="source-snippet">{snippet}…</div>
        </div>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/mark.js/8.11.1/mark.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0e1117;color:#fafafa;font-family:"Source Sans Pro",sans-serif;font-size:14px}}
  .wrap{{display:flex;height:580px}}
  .sidebar{{width:240px;flex-shrink:0;overflow-y:auto;border-right:1px solid #2d2f3e;padding:14px 12px}}
  .sidebar-title{{font-size:11px;font-weight:700;color:#888;letter-spacing:.08em;margin-bottom:10px}}
  .source-card{{background:#1a1d2e;border:1px solid #2d2f3e;border-radius:8px;
    padding:10px 12px;margin-bottom:8px;cursor:pointer;transition:border-color .15s,background .15s}}
  .source-card:hover{{border-color:#f0a500;background:#222540}}
  .source-card.active{{border-color:#f0a500;background:#252a3a}}
  .source-num{{font-size:11px;font-weight:700;color:#f0a500;margin-bottom:5px}}
  .source-snippet{{font-size:12px;color:#9a9aaa;line-height:1.5}}
  .doc-panel{{flex:1;overflow-y:auto;padding:24px 32px}}
  .doc-panel h1,.doc-panel h2,.doc-panel h3,.doc-panel h4{{color:#fafafa;margin:1.2em 0 .5em}}
  .doc-panel p{{margin:.6em 0;line-height:1.7;color:#d8d8e8}}
  .doc-panel ul,.doc-panel ol{{padding-left:1.5em;margin:.5em 0;color:#d8d8e8}}
  .doc-panel li{{margin:.25em 0;line-height:1.6}}
  .doc-panel table{{border-collapse:collapse;width:100%;margin:1em 0}}
  .doc-panel td,.doc-panel th{{border:1px solid #3a3d50;padding:8px 14px;color:#d8d8e8}}
  .doc-panel thead td,.doc-panel th{{background:#1a1d2e;font-weight:700;color:#fafafa}}
  .doc-panel tr:nth-child(even){{background:#13151f}}
  .doc-panel code{{background:#1a1d2e;padding:2px 6px;border-radius:4px;font-size:13px}}
  .doc-panel strong{{color:#fafafa}}
  mark{{background:rgba(240,165,0,.3);border-bottom:2px solid #f0a500;
    color:inherit;border-radius:2px;padding:0 1px}}
  mark.active-mark{{background:rgba(240,165,0,.55)}}
</style></head><body>
<div class="wrap">
  <div class="sidebar">
    <div class="sidebar-title">RETRIEVED SOURCES</div>{cards_html}
  </div>
  <div class="doc-panel" id="docPanel">{html_content}</div>
</div>
<script>
var chunks = {chunks_json};
var markIds = {{}};

function initHighlights() {{
  var instance = new Mark(document.getElementById('docPanel'));
  chunks.forEach(function(text, i) {{
    // Normalize whitespace — DOM text node concatenation can vary
    var searchText = text.replace(/\\s+/g, ' ').trim();
    instance.mark(searchText, {{
      acrossElements: true,
      separateWordSearch: false,
      each: function(el) {{
        if (!markIds[i]) {{ markIds[i] = true; el.id = 'mark-' + i; }}
      }}
    }});
  }});
}}

function jumpTo(n) {{
  var m = document.getElementById('mark-' + n);
  if (m) {{
    m.scrollIntoView({{behavior:'smooth', block:'center'}});
    document.querySelectorAll('mark').forEach(x => x.classList.remove('active-mark'));
    m.classList.add('active-mark');
  }}
  document.querySelectorAll('.source-card').forEach(x => x.classList.remove('active'));
  var c = document.getElementById('card-' + n);
  if (c) c.classList.add('active');
}}

window.onload = function() {{ initHighlights(); setTimeout(function(){{jumpTo(0);}}, 150); }};
</script></body></html>"""

# ── Session state: one-time init ──────────────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.initialized = True

    saved    = load_json(STATE_F, {})
    registry = load_registry()

    # Restore active doc selection
    valid   = {r["filename"] for r in registry}
    initial = [d for d in saved.get("active_docs", []) if d in valid]
    st.session_state.active_docs = initial
    for r in registry:
        st.session_state[f"doc_{r['collection']}"] = r["filename"] in initial

    # Restore run configs (model / mode / prompt per run)
    saved_runs = saved.get("runs", [])
    if saved_runs:
        st.session_state.run_ids = [r["run_id"] for r in saved_runs]
        st.session_state.next_id = saved.get("next_id", max(r["run_id"] for r in saved_runs) + 1)
        for r in saved_runs:
            st.session_state[f"model_{r['run_id']}"]  = r.get("model",  MODELS[0])
            st.session_state[f"mode_{r['run_id']}"]   = r.get("mode",   "RAG")
            st.session_state[f"prompt_{r['run_id']}"] = r.get("prompt", DEFAULT_PROMPT)
    else:
        st.session_state.run_ids = [0]
        st.session_state.next_id = 1

# Apply "restore from history" if triggered (runs before sidebar renders checkboxes)
if "restore_docs" in st.session_state:
    restore_docs = st.session_state.pop("restore_docs")
    registry     = load_registry()
    valid        = {r["filename"] for r in registry}
    restore_docs = [d for d in restore_docs if d in valid]
    st.session_state.active_docs = restore_docs
    for r in registry:
        st.session_state[f"doc_{r['collection']}"] = r["filename"] in restore_docs
    save_app_state()

# ── Title + History popover ───────────────────────────────────────────────────
title_col, hist_col = st.columns([7, 1])
with title_col:
    st.title("RAG Testing Labs")
    st.caption("Upload documents, configure runs, compare answers side by side.")
with hist_col:
    st.write("")   # vertical alignment nudge
    with st.popover("🕑 History", use_container_width=True):
        history = load_json(HISTORY_F, [])
        if not history:
            st.caption("No history yet. Run a query to start.")
        else:
            st.caption(f"{len(history)} saved queries")
            for item in reversed(history[-30:]):
                with st.container(border=True):
                    docs_label = ", ".join(item.get("docs", []))
                    st.caption(f"{item['timestamp']} · {docs_label}")
                    q_label = item["question"][:80] + ("…" if len(item["question"]) > 80 else "")
                    if st.button(q_label, key=f"h_{item['id']}", use_container_width=True):
                        st.session_state.question_input = item["question"]
                        st.session_state.restore_docs   = item.get("docs", [])
                        st.session_state.history_view   = item
                        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Document Library ──────────────────────────────────────────────────────
    st.header("Document Library")
    registry = load_registry()

    if not registry:
        st.caption("No documents yet. Upload one below.")
    else:
        new_active = []
        for entry in registry:
            row_a, row_b = st.columns([5, 1])
            is_checked = row_a.checkbox(
                entry["filename"],
                key=f"doc_{entry['collection']}",
                help=f"{entry['chunks']} chunks · ingested {entry['ingested_at']}",
            )
            if is_checked:
                new_active.append(entry["filename"])
            if row_b.button("×", key=f"del_{entry['collection']}", help="Remove from library"):
                try:
                    chromadb.PersistentClient(path=CHROMA_DIR).delete_collection(entry["collection"])
                except Exception:
                    pass
                updated = [r for r in registry if r["filename"] != entry["filename"]]
                save_json(REGISTRY_F, updated)
                if entry["filename"] in st.session_state.active_docs:
                    st.session_state.active_docs.remove(entry["filename"])
                save_app_state()
                st.rerun()

        if sorted(new_active) != sorted(st.session_state.active_docs):
            st.session_state.active_docs = new_active
            save_app_state()

    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload new document", type=["docx", "pdf", "txt"])
    if uploaded:
        existing = {r["filename"] for r in registry}
        if uploaded.name in existing:
            st.caption(f"'{uploaded.name}' is already in the library. Ingest again to refresh.")
        if st.button("Ingest", type="primary", use_container_width=True):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            save_path = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            with st.status(f"Processing {uploaded.name}…", expanded=True) as status:
                entry   = ingest_document(save_path, uploaded.name, status)
                new_reg = [r for r in registry if r["filename"] != uploaded.name]
                new_reg.append(entry)
                save_json(REGISTRY_F, new_reg)
                # Auto-select the newly ingested doc
                if uploaded.name not in st.session_state.active_docs:
                    st.session_state.active_docs.append(uploaded.name)
                    save_app_state()
                status.update(label=f"Ready: {uploaded.name}", state="complete")
            st.rerun()

    st.divider()

    st.divider()

    # ── Runs ──────────────────────────────────────────────────────────────────
    st.header("Runs")
    runs = []
    for run_id in st.session_state.run_ids:
        with st.container(border=True):
            lc, bc = st.columns([5, 1])
            lc.markdown(f"**Run {run_id + 1}**")
            if len(st.session_state.run_ids) > 1:
                if bc.button("－", key=f"remove_{run_id}", use_container_width=True):
                    st.session_state.run_ids.remove(run_id)
                    st.rerun()
            model  = st.selectbox("Model", options=MODELS, key=f"model_{run_id}")
            mode   = st.radio("Context mode", options=["RAG", "Full document"],
                              key=f"mode_{run_id}", horizontal=True)
            prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=200, key=f"prompt_{run_id}")
        runs.append({"model": model, "mode": mode, "prompt": prompt})

    st.divider()
    if len(st.session_state.run_ids) < 8:
        if st.button("＋ Add run", use_container_width=True):
            st.session_state.run_ids.append(st.session_state.next_id)
            st.session_state.next_id += 1
            st.rerun()
    else:
        st.caption("Maximum 8 runs reached.")

# Save run configs after sidebar renders so all widget values are captured
save_app_state()

# ── Query form ────────────────────────────────────────────────────────────────
components.html("""<script>
(function(){
  function attach(inp){
    inp.addEventListener('keydown',function(e){
      if(e.key!=='Tab'||inp.value.length>0)return;
      e.preventDefault();
      var text=inp.placeholder.replace(/^e\\.g\\. /,'');
      var setter=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;
      setter.call(inp,text);inp.dispatchEvent(new Event('input',{bubbles:true}));
    });
  }
  var done=false,iv=setInterval(function(){
    parent.document.querySelectorAll('input[type="text"]').forEach(function(inp){
      if(inp.placeholder&&inp.placeholder.startsWith('e.g.')&&!inp._tab){
        inp._tab=true;attach(inp);done=true;}
    });
    if(done)clearInterval(iv);
  },300);
  setTimeout(function(){clearInterval(iv);},8000);
})();
</script>""", height=0)

# Example questions and history clicks both write directly to the widget key
# so the value survives the rerun that happens before form submission.
if "restore_question" in st.session_state:
    st.session_state.question_input = st.session_state.pop("restore_question")

EXAMPLE_QUESTIONS = [
    "How does a patient book a doctor appointment?",
    "What payment methods are supported?",
    "How does the video consultation work?",
    "What is the symptom checker?",
    "What are the admin features?",
    "What is this document about?",
]

st.caption("Try an example:")
ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
for i, eq in enumerate(EXAMPLE_QUESTIONS):
    if ex_cols[i].button(eq, key=f"ex_{i}", use_container_width=True):
        st.session_state.question_input = eq
        st.rerun()

with st.form("query_form"):
    question   = st.text_input("Your question", key="question_input",
                               placeholder="Type a question or click an example above")
    run_button = st.form_submit_button("Run all", type="primary")

# ── Active document resolution ────────────────────────────────────────────────
registry       = load_registry()
active_entries = [r for r in registry if r["filename"] in st.session_state.active_docs]

if run_button:
    if not question:
        st.warning("Enter a question first.")
        st.stop()
    if not active_entries:
        st.error("Select at least one document from the Document Library in the sidebar.")
        st.stop()

    active_colls = [e["collection"] for e in active_entries]
    if "history_view" in st.session_state:
        del st.session_state["history_view"]   # fresh run clears any saved view

    run_results = []   # collect each run's output to save in history

    for i, run in enumerate(runs):
        llm        = ChatOpenAI(model=run["model"], temperature=0)
        prompt_tpl = ChatPromptTemplate.from_template(run["prompt"])
        mode_label = "RAG" if run["mode"] == "RAG" else "Full doc"
        doc_label  = ", ".join(e["filename"] for e in active_entries)
        run_label  = f"Run {i + 1} — {run['model']} — {mode_label} — {doc_label}"

        st.markdown(f"**{run_label}**")

        with st.spinner(f"Running {run['model']} ({mode_label})…"):
            with get_openai_callback() as cb:
                if run["mode"] == "RAG":
                    source_docs = retrieve_docs(question, active_colls)
                    context     = format_docs(source_docs)
                    messages    = prompt_tpl.format_messages(context=context, question=question)
                    answer      = (llm | StrOutputParser()).invoke(messages)
                else:
                    source_docs  = []
                    full_context = load_full_text(active_entries)
                    messages     = prompt_tpl.format_messages(context=full_context, question=question)
                    answer       = (llm | StrOutputParser()).invoke(messages)

        st.write(answer)

        if source_docs:
            # Group chunks by which document they came from, then show a
            # viewer per doc. If all chunks landed in one doc (common even
            # when multiple docs are selected), only one viewer appears.
            chunks_by_col = {}
            for doc in source_docs:
                col = doc.metadata.get("_collection", "")
                chunks_by_col.setdefault(col, []).append(doc)

            for col, col_docs in chunks_by_col.items():
                entry = next((e for e in active_entries if e["collection"] == col), None)
                if not entry:
                    continue
                # Prefer pre-rendered HTML (preserves tables); fall back to md→html
                viewer_path = entry.get("html_path", "") or ""
                if not viewer_path or not os.path.exists(viewer_path):
                    viewer_path = entry.get("md_path", "")
                    use_md = True
                else:
                    use_md = False
                if viewer_path and os.path.exists(viewer_path):
                    with open(viewer_path, encoding="utf-8") as f:
                        raw = f.read()
                    html_content = _md.markdown(raw, extensions=["tables", "fenced_code"]) if use_md else raw
                    if len(chunks_by_col) > 1:
                        st.caption(f"**{entry['filename']}**")
                    st.caption("Click a source card to jump to the highlighted passage.")
                    components.html(build_viewer_html(html_content, col_docs), height=600, scrolling=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens",     cb.prompt_tokens)
        col2.metric("Completion tokens", cb.completion_tokens)
        col3.metric("Cost",              f"${cb.total_cost:.6f}")

        run_results.append({
            "label":             run_label,
            "answer":            answer,
            "prompt_tokens":     cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "cost":              cb.total_cost,
            # Serialized so we can rebuild the viewer from history without re-querying
            "source_chunks": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in source_docs
            ],
        })

        if i < len(runs) - 1:
            st.divider()

    # ── Persist to history (with answers) ────────────────────────────────────
    history = load_json(HISTORY_F, [])
    history.append({
        "id":        f"{len(history):06d}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "docs":      [e["filename"] for e in active_entries],
        "question":  question,
        "runs":      run_results,
        # Store paths so viewer can be rebuilt from history
        "active_entries": [
            {"filename": e["filename"], "md_path": e["md_path"],
             "html_path": e.get("html_path", "")}
            for e in active_entries
        ],
    })
    save_json(HISTORY_F, history[-50:])

elif "history_view" in st.session_state:
    # ── Show saved answers from history (no API call needed) ─────────────────
    item = st.session_state.history_view
    st.info(f"Saved answer · {item['timestamp']} · {', '.join(item.get('docs', []))}")

    saved_runs     = item.get("runs", [])
    saved_entries  = item.get("active_entries", [])
    single_doc     = len(saved_entries) == 1

    if not saved_runs:
        st.caption("This entry was saved before answer storage was added. Re-run to get a fresh answer.")
    else:
        for j, r in enumerate(saved_runs):
            st.markdown(f"**{r['label']}**")
            st.write(r["answer"])

            # Rebuild the viewer — same grouped logic as the live run
            chunks = r.get("source_chunks", [])
            if chunks:
                chunks_by_col = {}
                for c in chunks:
                    col = c.get("metadata", {}).get("_collection", "")
                    chunks_by_col.setdefault(col, []).append(c)

                for col, col_chunks in chunks_by_col.items():
                    entry = next((e for e in saved_entries if e.get("collection") == col or col == ""), None)
                    if not entry and saved_entries:
                        entry = saved_entries[0]
                    if not entry:
                        continue
                    viewer_path = entry.get("html_path", "") or ""
                    if not viewer_path or not os.path.exists(viewer_path):
                        viewer_path = entry.get("md_path", "")
                        use_md = True
                    else:
                        use_md = False
                    if viewer_path and os.path.exists(viewer_path):
                        with open(viewer_path, encoding="utf-8") as f:
                            raw = f.read()
                        html_content = _md.markdown(raw, extensions=["tables", "fenced_code"]) if use_md else raw
                        rebuilt = [Document(page_content=c["page_content"],
                                            metadata=c.get("metadata", {})) for c in col_chunks]
                        if len(chunks_by_col) > 1:
                            st.caption(f"**{entry.get('filename', '')}**")
                        st.caption("Click a source card to jump to the highlighted passage.")
                        components.html(build_viewer_html(html_content, rebuilt), height=600, scrolling=False)

            c1, c2, c3 = st.columns(3)
            c1.metric("Prompt tokens",     r["prompt_tokens"])
            c2.metric("Completion tokens", r["completion_tokens"])
            c3.metric("Cost",              f"${r['cost']:.6f}")
            if j < len(saved_runs) - 1:
                st.divider()
