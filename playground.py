import gc
import html as _html
import os
import mammoth
import markdown as _md
import pymupdf4llm
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Playground", layout="wide")
st.title("RAG Playground")
st.caption("Upload a document, add runs, pick a model and prompt per run, compare answers.")

st.markdown("""
<style>
[data-testid="stBaseButton-secondary"] {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
}
[data-testid="stBaseButton-secondary"] p {
    line-height: 1;
    margin: 0;
    text-align: center;
}
/* hide the anchor link icon that appears on hover next to headings */
[data-testid="stHeaderActionElements"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Converters ────────────────────────────────────────────────────────────────
def convert_to_md(src_path: str) -> str:
    """Convert any supported file to Markdown. Returns path to the .md file."""
    base   = os.path.splitext(os.path.basename(src_path))[0]
    md_path = os.path.join(UPLOAD_DIR, base + ".md")
    ext    = os.path.splitext(src_path)[1].lower()

    if ext == ".docx":
        with open(src_path, "rb") as f:
            result = mammoth.convert_to_markdown(f)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result.value)

    elif ext == ".pdf":
        md_text = pymupdf4llm.to_markdown(src_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

    elif ext == ".txt":
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)

    return md_path

# ── Document viewer ───────────────────────────────────────────────────────────
def build_viewer_html(md_text: str, source_docs: list) -> str:
    """Build a two-panel HTML viewer: source cards on the left, highlighted doc on the right."""

    # Find each chunk's position in the raw MD, then insert <mark> tags.
    # Process in reverse position order so earlier insertions don't shift later indices.
    positions = []
    for i, doc in enumerate(source_docs):
        pos = md_text.find(doc.page_content)
        if pos != -1:
            positions.append((pos, i, doc.page_content))
    positions.sort(key=lambda x: -x[0])  # reverse order

    marked = md_text
    for pos, i, chunk in positions:
        marked = (
            marked[:pos]
            + f'<mark id="mark-{i}">'
            + chunk
            + "</mark>"
            + marked[pos + len(chunk):]
        )

    # Convert the MD (with embedded <mark> tags) to HTML.
    # The markdown library passes inline HTML through by default.
    doc_html = _md.markdown(marked, extensions=["tables", "fenced_code"])

    # Build sidebar source cards
    cards_html = ""
    for i, doc in enumerate(source_docs):
        snippet = _html.escape(doc.page_content[:160].replace("\n", " ").strip())
        cards_html += f"""
        <div class="source-card" id="card-{i}" onclick="jumpTo({i})">
            <div class="source-num">Source {i + 1}</div>
            <div class="source-snippet">{snippet}…</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0e1117; color: #fafafa; font-family: "Source Sans Pro", sans-serif; font-size: 14px; }}
  .wrap {{ display: flex; height: 580px; }}

  /* ── left sidebar ── */
  .sidebar {{
    width: 240px; flex-shrink: 0;
    overflow-y: auto; border-right: 1px solid #2d2f3e;
    padding: 14px 12px;
  }}
  .sidebar-title {{ font-size: 11px; font-weight: 700; color: #888; letter-spacing: .08em; margin-bottom: 10px; }}
  .source-card {{
    background: #1a1d2e; border: 1px solid #2d2f3e; border-radius: 8px;
    padding: 10px 12px; margin-bottom: 8px; cursor: pointer;
    transition: border-color .15s, background .15s;
  }}
  .source-card:hover {{ border-color: #f0a500; background: #222540; }}
  .source-card.active {{ border-color: #f0a500; background: #252a3a; }}
  .source-num {{ font-size: 11px; font-weight: 700; color: #f0a500; margin-bottom: 5px; }}
  .source-snippet {{ font-size: 12px; color: #9a9aaa; line-height: 1.5; }}

  /* ── right document panel ── */
  .doc-panel {{ flex: 1; overflow-y: auto; padding: 24px 32px; }}
  .doc-panel h1,.doc-panel h2,.doc-panel h3,.doc-panel h4 {{
    color: #fafafa; margin: 1.2em 0 .5em;
  }}
  .doc-panel p {{ margin: .6em 0; line-height: 1.7; color: #d8d8e8; }}
  .doc-panel ul,.doc-panel ol {{ padding-left: 1.5em; margin: .5em 0; color: #d8d8e8; }}
  .doc-panel li {{ margin: .25em 0; line-height: 1.6; }}
  .doc-panel table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  .doc-panel td,.doc-panel th {{ border: 1px solid #2d2f3e; padding: 6px 12px; }}
  .doc-panel th {{ background: #1a1d2e; font-weight: 600; }}
  .doc-panel code {{ background: #1a1d2e; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
  mark {{
    background: rgba(240, 165, 0, 0.28);
    border-bottom: 2px solid #f0a500;
    color: inherit;
    border-radius: 2px;
    padding: 0 1px;
  }}
  mark.active-mark {{ background: rgba(240, 165, 0, 0.5); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="sidebar">
    <div class="sidebar-title">RETRIEVED SOURCES</div>
    {cards_html}
  </div>
  <div class="doc-panel" id="docPanel">
    {doc_html}
  </div>
</div>
<script>
  function jumpTo(n) {{
    var mark = document.getElementById('mark-' + n);
    if (mark) {{
      mark.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
      document.querySelectorAll('mark').forEach(m => m.classList.remove('active-mark'));
      mark.classList.add('active-mark');
    }}
    document.querySelectorAll('.source-card').forEach(c => c.classList.remove('active'));
    var card = document.getElementById('card-' + n);
    if (card) card.classList.add('active');
  }}
  // Auto-jump to first source on load
  window.onload = function() {{ jumpTo(0); }};
</script>
</body>
</html>"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_md_path() -> str:
    """Return the path to the current document's MD file."""
    return st.session_state.get("current_md_path", "")

def load_full_text() -> str:
    """Read the entire MD file as a string for full-context mode."""
    md_path = get_md_path()
    if md_path and os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# ── Session state ─────────────────────────────────────────────────────────────
if "run_ids" not in st.session_state:
    st.session_state.run_ids       = [0]
    st.session_state.next_id       = 1
if "current_doc" not in st.session_state:
    st.session_state.current_doc    = ""
if "current_md_path" not in st.session_state:
    st.session_state.current_md_path = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Document ──────────────────────────────────────────────────────────────
    st.header("Document")
    doc_label = st.session_state.current_doc or "None — upload a document to begin"
    st.caption(f"Current: **{doc_label}**")

    uploaded = st.file_uploader("Upload a document", type=["docx", "pdf", "txt"])
    if uploaded and uploaded.name != st.session_state.current_doc:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        save_path = os.path.join(UPLOAD_DIR, uploaded.name)

        with st.status("Processing document...", expanded=True) as status:
            st.write("Saving file...")
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            st.write("Converting to Markdown...")
            md_path = convert_to_md(save_path)
            st.write(f"Saved as `{os.path.basename(md_path)}`")

            st.write("Releasing old database...")
            if os.path.exists(CHROMA_DIR):
                old = Chroma(persist_directory=CHROMA_DIR, embedding_function=OpenAIEmbeddings())
                old.delete_collection()
                del old
                gc.collect()

            st.write("Chunking...")
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            raw_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(md_text)
            chunks     = [Document(page_content=c) for c in raw_chunks]
            st.write(f"Created {len(chunks)} chunks.")

            st.write("Embedding and saving to database...")
            Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DIR)
            status.update(label=f"Ready: {uploaded.name}", state="complete")

        st.session_state.current_doc     = uploaded.name
        st.session_state.current_md_path = md_path
        st.rerun()

    st.divider()

    # ── Runs ──────────────────────────────────────────────────────────────────
    st.header("Runs")

    runs = []
    for run_id in st.session_state.run_ids:
        with st.container(border=True):
            label_col, btn_col = st.columns([5, 1])
            label_col.markdown(f"**Run {run_id + 1}**")
            if len(st.session_state.run_ids) > 1:
                if btn_col.button("－", key=f"remove_{run_id}", use_container_width=True):
                    st.session_state.run_ids.remove(run_id)
                    st.rerun()

            model = st.selectbox("Model", options=MODELS, key=f"model_{run_id}")
            mode  = st.radio(
                "Context mode",
                options=["RAG", "Full document"],
                key=f"mode_{run_id}",
                horizontal=True,
            )
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

# ── Main: question input ──────────────────────────────────────────────────────
# Tab key fills in the placeholder text (uses React's native value setter so Streamlit picks it up)
components.html("""
<script>
(function() {
    function attach(input) {
        input.addEventListener('keydown', function(e) {
            if (e.key !== 'Tab' || input.value.length > 0) return;
            e.preventDefault();
            var text = input.placeholder.replace(/^e\\.g\\. /, '');
            var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
            setter.call(input, text);
            input.dispatchEvent(new Event('input', { bubbles: true }));
        });
    }
    var done = false;
    var iv = setInterval(function() {
        var inputs = parent.document.querySelectorAll('input[type="text"]');
        inputs.forEach(function(inp) {
            if (inp.placeholder && inp.placeholder.startsWith('e.g.') && !inp._tab) {
                inp._tab = true;
                attach(inp);
                done = true;
            }
        });
        if (done) clearInterval(iv);
    }, 300);
    setTimeout(function() { clearInterval(iv); }, 8000);
})();
</script>
""", height=0)

with st.form("query_form"):
    question   = st.text_input("Your question", placeholder="e.g. How does a patient book a doctor appointment?")
    run_button = st.form_submit_button("Run all", type="primary")

# ── Run all chains ────────────────────────────────────────────────────────────
if run_button and question:

    needs_rag = any(r["mode"] == "RAG" for r in runs)
    if needs_rag:
        if not os.path.exists(CHROMA_DIR):
            st.error("No document ingested yet. Upload a document first.")
            st.stop()
        embeddings  = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    for i, run in enumerate(runs):
        llm    = ChatOpenAI(model=run["model"], temperature=0)
        prompt = ChatPromptTemplate.from_template(run["prompt"])

        mode_label = "RAG" if run["mode"] == "RAG" else "Full doc"
        st.markdown(f"**Run {i + 1} — {run['model']} — {mode_label}**")

        with st.spinner(f"Running {run['model']} ({mode_label})..."):
            with get_openai_callback() as cb:
                if run["mode"] == "RAG":
                    source_docs = retriever.invoke(question)
                    context     = format_docs(source_docs)
                    messages    = prompt.format_messages(context=context, question=question)
                    answer      = (llm | StrOutputParser()).invoke(messages)
                else:
                    source_docs  = []
                    full_context = load_full_text()
                    messages     = prompt.format_messages(context=full_context, question=question)
                    answer       = (llm | StrOutputParser()).invoke(messages)

        st.write(answer)

        if source_docs:
            md_text = load_full_text()
            if md_text:
                st.caption("Document viewer — click a source card to jump to the highlighted passage")
                components.html(build_viewer_html(md_text, source_docs), height=600, scrolling=False)
            else:
                # Fallback: no MD file on disk yet (doc was ingested before this feature)
                with st.expander(f"Sources ({len(source_docs)} chunks retrieved)"):
                    for j, doc in enumerate(source_docs):
                        st.markdown(f"**Source {j + 1}**")
                        st.text(doc.page_content)
                        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens", cb.prompt_tokens)
        col2.metric("Completion tokens", cb.completion_tokens)
        col3.metric("Cost", f"${cb.total_cost:.6f}")

        if i < len(runs) - 1:
            st.divider()

elif run_button and not question:
    st.warning("Enter a question first.")
