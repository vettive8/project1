import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
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

MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Playground", layout="wide")
st.title("RAG Playground")
st.caption("Add runs, pick a model and prompt per run, ask one question — compare answers side by side.")

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
</style>
""", unsafe_allow_html=True)

# ── Session state: stable list of run IDs ─────────────────────────────────────
# Using unique IDs (not indexes) so removing one run doesn't mess up the others
if "run_ids" not in st.session_state:
    st.session_state.run_ids = [0]
    st.session_state.next_id = 1

# ── Sidebar: run cards ────────────────────────────────────────────────────────
with st.sidebar:
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

            model = st.selectbox(
                "Model",
                options=MODELS,
                key=f"model_{run_id}",
            )
            prompt = st.text_area(
                "Prompt",
                value=DEFAULT_PROMPT,
                height=220,
                key=f"prompt_{run_id}",
            )

        runs.append({"model": model, "prompt": prompt})

    # Add button lives at the bottom, after all cards
    st.divider()
    if len(st.session_state.run_ids) < 8:
        if st.button("＋ Add run", use_container_width=True):
            st.session_state.run_ids.append(st.session_state.next_id)
            st.session_state.next_id += 1
            st.rerun()
    else:
        st.caption("Maximum 8 runs reached.")

# ── Main: question input ──────────────────────────────────────────────────────
question = st.text_input("Your question", placeholder="e.g. How does a patient book a doctor appointment?")
run_button = st.button("Run all", type="primary")

# ── Run all chains and display side by side ───────────────────────────────────
if run_button and question:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    for i, run in enumerate(runs):
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(run["prompt"])
            | ChatOpenAI(model=run["model"], temperature=0)
            | StrOutputParser()
        )

        st.markdown(f"**Run {i + 1} — {run['model']}**")
        with st.spinner(f"Running {run['model']}..."):
            with get_openai_callback() as cb:
                answer = chain.invoke(question)

        st.write(answer)

        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt tokens", cb.prompt_tokens)
        col2.metric("Completion tokens", cb.completion_tokens)
        col3.metric("Cost", f"${cb.total_cost:.6f}")

        if i < len(runs) - 1:
            st.divider()

elif run_button and not question:
    st.warning("Enter a question first.")
