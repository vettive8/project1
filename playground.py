import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Playground", layout="wide")
st.title("RAG Playground")
st.caption("Ask a question about your document. Edit the prompt to change how the AI responds.")

# ── Sidebar: prompt editor ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Prompt Template")
    prompt_text = st.text_area(
        label="Edit the instructions the AI receives:",
        value=(
            "You are a helpful assistant. Answer the question using only the context below.\n"
            "If the answer is not in the context, say \"I don't know based on the provided documents.\"\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        height=300,
    )

# ── Main: question input ──────────────────────────────────────────────────────
question = st.text_input("Your question", placeholder="e.g. How does a patient book a doctor appointment?")
run = st.button("Run", type="primary")

# ── Run the RAG chain ─────────────────────────────────────────────────────────
if run and question:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(prompt_text)
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )

    with st.spinner("Thinking..."):
        answer = chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)

elif run and not question:
    st.warning("Enter a question first.")
