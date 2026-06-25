from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Load the vector database that ingest.py created on disk
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Retriever: given a question, finds the 3 most relevant chunks from the database
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt: tells the LLM to only answer from the retrieved context, not from its own training data
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:""")

# LLM: gpt-4o-mini is fast and cheap; temperature=0 keeps answers factual and consistent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    # Joins the retrieved chunks into one block of text for the prompt
    return "\n\n".join(doc.page_content for doc in docs)

# Chain: question goes in → chunks retrieved → formatted into prompt → LLM → answer string out
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Document Q&A ready. Ask a question or type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        break
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
