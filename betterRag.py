from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# 1. Load Data
# ---------------------------
loader = TextLoader("data.txt")
documents = loader.load()

# ---------------------------
# 2. Better Chunking
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# ---------------------------
# 3. Embeddings
# ---------------------------
embeddings = OpenAIEmbeddings()

# ---------------------------
# 4. Vector Store
# ---------------------------
vectorstore = FAISS.from_documents(chunks, embeddings)

# ---------------------------
# 5. Retriever (tuned)
# ---------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------
# 6. LLM
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------------------
# 7. Prompt (strict grounding)
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the context below to answer the question.

If the answer is not in the context, say:
"I don't know from the provided data."

Context:
{context}

Question: {question}

Answer:
""")

# ---------------------------
# Helper: format docs
# ---------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------
# 8. Main Loop
# ---------------------------
print("Better RAG system ready (type 'exit' to quit)\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    # ---------------------------
    # Retrieve documents
    # ---------------------------
    docs = retriever.invoke(query)

    print("\n🔍 Retrieved Context:\n")
    for doc in docs:
        print(doc.page_content)
        print("------")

    context = format_docs(docs)

    # ---------------------------
    # Generate answer
    # ---------------------------
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    # ---------------------------
    # Fallback (hybrid mode)
    # ---------------------------
    if "I don't know" in response:
        fallback = llm.invoke(query).content
        print("\n⚠️ Fallback Answer:\n", fallback, "\n")
    else:
        print("\n🤖 Answer:\n", response, "\n")