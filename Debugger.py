from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Load error dataset
# ---------------------------
loader = TextLoader("errors.txt")
documents = loader.load()

# ---------------------------
# Split data
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# ---------------------------
# Embeddings + vector store
# ---------------------------
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ---------------------------
# LLM
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------------------
# Prompt (VERY IMPORTANT)
# ---------------------------
prompt = ChatPromptTemplate.from_template("""
You are a senior Java engineer helping debug issues.

Use the context below to answer.

Give response in this format:

1. Error Explanation
2. Root Cause
3. Fix

If not found in context, say:
"I don't know this error from the provided data"

Context:
{context}

Error:
{question}
""")

# ---------------------------
# Helper
# ---------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------
# Main loop
# ---------------------------
print("Debug Assistant Ready (type 'exit' to quit)\n")

while True:
    query = input("Enter error: ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    # print("\n🔍 Retrieved Context:\n")
    # for doc in docs:
    #     print(doc.page_content)
    #     print("------")

    context = format_docs(docs)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": query
    })

    print("\n🛠️ Solution:\n", response, "\n")