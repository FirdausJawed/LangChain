from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------------------
# Step 1: Summarize
# ---------------------------
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this topic in 2 lines:\n{input}"
)

# ---------------------------
# Step 2: Extract keywords
# ---------------------------
keywords_prompt = ChatPromptTemplate.from_template(
    "Extract 5 important keywords from this:\n{input}"
)

# ---------------------------
# Step 3: Final explanation
# ---------------------------
final_prompt = ChatPromptTemplate.from_template(
    """
Using the summary and keywords, explain clearly:

Summary:
{summary}

Keywords:
{keywords}
"""
)

parser = StrOutputParser()

# ---------------------------
# Chains
# ---------------------------
summary_chain = summary_prompt | llm | parser
keywords_chain = keywords_prompt | llm | parser

# ---------------------------
# Main flow
# ---------------------------
print("Multi-step AI system (type 'exit' to quit)\n")

while True:
    user_input = input("Enter topic: ")

    if user_input.lower() == "exit":
        break

    # Step 1
    summary = summary_chain.invoke({"input": user_input})

    # Step 2
    keywords = keywords_chain.invoke({"input": user_input})

    # Step 3
    final_chain = final_prompt | llm | parser

    response = final_chain.invoke({
        "summary": summary,
        "keywords": keywords
    })

    print("\n📌 Summary:\n", summary)
    print("\n🔑 Keywords:\n", keywords)
    print("\n🧠 Final Explanation:\n", response, "\n")