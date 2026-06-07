from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embedding = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embedding
)

retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})


llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.0
            )


prompt =ChatPromptTemplate.from_template("""
You are a friendly medical assistant.

Rules:
1. If the user greets you (Hi, Hello, Good Morning, etc.), respond politely.
2. If the answer is available in the provided context, answer using the context.
3. If the answer is not available in the context, simply say:
   "I don't know the answer to that question."
4. Do not mention the context or say things like
   "The information is not available in the context."

Context:
{context}

Question:
{input}
""")

chain = prompt | llm | StrOutputParser()


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    docs = retriever.invoke(msg)
    context = "\n\n".join(
        doc.page_content for doc in docs
    )
    response = chain.invoke({
        "context": context,
        "input": msg
    })
    print("Response:", response)
    return response

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8080, debug = True)