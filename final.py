import glob
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings=OllamaEmbeddings(model="nomic-embed-text")
if os.path.exists("faiss_db"):
    vs=FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True

    )
else:
    all_docs = []
    pdf_files = glob.glob("data/*.pdf")

    for pdf in pdf_files:
        loader = PyMuPDFLoader(pdf)
        doc = loader.load()
        all_docs.extend(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    vs = FAISS.from_documents(
        chunks,
        embeddings
    )
    vs.save_local("faiss_db")

retriever=vs.as_retriever()

llm=OllamaLLM(model="phi3")

prompt=PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer ONLY using the provided context.

If answer is not found, say:
"I could not find that in the documents."

Context:
{context}

Question:
{question}
"""
)

chain=(
    {
        "context":retriever,
        "question":RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    question=input("You: ")
    if question.lower()=="quit":
        break

    response=chain.invoke(question)

    print(f"AI: {response}")
