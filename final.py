import os
import glob
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings=OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists("fais_db"):
    vs=FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    all_doc=[]
    pdf_file=glob.glob("data/*.pdf")
    for pdf in pdf_file:
        loader=PyMuPDFLoader(pdf)
        final_pdf=loader.load()
        all_doc.extend(final_pdf)

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    chunks=splitter.split_documents(all_doc)
    vs=FAISS.from_documents(
        chunks,
        embeddings
    )
    vs.save_local("fais_db")

retriever=vs.as_retriever(search_kwargs={"k": 3})

llm=OllamaLLM(model="phi3")
prompt=PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer only from the give context. if answer is not found, say i couldn't found that in document
    context: {context}
    question: {question}
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
    print(response)
    retrieved_doc=retriever.invoke(question)
    for doc in retrieved_doc:
        print(f"source: {doc.metadata["source"]}")
        print(f"page: {doc.metadata["page"]}")



