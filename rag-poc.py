from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import openai

def load_documents():
    return [
        Document(page_content="""Política de escalado: si un ticket no recibe
        respuesta en 24h, se escala automáticamente al team lead."""),

        Document(page_content="""Proceso de despliegue: todos los despliegues
        a producción requieren aprobación de al menos 2 reviewers.""")
    ]

def printChunks(chunks):
        print(f"Total chunks created: {len(chunks)}")
        print(f"Characters in first chunk: {len(chunks[0].page_content)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {chunk.page_content}")

def buildPrompt(query: str, context: str) -> str:
     # Build prompt for LLM with retrieved context
    return f"""Answer just basing ONLY in the following context.
    If not in the context, say 'I don't have that information'.
    Context: {context}
    Question: {query}"""
        
def main():

    # PHASE 1: Indexation
    # Our documents may come from PDFLoader, ConfluenceLoader, etc
    docs = load_documents()


    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    printChunks(chunks)

    # Create embeddings and store in Chroma
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embedding_model)


    # PHASE 2: Querying
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    query = input("Enter your question: ")
    results_vector = retriever.invoke(query)
    #print(f"first result from vector retriever: {results_vector[0].page_content}")

    # Reasoning with LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    company_context = "\n\n".join([d.page_content for d in results_vector])
    prompt = buildPrompt(query, company_context)
    #print(f"Prompt sent to LLM: {prompt}")

    llm_response = llm.invoke(prompt)
    print(f"LLM response: {llm_response.content}")

    

if __name__ == "__main__":
    main()