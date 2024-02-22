import os
import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

####################################################################

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)


#######################################################################


# Initialize LangChain
def conversational_rag_chain(retriever):
    # Initialize the LLM
    llm = GoogleGenerativeAI(model="gemini-pro")
    # Initialize the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}\n\n"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    # Initialize the chain
    stuff_document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    return create_retrieval_chain(retriever, stuff_document_chain)

# Initialize the vector store
def get_vectorstore_from_url(url):
        # Load the web page
        loader = WebBaseLoader(url)
        document = loader.load()

        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.split_documents(document)
        
        # Initialize the vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                                google_api_key=google_api_key)
        db = Chroma.from_documents(chunks, embeddings)
        return db

    # Initialize the retriever
def get_context_retriever_chain(db):
    # Initialize the retriever
    retriever = db.as_retriever()
    
    # Initialize the prompt
    system = "Given the above chat history and conversation, answer the question based on the context.\
    If you don't know the answer, just say you don't know. Don't make up an answer."
    prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("system", system),
    ("human", "{input}")
    ])
    
    # Initialize the chain
    llm = GoogleGenerativeAI(model="gemini-pro")
    chain = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=prompt
    )

    return chain

    # Get response
def get_response(user_query):
    # Conversational RAG Chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    
    conversational = conversational_rag_chain(retriever_chain)
    
    # Get response
    response = conversational.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    
    })
    
    return response['answer']



    ##########################################################################


def main():
    # Set page config
    st.set_page_config(
        page_title="Chat with website",
        page_icon="🤖",
        layout="centered",   
    )

    # Set title
    st.title("Chat with website 👋 🤖")

    # Set sidebar
    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL",key="url")

    # Set chat history     
    if website_url is None or website_url == "":
        st.info("Please enter a website URL.")  

    else:
        
        if "chat_history" not in st.session_state:  
            st.session_state.chat_history = [
                AIMessage(content="Hello, how can I help you?"),
            ]
    
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
        
        # user input
        user_query = st.chat_input("Ask a question", key="input")

        if user_query:
            response = get_response(user_query)
            
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        
        # display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            else:
                with st.chat_message("Human"):
                    st.write(message.content)

if __name__ == "__main__":
    main()