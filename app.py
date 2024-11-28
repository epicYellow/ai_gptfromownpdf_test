import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

from htmlTemplates import css, bot_template, user_template

import fitz  # PyMuPDF

import fitz  # PyMuPDF

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        # Use the file-like object directly with PyMuPDF
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore_Hug(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def get_convo_chainHug(vectorstore):
    # Define the HuggingFace model without overriding the task
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.6, "max_length": 512})

    # Define memory and conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        memory=memory, llm=llm, retriever=vectorstore.as_retriever()
    )
    return conversation_chain


def handle_userInput(user_question):
    if st.session_state.conversation is None:
        st.error("No conversation chain found. Please upload and process PDFs first.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(reversed(st.session_state.chat_history)):
        template = user_template if i % 2 != 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="Chat with PDFs")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with PDFs")
    user_question = st.text_input("Ask a question:")

    if user_question:
        handle_userInput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore_Hug(chunks)
                st.write("Vector store created!")
                st.session_state.conversation = get_convo_chainHug(vectorstore)
                st.success("Conversation chain initialized!")

if __name__ == "__main__":
    main()
