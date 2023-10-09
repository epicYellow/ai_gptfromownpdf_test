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

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        #creates pdf object with pages
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks=text_splitter.split_text(text)

    return chunks

def get_vectorstore_openAI(chunks):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore

def get_vectorstore_Hug(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vectorstore

def get_convo_chainOpenAI(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        memory=memory,
        llm=llm,
        retriever=vectorstore.as_retriever()
        )
    return conversation_chain

def get_convo_chainHug(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        memory=memory,
        llm=llm,
        retriever=vectorstore.as_retriever()
        )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        
def main():
    load_dotenv()

    #initialize session storage variable
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title="chat")
    st.write(css, unsafe_allow_html=True)
    
    st.header("Chat With Pdfs")
    user_question = st.text_input("Ask question")

    if user_question:
        handle_userInput(user_question)

    with st.sidebar:
        st.subheader("documents")
        pdf_docs = st.file_uploader("Upload pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text raw
                raw_text = get_pdf_text(pdf_docs)

                # Get text chunks
                chunks = get_text_chunks(raw_text)

                # create vector store (embeddings)
                # Embeddings refer to a technique used to represent words, phrases, or entire documents as numerical vectors in a high-dimensional space
                # Word embeddings are a type of text embedding commonly used in NLP. They map words from a vocabulary to continuous vector spaces, where words with similar meanings are closer to each other in the vector space
                vectorstore = get_vectorstore_openAI(chunks)
                st.write(vectorstore)

                # conversation chain
                #saves in session
                st.session_state.conversation = get_vectorstore_Hug(vectorstore)

    # st.session_state.conversation

if __name__ == '__main__':
    main()