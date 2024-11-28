import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader

# Load the model and tokenizer
MODEL_NAME = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to handle user input and model response
def handle_user_input(user_question, pdf_text):
    # Combine the user's question with the PDF content
    context = f"Context: {pdf_text}\n\nQuestion: {user_question}"

    # Tokenize input
    inputs = tokenizer(context, return_tensors="pt", max_length=512, truncation=True)

    # Generate response
    outputs = model.generate(inputs.input_ids, max_length=150, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit App
def main():
    st.set_page_config(page_title="Gemma-2-2B PDF Chat")
    st.title("Chat with PDFs using Gemma-2-2B")

    # Sidebar for uploading PDFs
    st.sidebar.header("Upload PDFs")
    pdf_docs = st.sidebar.file_uploader("Upload PDF documents", accept_multiple_files=True)

    if pdf_docs:
        with st.spinner("Processing PDFs..."):
            pdf_text = get_pdf_text(pdf_docs)
            st.sidebar.success("PDFs processed successfully!")

        # User input
        user_question = st.text_input("Ask a question about the uploaded PDFs:")

        if user_question:
            with st.spinner("Generating response..."):
                response = handle_user_input(user_question, pdf_text)
                st.write("### Response:")
                st.write(response)

if __name__ == "__main__":
    main()
