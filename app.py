import streamlit as st
import tempfile
import os

st.title("File Based QA RAG System")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # To save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    st.success(f"PDF uploaded and saved temporarily at: {temp_file_path}")
    st.session_state['temp_pdf_path'] = temp_file_path # Store path in session state
else:
    st.info("Please upload a PDF file.")
import fitz # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


if uploaded_file is not None:
    # To save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    st.success(f"PDF uploaded and saved temporarily at: {temp_file_path}")
    st.session_state['temp_pdf_path'] = temp_file_path # Store path in session state

    if 'vectorstore' not in st.session_state:
        # Load PDF and extract text
        text_content = ""
        try:
            document = fitz.open(temp_file_path)
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                text_content += page.get_text()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            st.stop()

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        # Split the text content into smaller documents/chunks
        docs = text_splitter.create_documents([text_content])

        # Initialize the Hugging Face embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create a Chroma vector store
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

        st.session_state['vectorstore'] = vectorstore
        st.success("Document processed and vector store created.")

    # Clean up the temporary PDF file
    os.remove(temp_file_path)
    # Clear the temporary file path from session state after processing
    if 'temp_pdf_path' in st.session_state:
        del st.session_state['temp_pdf_path']
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline

@st.cache_resource
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

# Load LLM and store in session state
if 'llm' not in st.session_state:
    st.session_state['llm'] = load_llm()
    st.success("Hugging Face LLM loaded successfully.")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Ensure vectorstore and llm are in session_state before proceeding
if 'vectorstore' in st.session_state and 'llm' in st.session_state:
    vectorstore = st.session_state['vectorstore']
    llm = st.session_state['llm']

    # 1. Create a retriever from the ChromaDB vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}

Question: {question}"""
    )

    # 3. Create the RAG chain
    rag_chain = (
        {
            "context": itemgetter("question") | retriever, # Retriever receives only the question string
            "question": RunnablePassthrough()             # Question still passed through for the prompt
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    st.session_state['rag_chain'] = rag_chain
    st.success("RAG chain setup successfully.")
else:
    st.info("Waiting for PDF upload and LLM to load before setting up RAG chain.")

# User input for questions
if 'rag_chain' in st.session_state:
    st.subheader("Ask a question about the document")
    question = st.text_input("Your question:")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                rag_chain = st.session_state['rag_chain']
                try:
                    answer = rag_chain.invoke({"question": question})
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF and wait for processing to complete before asking questions.")
