import os
import streamlit as st
import time
import requests
import faiss
import torch
from PIL import Image
from io import BytesIO
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from huggingface_hub import login
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key and Hugging Face token are available
if 'OPENAI_API_KEY' not in os.environ:
    st.error("Did not find openai_api_key, please add an environment variable `OPENAI_API_KEY` which contains it.")
    st.stop()
if 'HUGGINGFACE_TOKEN' not in os.environ:
    st.error("Did not find huggingface_token, please add an environment variable `HUGGINGFACE_TOKEN` which contains it.")
    st.stop()

# Login to Hugging Face
login(os.getenv("HUGGINGFACE_TOKEN"))

# Set up Stable Diffusion
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

# Set up your Streamlit app
st.title("AdiBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.index"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Initialize session state variables
if "vectorstore_openai" not in st.session_state:
    st.session_state.vectorstore_openai = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

def generate_image(prompt):
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # Create embeddings and save to FAISS index

    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.vectorstore_openai = FAISS.from_documents(docs, st.session_state.embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a file
    faiss.write_index(st.session_state.vectorstore_openai.index, file_path)

query = main_placeholder.text_input("Question: ")
if query:
    if st.session_state.vectorstore_openai is None and os.path.exists(file_path):
        # Load the FAISS index from the file
        index = faiss.read_index(file_path)
        # Access the docstore and index_to_docstore_id from the existing vectorstore
        docstore = st.session_state.vectorstore_openai.docstore
        index_to_docstore_id = st.session_state.vectorstore_openai.index_to_docstore_id
        st.session_state.vectorstore_openai = FAISS(st.session_state.embeddings.embed_query, index, docstore, index_to_docstore_id)

    if st.session_state.vectorstore_openai:
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state.vectorstore_openai.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
        
        # Generate and display an image based on the answer
        image = generate_image(result["answer"])
        st.image(image, caption="Generated Image")
