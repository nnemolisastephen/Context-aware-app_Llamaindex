import os
import base64
import tempfile
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

st.set_page_config("Chat with your PDF",page_icon="📃",layout="wide")

st.title("Chat with your document 📃")
st.subheader("How to use the application")
st.info("""
        1. Make sure you have your Groq Api Key. If you don't have it visit Groq.com to get one.
        2. Input your Groq Api key. 
        3. Click on Browse to upload your pdf file. The application can only accept a pdf file.
        4. Then once you have uploaded the pdf file. You can start chating with your pdf.
        Note that this application has a chat history. So you can also get more information on previously answered questions.
        
        Happy chatting!!!
        """)

with st.sidebar:
    st.sidebar.title("Settings")
    groq_api_key= st.text_input('GROQ_API_KEY',value="",type="password")

llm = Groq(model="llama-3.3-70b-versatile",api_key= groq_api_key)
embed_model = HuggingFaceEmbedding(model_name= "sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = llm
Settings.embed_model = embed_model


def display_pdf(file_bytes):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" 
                      style="height:100vh; width:100%"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file",type="pdf")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        display_pdf(file_bytes)

        if "file_cache" not in st.session_state:
            st.session_state.file_cache = {}

        session_id = "user_session"
        file_key = f"{session_id}-{uploaded_file.name}"

        if file_key not in st.session_state.file_cache:
            with st.spinner("Indexing your document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_path = tmp_file.name

                

                data_loded = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
                index = VectorStoreIndex.from_documents(data_loded)
                memory = ChatMemoryBuffer(token_limit=3900)

                chat_engine = CondensePlusContextChatEngine.from_defaults(
                    index.as_retriever(),
                    memory=memory,
                    llm=llm,
                    system_prompt="You are a helpful assistant that answers question based on the provided PDF")
                
                st.session_state.file_cache[file_key] = chat_engine
                st.success("Document indexed sucessfully")

chat_engine = None
if uploaded_file:
    file_key = f"user_session-{uploaded_file.name}"
    chat_engine = st.session_state.file_cache.get(file_key)

if chat_engine:
    user_query = st.text_input("Ask your question")

    if st.button("Submit"):
        if not groq_api_key:
            st.error("Put your api key")
        elif not uploaded_file:
            st.warning("Please upload a pdf file to start chating")
        elif not user_query:
            st.warning("Please enter a question")
        else:
            with st.spinner("Fetching the answer"):
                response = chat_engine.chat(user_query)
                st.markdown("**Answer:**")
                st.write(response.response)