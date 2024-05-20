from PIL import Image
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from dotenv import load_dotenv
import openai
import os

load_dotenv()

API_KEY = os.environ.get('GPT_API_KEY')

def writeStatements(chat_history):
    for speaker, text in chat_history:
        title_container = st.container()
        col1, col2 = st.columns([1, 20])
        if speaker == 'You':
            image_path = "human_image.png"
        else:
            st.markdown("<hr>", unsafe_allow_html=True)
            image_path = "bot_image.png"
        image = Image.open(image_path)
        with title_container:
            with col1:
                st.image(image, width=32)
            with col2:
                st.write(text)

def main():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    st.title("Ask Me - Multiple PDFs")

    st.sidebar.title("Upload Files")

    def onclick():
        st.session_state['chat_history'] = []

    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, on_change=onclick)

    if uploaded_files:
        st.write("Uploaded Files:")
        for file in uploaded_files:
            st.write(f"- {file.name}")

        documents = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, mode='wb') as w:
                w.write(uploaded_file.getvalue())

            loader = UnstructuredPDFLoader(uploaded_file.name)
            pages = loader.load_and_split()
            documents.extend(pages)

            if os.path.exists(uploaded_file.name):
                os.remove(uploaded_file.name)

        embeddings = OpenAIEmbeddings(api_key=API_KEY)
        docsearch = Chroma.from_documents(documents, embeddings).as_retriever()

        writeStatements(chat_history=st.session_state['chat_history'])

        styl = f"""
        <style>
            div[data-testid="stTextInput"] label {{
                display: none;
            }}
            .stTextInput {{
            position: fixed;
            # margin: 20px 0px;
            padding-bottom: 30px;
            bottom: 0;
            background-color: rgb(14, 17, 23);
            }}
        </style>
        """

        st.markdown(styl, unsafe_allow_html=True)
        if 'something' not in st.session_state:
            st.session_state.something = ''

        def submit():
            st.session_state.something = st.session_state.widget
            st.session_state.widget = ''
            docs = docsearch.invoke(st.session_state.something)
            chain = load_qa_chain(ChatOpenAI(api_key=API_KEY, temperature=0), chain_type="stuff")
            output = chain.run(input_documents=docs, question=st.session_state.something)
            st.session_state['chat_history'].append(("You", st.session_state.something))
            st.session_state['chat_history'].append(("PDF", output))

        st.text_input('Something', placeholder="Ask your query",key='widget', on_change=submit)
        # st.write('Built by Chaity and Aadi 😊')

if __name__ == "__main__":
    main()
