import streamlit as st
from dotenv import load_dotenv
from UploadData_Backend import *


def main():
    load_dotenv()
    st.set_page_config(page_title="Data Upload")
    st.title("Upload files for PerformanceX📁")

    # Upload the text file
    file = st.file_uploader("Only text files allowed", type=["txt"], key='file_uploader')

    # Extract the whole text from the uploaded text file
    if file is not None:
        with st.spinner('In Progress ⌛'):
            
            # Read the text from the uploaded file
            text = read_text_data(file)
            st.write("Reading the file ✔️")

            # Split the text into chunks
            docs_chunks = split_data(text)
            st.write("Splitting data into chunks ✔️")

            # Clean the chunks before creating embeddings
            cleaned_chunks = clean_chunks(docs_chunks)
            st.write("Cleaning data chunks ✔️")

            # Create the embeddings
            embeddings = create_embeddings_load_data()
            st.write("Creating embeddings ✔️")
            
            pinecone_key=st.secrets['PINECONE_API_KEY']
    

            # Push to Pinecone
            push_to_pinecone(
                pinecone_key, 
                "us-east-1", 
                "performancex", 
                embeddings, 
                cleaned_chunks
            )

        st.success("Data Upload Complete")


if __name__ == '__main__':
    main()
