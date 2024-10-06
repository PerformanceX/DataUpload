import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone

# Read text data from a file

def read_text_data(uploaded_file):
    text = uploaded_file.read().decode('utf-8')  # Read and decode the file contents
    return text


# Split data based on headings and chunk it logically
def split_data(text):
    # Define the regex patterns for headings, subheadings, and bullet points
    pattern = r"(#+\s?.+?(\n|$))|(-\s?.+?(\n|$))"
    
    # Split the text based on the pattern, keeping the split delimiters
    sections = re.split(pattern, text)
    
    # Initialize a list to hold logical sections
    logical_sections = []
    
    # Group the content logically by combining headings with their related content
    current_section = ""
    for section in sections:
        if section and isinstance(section, str):  # Check if the section is not None and is a string
            if re.match(r"(#+\s?.+)", section):  # If it's a heading or subheading
                if current_section:
                    logical_sections.append(current_section.strip())  # Save the previous section
                current_section = section  # Start a new section with the heading
            else:
                current_section += section  # Append the content to the current section

    # Add the last section
    if current_section:
        logical_sections.append(current_section.strip())
    
    # Now apply the text splitter to each logical section
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs_chunks = []
    
    for section in logical_sections:
        section_chunks = text_splitter.create_documents([section])
        docs_chunks.extend(section_chunks)
    
    return docs_chunks

# Clean text by removing special characters like #, \u200b, and unnecessary newlines
def clean_text(text):
    # Remove zero-width spaces
    cleaned_text = text.replace('\u200b', '')
    
    # Replace # with a space or remove entirely
    cleaned_text = cleaned_text.replace('#', '')
    cleaned_text = cleaned_text.replace('-', '')
    
    # Normalize newline characters
    cleaned_text = cleaned_text.replace('\n\n', '\n').replace('\n', ' ')  # Replace newlines with space for smoother flow

    # Optionally remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# Apply the cleaning to each chunk
def clean_chunks(docs_chunks):
    for idx, chunk in enumerate(docs_chunks):
        chunk_content = chunk.page_content
        cleaned_chunk_content = clean_text(chunk_content)
        docs_chunks[idx].page_content = cleaned_chunk_content  # Update the chunk with cleaned content
    return docs_chunks

# Create embeddings instance
def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
    return embeddings

# Function to push data to Pinecone
def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    PineconeClient(api_key=pinecone_apikey, environment=pinecone_environment)
    index_name = pinecone_index_name
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

# Main function to read, chunk, clean, and push data
def process_and_push_data(file_path, pinecone_apikey, pinecone_environment, pinecone_index_name):
    # Read the data from a text file
    text_data = read_text_data(file_path)
    
    # Split the text into chunks
    docs_chunks = split_data(text_data)
    
    # Clean each chunk
    cleaned_chunks = clean_chunks(docs_chunks)
    
    # Create embeddings
    embeddings = create_embeddings_load_data()
    
    # Push the cleaned, chunked data to Pinecone
    push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, cleaned_chunks)
