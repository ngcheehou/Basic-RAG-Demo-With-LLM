import os
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Tuple

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text from PDF file with page numbers."""
    reader = PdfReader(pdf_path)
    pages_data = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages_data.append({
            'text': text,
            'page_num': page_num
        })

    return pages_data

def detect_act_scene(text: str) -> Tuple[str, str]:
    """Detect Act and Scene from text."""
    act_pattern = r'ACT\s+([IVX]+)'
    scene_pattern = r'SCENE\s+([IVX]+)'

    act_match = re.search(act_pattern, text, re.IGNORECASE)
    scene_match = re.search(scene_pattern, text, re.IGNORECASE)

    act = f"Act {act_match.group(1)}" if act_match else None
    scene = f"Scene {scene_match.group(1)}" if scene_match else None

    return act, scene

def chunk_text_with_metadata(pages_data: List[Dict], chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split text into overlapping chunks while preserving page and act/scene metadata."""
    chunks_with_metadata = []
    current_act = None
    current_scene = None

    # Create a continuous text with position markers
    full_text = ""
    char_to_page = []

    for page_data in pages_data:
        page_text = page_data['text']
        page_num = page_data['page_num']

        # Update act/scene tracking
        act, scene = detect_act_scene(page_text)
        if act:
            current_act = act
        if scene:
            current_scene = scene

        # Track character positions to page numbers
        for char in page_text:
            char_to_page.append({
                'page_num': page_num,
                'act': current_act,
                'scene': current_scene
            })

        full_text += page_text + "\n"
        char_to_page.append({
            'page_num': page_num,
            'act': current_act,
            'scene': current_scene
        })

    # Now chunk the text
    start = 0
    text_length = len(full_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = full_text[start:end]

        if chunk_text.strip():
            # Get metadata from the middle of the chunk
            mid_pos = min(start + chunk_size // 2, len(char_to_page) - 1)
            metadata = char_to_page[mid_pos]

            # Collect all pages this chunk spans
            chunk_pages = set()
            for i in range(start, min(end, len(char_to_page))):
                chunk_pages.add(char_to_page[i]['page_num'])

            chunks_with_metadata.append({
                'text': chunk_text,
                'page_num': metadata['page_num'],
                'pages': sorted(list(chunk_pages)),
                'act': metadata['act'],
                'scene': metadata['scene']
            })

        start += chunk_size - overlap

    return chunks_with_metadata

def create_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    """Create embeddings using OpenAI API."""
    embeddings = []

    print(f"Creating embeddings for {len(texts)} chunks...")
    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"Processing chunk {i}/{len(texts)}")

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    return embeddings

def store_in_qdrant(chunks_with_metadata: List[Dict], embeddings: List[List[float]],
                    qdrant_client: QdrantClient, collection_name: str = "hamlet"):
    """Store embeddings in Qdrant with metadata."""

    # Get vector size from first embedding
    vector_size = len(embeddings[0])

    # Recreate collection
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created collection: {collection_name}")

    # Prepare points
    points = []
    for idx, (chunk_data, embedding) in enumerate(zip(chunks_with_metadata, embeddings)):
        payload = {
            "text": chunk_data['text'],
            "chunk_id": idx,
            "page_num": chunk_data['page_num'],
            "pages": chunk_data['pages'],
            "act": chunk_data['act'],
            "scene": chunk_data['scene']
        }

        point = PointStruct(
            id=idx,
            vector=embedding,
            payload=payload
        )
        points.append(point)

    # Upload in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

    print(f"Successfully stored {len(points)} chunks in Qdrant!")

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configuration
    PDF_PATH = "Source/Hamlet-PDF.pdf"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "hamlet"

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        return

    # Initialize clients
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("Step 1: Extracting text from PDF...")
    pages_data = extract_text_from_pdf(PDF_PATH)
    total_chars = sum(len(page['text']) for page in pages_data)
    print(f"Extracted {total_chars} characters from {len(pages_data)} pages")

    print("\nStep 2: Chunking text with metadata...")
    chunks_with_metadata = chunk_text_with_metadata(pages_data, chunk_size=1000, overlap=200)
    print(f"Created {len(chunks_with_metadata)} chunks")

    print("\nStep 3: Creating embeddings...")
    chunk_texts = [chunk['text'] for chunk in chunks_with_metadata]
    embeddings = create_embeddings(chunk_texts, openai_client)

    print("\nStep 4: Storing in Qdrant with metadata...")
    store_in_qdrant(chunks_with_metadata, embeddings, qdrant_client, COLLECTION_NAME)

    print("\n✓ Embedding complete! Ready to query Hamlet.")

if __name__ == "__main__":
    main()
