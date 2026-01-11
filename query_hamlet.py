import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from typing import List, Tuple, Dict

def embed_query(query: str, client: OpenAI) -> List[float]:
    """Create embedding for user query."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def search_qdrant(query_vector: List[float], qdrant_client: QdrantClient,
                  collection_name: str = "hamlet", top_k: int = 10) -> List[Dict]:
    """Search Qdrant for similar chunks with metadata."""
    from qdrant_client.models import SearchRequest, NamedVector

    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )

    results = []
    for hit in search_result.points:
        results.append({
            'text': hit.payload["text"],
            'score': hit.score,
            'page_num': hit.payload.get("page_num"),
            'pages': hit.payload.get("pages", []),
            'act': hit.payload.get("act"),
            'scene': hit.payload.get("scene")
        })

    return results

def generate_answer(query: str, search_results: List[Dict], client: OpenAI) -> str:
    """Generate answer using OpenAI GPT with retrieved context."""

    # Combine context chunks with metadata
    context_parts = []
    for i, result in enumerate(search_results, 1):
        ref = f"[Passage {i}"
        if result['act']:
            ref += f" - {result['act']}"
            if result['scene']:
                ref += f", {result['scene']}"
        if result['page_num']:
            ref += f" - Page {result['page_num']}"
        ref += "]"

        context_parts.append(f"{ref}\n{result['text']}")

    context = "\n\n---\n\n".join(context_parts)

    # Create prompt
    prompt = f"""You are a helpful assistant answering questions about Shakespeare's Hamlet.
Use the following excerpts from Hamlet to answer the user's question. If the answer cannot be found in the excerpts, say so.

Context from Hamlet:
{context}

User Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant helping users understand Shakespeare's Hamlet."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content

def display_header():
    """Display application header."""
    print("\n" + "="*60)
    print(" "*15 + "HAMLET Q&A SYSTEM")
    print(" "*10 + "Ask questions about Shakespeare's Hamlet")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to end the session")
    print("  - Type 'context' after a question to see retrieved chunks")
    print("="*60 + "\n")

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configuration
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "hamlet"
    TOP_K = 10

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set!")
        return

    # Initialize clients
    try:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Test connection
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]

        if COLLECTION_NAME not in collection_names:
            print(f"Error: Collection '{COLLECTION_NAME}' not found in Qdrant!")
            print(f"Available collections: {collection_names}")
            print("\nPlease run 'python embed_hamlet.py' first to create the collection.")
            return

        print(f"✓ Connected to Qdrant. Collection '{COLLECTION_NAME}' found.")

    except Exception as e:
        print(f"Error connecting to services: {e}")
        return

    # Display header
    display_header()

    # Store last context for optional display
    last_context = []
    show_context = False

    # Main query loop
    while True:
        try:
            user_input = input("\n🎭 Your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for exploring Hamlet! Goodbye.\n")
                break

            if user_input.lower() == 'context':
                if last_context:
                    print("\n" + "-"*60)
                    print("RETRIEVED CONTEXT CHUNKS:")
                    print("-"*60)
                    for i, result in enumerate(last_context, 1):
                        ref_parts = []
                        if result['act']:
                            ref_parts.append(result['act'])
                            if result['scene']:
                                ref_parts.append(result['scene'])
                        if result['page_num']:
                            ref_parts.append(f"Page {result['page_num']}")

                        ref_str = " - ".join(ref_parts) if ref_parts else "No metadata"
                        print(f"\nChunk {i} (Similarity: {result['score']:.3f}) - {ref_str}:")
                        text = result['text']
                        print(text[:300] + "..." if len(text) > 300 else text)
                    print("-"*60)
                else:
                    print("No previous query to show context for.")
                continue

            print("\n⏳ Searching for relevant passages...")

            # Embed query
            query_vector = embed_query(user_input, openai_client)

            # Search Qdrant
            search_results = search_qdrant(query_vector, qdrant_client, COLLECTION_NAME, TOP_K)
            last_context = search_results

            if not search_results:
                print("No relevant passages found.")
                continue

            # Display top 10 results from VDB
            print(f"\n✓ Found {len(search_results)} relevant passages from Qdrant VDB:\n")
            print("="*80)
            print("TOP 10 RETRIEVED PASSAGES (Ranked by Confidence Score)")
            print("="*80)

            for i, result in enumerate(search_results, 1):
                # Format reference
                ref_parts = []
                if result['act']:
                    ref_parts.append(result['act'])
                    if result['scene']:
                        ref_parts.append(result['scene'])
                if result['page_num']:
                    ref_parts.append(f"Page {result['page_num']}")
                ref_str = " | ".join(ref_parts) if ref_parts else "No metadata"

                # Display result
                print(f"\n[{i}] Confidence: {result['score']:.4f} | {ref_str}")
                text_preview = result['text'][:200].replace('\n', ' ')
                print(f"    Preview: {text_preview}...")

            print("\n" + "="*80)

            # Select top 5 for LLM context
            top_5_results = search_results[:5]

            print(f"\n⏳ Sending top 5 passages to LLM for analysis and answer generation...")
            print(f"   Using model: gpt-4o-mini")
            print(f"   Context passages: {len(top_5_results)}")
            print(f"   LLM will analyze, pick relevant information, and rephrase the answer.\n")

            # Generate answer
            answer = generate_answer(user_input, top_5_results, openai_client)

            # Display answer
            print("="*80)
            print("LLM GENERATED ANSWER:")
            print("="*80)
            print(answer)
            print("="*80)

            # Display references (top 5 used by LLM)
            print("\nSOURCES USED BY LLM (Top 5 passages sent to GPT-4o-mini):")
            for i, result in enumerate(top_5_results, 1):
                ref_parts = []
                if result['act']:
                    ref_parts.append(result['act'])
                    if result['scene']:
                        ref_parts.append(result['scene'])
                if result['pages'] and len(result['pages']) > 1:
                    ref_parts.append(f"Pages {result['pages'][0]}-{result['pages'][-1]}")
                elif result['page_num']:
                    ref_parts.append(f"Page {result['page_num']}")

                ref_str = " - ".join(ref_parts) if ref_parts else "No metadata"
                print(f"  [{i}] {ref_str} (Confidence: {result['score']:.4f})")

            print("\n💡 Tip: Type 'context' to see the full retrieved text chunks")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()
