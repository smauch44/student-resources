# TODO: Add your import statements
import os
import sys
from glob import glob
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.index.bruteforce import FaissBruteForce
import pandas as pd

from flask import Flask, request, jsonify
from modules.retrieval.reranker import Reranker
from modules.generator.question_answering import QA_Generator


app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    """
    POST /generate
    Request body (JSON):
    {
        "question": "What are the benefits of exercise?",
        "k": 5,
        "rerank_strategy": "hybrid"
    }

    Response (JSON):
    {
        "question": "...",
        "top_chunks": [
            {"text": "....", "score": 0.923},
            ...
        ],
        "answer": "..."
    }
    """
    try:
        data = request.get_json()
        question = data.get("question")
        k = data.get("k", 5)
        strategy = data.get("rerank_strategy", "tfidf")

        if not question:
            return jsonify({"error": "Missing 'question' in request body"}), 400

        # Retrieve top-k (brute force)
        top_k_chunks, top_k_metadata = faiss_index.search(question, top_k=k)

        # Extract raw text chunks
        context_chunks = [meta["text"] for meta in top_k_metadata]

        # Apply reranking
        reranker = Reranker(type=strategy)
        ranked_chunks, _, scores = reranker.rerank(question, context_chunks)

        # Prepare final top-k response
        top_chunks = [
            {"text": chunk, "score": round(score, 4)}
            for chunk, score in zip(ranked_chunks, scores)
        ]

        # Generate final answer using top-ranked chunks
        generator = QA_Generator()
        answer = generator.generate_answer(question, [chunk["text"] for chunk in top_chunks])

        return jsonify({
            "question": question,
            "top_chunks": top_chunks,
            "answer": answer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
# TODO: Add you default parameters here
# For example: 
STORAGE_DIRECTORY = "storage/"
CHUNKING_STRATEGY = 'fixed-length' # or 'sentence'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 300
OVERLAP_SIZE = 20

def initialize_index(
    storage_path=STORAGE_DIRECTORY,
    chunk_strategy=CHUNKING_STRATEGY,
    chunk_size=CHUNK_SIZE,
    overlap_size=OVERLAP_SIZE
):
    """
    1. Parse through all the documents contained in storage/corpus directory
    2. Chunk the documents using either a'sentence' and 'fixed-length' chunking strategies (indicated by the CHUNKING_STRATEGY value):
        - The CHUNKING_STRATEGY will configure either fixed chunk or sentence chunking
    3. Embed each chunk using Embedding class, using 'all-MiniLM-L6-v2' text embedding model as default.
    4. Store vector embeddings of these chunks in a BruteForace index, along with the chunks as metadata. 
    5. This function should return the FAISS index
    """

    #######################################
    # TODO: Implement initialize()
    #######################################
    processor = DocumentProcessing()
    model_name = os.environ.get("CURRENT_EMBEDDING_MODEL", EMBEDDING_MODEL)
    embedder = Embedding(model_name=model_name)
    index = FaissBruteForce(dim=384)

    files = glob(os.path.join(storage_path, "*.txt.clean"))

    for file_path in files:
        # Chunk based on strategy
        if chunk_strategy == "sentence":
            chunks = processor.sentence_chunking(
                file_path, num_sentences=chunk_size, overlap_size=overlap_size
            )
        else:
            chunks = processor.fixed_length_chunking(
                file_path, chunk_size=chunk_size, overlap_size=overlap_size
            )

        # Embed all chunks in batch
        embeddings = embedder.encode(chunks)
        metadata = [
            {
                "source": os.path.basename(file_path),
                "text": chunk
            }
            for chunk in chunks
        ]

        index.add_embeddings(embeddings, metadata)

    return index
