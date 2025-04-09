# TODO: Add your import statements



# TODO: Add you default parameters here
# For example: 
STORAGE_DIRECTORY = "storage/"
CHUNKING_STRATEGY = 'fixed-length' # or 'sentence'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# add more as needed...


def initialize_index():
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
    pass




@app.route("/generate", methods=["POST"])
def generate_answer():
    """
    Generate an answer to a given query by running the retrieval and reranking pipeline.

    This endpoint accepts a POST request with a JSON body containing the "query" field.
    It preprocesses and indexes the corpus if necessary, retrieves top-k relevant documents,
    reranks them using a hybrid model, and uses a language model to generate a final answer.

    :return: JSON response containing the generated answer.
    """
    #######################################
    # TODO: Implement initialize()
    #######################################

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)







