from sentence_transformers import SentenceTransformer
def getSentenceTransformerMiniModel():
    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def getSentenceTransformerGeminiModel():
    # Load the model
    model = SentenceTransformer("Gemini")
    return model
def getSentenceTransformerHuggingFaceModel():
    model = "intfloat/e5-large-v2"
    return model