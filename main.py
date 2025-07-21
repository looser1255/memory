from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import uuid
import time
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

# OpenAI & Pinecone setup
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
pinecone_api_key = os.environ['PINECONE_API_KEY']
index_name = 'custom-gpt'

pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Index erstellen, falls nicht vorhanden
existing = [i["name"] for i in pc.list_indexes()]
if index_name not in existing:
    pc.create_index(name=index_name, dimension=1536, metric='cosine', spec=spec)
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
index = pc.Index(index_name)


@app.route('/retrieve_db', methods=['GET'])
def retrieve_db():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # top_k aus Query-Parameter auslesen
    top_k_param = request.args.get('top_k', None)
    if top_k_param:
        if top_k_param.lower() == 'all':
            stats = index.describe_index_stats()
            top_k = stats.get('total_vector_count', 10)
        else:
            try:
                top_k = int(top_k_param)
            except ValueError:
                return jsonify({"error": "Invalid top_k parameter"}), 400
    else:
        top_k = 10

    embedding_vector = get_embedding_vector(text)
    if embedding_vector is None:
        return jsonify({"error": "Failed to generate embeddings"}), 500

    similar_texts = query_similar_texts(embedding_vector, top_k)
    return jsonify(similar_texts)


@app.route('/add_db', methods=['POST'])
def add_db():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text']

    embedding_vector = get_embedding_vector(text)
    if embedding_vector is None:
        return jsonify({"error": "Failed to generate embeddings"}), 500

    if not save_to_pinecone(text, embedding_vector):
        return jsonify({"error": "Failed to save to Pinecone database"}), 500

    return jsonify({"message": "Text added successfully"}), 200


@app.route('/delete_db', methods=['POST'])
def delete_db():
    data = request.json
    if not data or 'id' not in data:
        return jsonify({"error": "No vector ID provided"}), 400
    vector_id = data['id']

    try:
        delete_response = index.delete(ids=[vector_id])
        return jsonify({
            "message": "Vector deleted successfully",
            "details": delete_response
        }), 200
    except Exception as e:
        print(f"Error deleting from Pinecone: {e}")
        return jsonify({"error": "Failed to delete from Pinecone database"}), 500


def get_embedding_vector(text):
    try:
        response = openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def query_similar_texts(embedding_vector, top_k):
    results = index.query(
        vector=embedding_vector,
        top_k=top_k,
        include_metadata=True
    )
    return [
        {
            'id': match['id'],
            'score': match['score'],
            'text': match.get('metadata', {}).get('text', '')
        }
        for match in results['matches']
    ]


def save_to_pinecone(text, embedding_vector):
    try:
        unique_id = str(uuid.uuid4())
        index.upsert(vectors=[(unique_id, embedding_vector, {"text": text})])
        return True
    except Exception as e:
        print(f"Error saving to Pinecone: {e}")
        return False


if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')
