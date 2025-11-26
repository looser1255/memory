from flask import Flask, request, jsonify
from openai import OpenAI
import os
import re
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


def get_embedding_vector(text):
    """Generiert einen Embedding-Vektor für den gegebenen Text."""
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
    """Sucht ähnliche Texte in Pinecone."""
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


def extract_obsidian_links(text):
    """
    Extrahiert alle [[wiki-links]] aus einem Text.
    Gibt eine Liste von Note-Namen zurück.
    """
    # Pattern für [[Note Name]] oder [[Note Name|Alias]]
    pattern = r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]'
    links = re.findall(pattern, text)
    # Duplikate entfernen und bereinigen
    return list(set([link.strip() for link in links]))


def search_linked_notes(linked_note_names, original_query, already_found_ids):
    """
    Sucht nach den verlinkten Notes in Pinecone.
    
    Args:
        linked_note_names: Liste der Note-Namen aus [[Links]]
        original_query: Die ursprüngliche Suchanfrage (für Relevanz-Scoring)
        already_found_ids: IDs der bereits gefundenen Ergebnisse (um Duplikate zu vermeiden)
    
    Returns:
        Liste von Ergebnissen mit Zusatz-Info woher der Link kam
    """
    linked_results = []
    
    for note_name in linked_note_names:
        # Suche nach dem Note-Namen
        embedding = get_embedding_vector(note_name)
        if embedding is None:
            continue
        
        # Weniger Ergebnisse pro verlinkter Note
        matches = query_similar_texts(embedding, top_k=3)
        
        for match in matches:
            # Duplikate überspringen
            if match['id'] in already_found_ids:
                continue
            
            # Prüfen ob der Match wirklich zur gesuchten Note gehört
            # (Der Note-Name sollte im Text vorkommen)
            match_text_lower = match['text'].lower()
            note_name_lower = note_name.lower()
            
            # Nur hinzufügen wenn der Score gut genug ist ODER der Note-Name im Text vorkommt
            if match['score'] > 0.7 or note_name_lower in match_text_lower:
                match['source'] = 'linked'
                match['linked_from'] = note_name
                linked_results.append(match)
                already_found_ids.add(match['id'])
    
    return linked_results


@app.route('/retrieve_db', methods=['GET'])
def retrieve_db():
    """
    Erweitertes Retrieval mit Graph-Awareness.
    
    Parameter:
        - text: Suchbegriff (required)
        - top_k: Anzahl der Ergebnisse (optional, default: 10)
        - follow_links: "true"/"false" - Soll den [[Links]] gefolgt werden? (optional, default: true)
        - link_depth: Wie viele Link-Ebenen verfolgen? (optional, default: 1)
    """
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Parameter auslesen
    top_k_param = request.args.get('top_k', None)
    follow_links = request.args.get('follow_links', 'true').lower() == 'true'
    
    # top_k bestimmen
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

    # Phase 1: Normale Suche
    embedding_vector = get_embedding_vector(text)
    if embedding_vector is None:
        return jsonify({"error": "Failed to generate embeddings"}), 500

    primary_results = query_similar_texts(embedding_vector, top_k)
    
    # Markiere primäre Ergebnisse
    for result in primary_results:
        result['source'] = 'primary'
    
    # Wenn follow_links deaktiviert ist, nur primäre Ergebnisse zurückgeben
    if not follow_links:
        return jsonify(primary_results)
    
    # Phase 2: Links aus den gefundenen Notes extrahieren
    all_links = set()
    already_found_ids = set([r['id'] for r in primary_results])
    
    for result in primary_results:
        links = extract_obsidian_links(result.get('text', ''))
        all_links.update(links)
    
    # Phase 3: Verlinkte Notes suchen
    linked_results = []
    if all_links:
        linked_results = search_linked_notes(
            list(all_links), 
            text, 
            already_found_ids
        )
    
    # Ergebnis strukturieren
    response = {
        'primary_results': primary_results,
        'linked_results': linked_results,
        'extracted_links': list(all_links),
        'hint': 'Die linked_results stammen aus Notes, die in den primary_results verlinkt wurden. Prüfe ob diese zusätzlichen Infos für die Anfrage relevant sind.'
    }
    
    return jsonify(response)


@app.route('/add_db', methods=['POST'])
def add_db():
    """Fügt einen Text zur Datenbank hinzu."""
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
    """Löscht einen Eintrag aus der Datenbank."""
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


def save_to_pinecone(text, embedding_vector):
    """Speichert einen Text mit seinem Embedding in Pinecone."""
    try:
        unique_id = str(uuid.uuid4())
        index.upsert(vectors=[(unique_id, embedding_vector, {"text": text})])
        return True
    except Exception as e:
        print(f"Error saving to Pinecone: {e}")
        return False


if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')
