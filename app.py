from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI # Utilisation du client Azure spécifique
import os
from dotenv import load_dotenv # Recommandé pour charger le .env

# Charge les variables d'environnement
load_dotenv()

app = Flask(__name__)

# 1. Configuration du client Azure OpenAI
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key        = os.getenv("AZURE_OPENAI_KEY"),
    api_version    = "2024-02-15-preview" # Cette version supporte bien le RAG
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# 2. Configuration Azure AI Search (RAG)
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key      = os.getenv("AZURE_SEARCH_KEY")
search_index    = os.getenv("AZURE_SEARCH_INDEX") # Nom de l'index créé étape 4.3

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")
        
        # 3. Appel avec RAG via "extra_body"
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system", 
                    "content": "Tu es un assistant intelligent. Utilise les documents fournis pour répondre à la question."
                },
                {
                    "role": "user", 
                    "content": user_input
                }
            ],
            max_tokens=800,
            temperature=0, # Température basse recommandée pour le RAG
            extra_body={
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": search_endpoint,
                            "index_name": search_index,
                            "authentication": {
                                "type": "api_key",
                                "key": search_key
                            },
                                "fields_mapping": {
                                "content_fields": ["content"],   
                                "title_field": "metadata_storage_name",
                                "url_field": "metadata_storage_path",
                                "filepath_field": "metadata_storage_name"
                            }
                        }
                    }
                ]
            }
        )
        
        # On récupère la réponse
        # Note: Azure ajoute parfois des citations dans le message context
        return jsonify({"response": response.choices[0].message.content})

    except Exception as e:
        print(f"Erreur: {e}") # Affiche l'erreur dans la console pour le debug
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)