from flask import Flask, request, jsonify,render_template
import openai
import os

app = Flask(__name__)

openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_version = "2024-02-01"

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

openai.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")
        
        response = openai.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": user_input}],
            max_tokens=200
        )
        
        return jsonify({"response": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True) 
