from flask import Flask, request, jsonify, send_from_directory
from chatbot import SLTChatbot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize chatbot
chatbot = SLTChatbot(use_local_llm=False, gemini_api_key="AIzaSyC5OS1W-eFvncp4Kj0VrsgMoSs7EZD94XQ")

app = Flask(__name__, static_folder='static')

# Serve index.html
@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    user_id = data.get('user_id', 'web_user')

    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        # If you have vector chunks, you can load them here
        chunks = chatbot.find_relevant_chunks(message, top_n=5)
        reply = chatbot.query_llm(message, chunks, user_id)
        return jsonify({
            "reply": reply,
            "llm_used": "Google Gemini",
            "action_buttons": [],
            "request_location": False
        })
    except Exception as e:
        logger.error(f"Error in /chat: {e}")
        return jsonify({"error": str(e)}), 500

# Optional: location endpoint
@app.route('/location', methods=['POST'])
def location():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude required"}), 400

    try:
        reply = chatbot.handle_location_query_from_coords(lat, lon)
        return jsonify({
            "reply": reply,
            "action_buttons": [],
            "llm_used": "Google Gemini"
        })
    except Exception as e:
        logger.error(f"Error in /location: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4321, debug=True)