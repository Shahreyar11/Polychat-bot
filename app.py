import requests
from flask import Flask, request, jsonify, render_template
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import logging
import os
import pyttsx3

app = Flask(__name__)

# GaiaNet Configuration
GAIA_API_ENDPOINT = os.getenv("GAIA_API_ENDPOINT", "https://llama.us.gaianet.network/v1")
GAIA_MODEL_NAME = os.getenv("GAIA_MODEL_NAME", "llama")

translator = Translator()
sentiment_analyzer = SentimentIntensityAnalyzer()

logging.basicConfig(level=logging.DEBUG)

def load_knowledge_base():
    try:
        with open('knowledge_base.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error("knowledge_base.json file not found!")
        return {}

knowledge_base = load_knowledge_base()

def gaia_request(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": GAIA_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(f"{GAIA_API_ENDPOINT}/chat/completions", headers=headers, json=data)
    
    # Check if the response was successful
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Error with GaiaNet API request: {response.status_code} - {response.text}")
        return {"choices": [{"message": {"content": "Sorry, I couldn't fetch a response from GaiaNet."}}]}


def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    return scores

def retrieve_custom_reply(user_message):
    user_message_lower = user_message.lower()
    return knowledge_base.get(user_message_lower, None)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"message": "Invalid input format. JSON expected."}), 400

    data = request.json
    user_message = data.get('message')
    is_voice = data.get('isVoice', False)  # Default to False if not provided

    if not user_message:
        return jsonify({"message": "No message provided."}), 400

    logging.debug(f"User message: {user_message}")

    detected_lang = translator.detect(user_message).lang

    custom_reply = retrieve_custom_reply(user_message)
    if custom_reply:
        if is_voice:
            speak(custom_reply)
        return jsonify({"message": custom_reply})

    user_message_translated = user_message if detected_lang == 'en' else translator.translate(user_message, dest='en').text
    sentiment = analyze_sentiment(user_message_translated)
    sentiment_response = "I see. Let's continue our conversation."
    if sentiment['compound'] >= 0.05:
        sentiment_response = "I'm glad you're feeling good!"
    elif sentiment['compound'] <= -0.05:
        sentiment_response = "I'm sorry you're feeling down. I'm here to help."

    prompt = f"{user_message_translated}\n\n{sentiment_response}"
    gaia_response = gaia_request(prompt)

    if not gaia_response or 'choices' not in gaia_response or not gaia_response['choices']:
        return jsonify({"message": "No valid response from AI."}), 500

    ai_message = gaia_response['choices'][0]['message']['content']
    ai_message += f"\n\n{sentiment_response}"

    if detected_lang != 'en':
        ai_message = translator.translate(ai_message, dest=detected_lang).text

    if is_voice:
        speak(ai_message)

    return jsonify({"message": ai_message})

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
