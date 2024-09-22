import requests
from flask import Flask, request, jsonify, render_template
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# GaiaNet Configuration
GAIA_API_ENDPOINT = "https://llama.us.gaianet.network/v1"
GAIA_MODEL_NAME = "llama"

# Initialize Translator and Sentiment Analyzer
translator = Translator()
sentiment_analyzer = SentimentIntensityAnalyzer()

# Interactive Learning Data
learning_mode = False
learning_topic = ""
vocabulary = {
    'english': ['hello', 'world', 'python', 'chatbot', 'language'],
    'spanish': ['hola', 'mundo', 'python', 'chatbot', 'idioma'],
    'french': ['bonjour', 'monde', 'python', 'chatbot', 'langue']
}

# Emergency Phrases
emergency_phrases = {
    'english': ['Help!', 'Call the police.', 'I need a doctor.', 'Emergency!'],
    'spanish': ['¡Ayuda!', 'Llame a la policía.', 'Necesito un médico.', '¡Emergencia!'],
    'french': ['À l\'aide!', 'Appelez la police.', 'J\'ai besoin d\'un médecin.', 'Urgence!']
}

def gaia_request(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": GAIA_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(f"{GAIA_API_ENDPOINT}/chat/completions", headers=headers, json=data)
    return response.json()

def analyze_sentiment(text):
    scores = sentiment_analyzer.polarity_scores(text)
    return scores

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global learning_mode, learning_topic
    user_message = request.json['message']
    detected_lang = translator.detect(user_message).lang

    # Emergency Language Support
    if 'emergency' in user_message.lower() or 'help' in user_message.lower():
        if detected_lang in emergency_phrases:
            emergency_response = " ".join(emergency_phrases[detected_lang])
        else:
            emergency_response = " ".join(emergency_phrases['english'])
        return jsonify({"message": emergency_response})

    # Interactive Learning Commands
    if user_message.lower().startswith("/learn"):
        parts = user_message.split()
        if len(parts) > 1:
            topic = parts[1].lower()
            if topic in vocabulary:
                learning_mode = True
                learning_topic = topic
                vocab_list = ", ".join(vocabulary[topic])
                return jsonify({"message": f"Let's learn {topic}! Here are some words: {vocab_list}"})
            else:
                return jsonify({"message": "Sorry, I don't have data for that language."})
        else:
            return jsonify({"message": "Please specify a language to learn, e.g., /learn spanish."})

    if learning_mode:
        if detected_lang == learning_topic:
            return jsonify({"message": "Great job! Keep practicing your vocabulary."})
        else:
            learning_mode = False
            return jsonify({"message": "Learning session ended."})

    # Detect and translate user's language input to English if not already
    if detected_lang != 'en':
        user_message_translated = translator.translate(user_message, dest='en').text
    else:
        user_message_translated = user_message

    # Analyze Sentiment
    sentiment = analyze_sentiment(user_message_translated)
    if sentiment['compound'] >= 0.05:
        sentiment_response = "I'm glad you're feeling good!"
    elif sentiment['compound'] <= -0.05:
        sentiment_response = "I'm sorry you're feeling down. I'm here to help."
    else:
        sentiment_response = "I see. Let's continue our conversation."

    # Preprocess prompt by adding sentiment context
    prompt = f"{user_message_translated}\n\n{sentiment_response}"

    # Get response from GaiaNet AI
    gaia_response = gaia_request(prompt)

    # Extract AI message
    ai_message = gaia_response['choices'][0]['message']['content']

    # Postprocess: Append sentiment response
    ai_message += f"\n\n{sentiment_response}"

    # Translate AI response back to user's language if needed
    if detected_lang != 'en':
        ai_message = translator.translate(ai_message, dest=detected_lang).text

    return jsonify({"message": ai_message})

if __name__ == '__main__':
    app.run(debug=True)
