import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import re
import tempfile
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import io

app = Flask(__name__)
CORS(app)

# Configuration
EMOTIONS = {
    'joy': {'cta': 'Want to share more happy moments?'},
    'love': {'cta': 'Tell me more about your feelings.'},
    'fear': {'cta': 'Would you like to talk about what worries you?'},
    'surprise': {'cta': 'What happened next?'},
    'anger': {'cta': 'Would expressing more help you feel better?'},
    'sadness': {'cta': 'Would sharing more help lighten your burden?'}
}

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_tokenizer, emotion_model = None, None
response_tokenizer, response_model = None, None

def load_models():
    """Lazy load models when first needed"""
    global emotion_tokenizer, emotion_model, response_tokenizer, response_model
    
    if emotion_model is None:
        print("Loading emotion model...")
        emotion_tokenizer = AutoTokenizer.from_pretrained("davidvensilinr/emotion_detection")
        emotion_model = AutoModelForSequenceClassification.from_pretrained("davidvensilinr/emotion_detection")
        emotion_model.to(device)
        emotion_model.eval()
    
    if response_model is None:
        print("Loading response model...")
        response_tokenizer = AutoTokenizer.from_pretrained("davidvensilinr/response_generator")
        response_model = AutoModelForCausalLM.from_pretrained("davidvensilinr/response_generator")
        if response_tokenizer.pad_token is None:
            response_tokenizer.pad_token = response_tokenizer.eos_token
            response_model.config.pad_token_id = response_tokenizer.pad_token_id
        response_model.to(device)
        response_model.eval()

def process_response(text):
    """Clean and format the model response"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'(.{1,20})( \1){2,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'\b(you\?|you\s+\?|you\s*[?])\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ive been|ive|ive\s+been)\b', 'I\'ve been', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(you\?s|you\'s)\b', 'you\'re', text, flags=re.IGNORECASE)
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    meaningful = [s for s in sentences if len(s) > 4 and not re.match(r'^[?!.,\s]+$', s)]
    
    summary = ' '.join(meaningful[:2]).strip() or text[:150].strip()
    if summary and summary[-1] not in '.!?':
        summary += '.'
    
    return summary

def text_to_speech(text):
    """Convert text to base64 encoded audio"""
    if not text.strip():
        return None

    try:
        # Generate to in-memory bytes buffer instead of file
        audio_bytes = io.BytesIO()
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return base64.b64encode(audio_bytes.read()).decode('utf-8')
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
        
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        load_models()
        
        # Generate response
        inputs = response_tokenizer(message + response_tokenizer.eos_token, return_tensors="pt").to(device)
        outputs = response_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2
        )
        reply = process_response(response_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(message):].strip())
        
        # Get emotion
        inputs = emotion_tokenizer(message, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = emotion_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            emotion_id = torch.argmax(probs).item()
            emotion = emotion_model.config.id2label[emotion_id].lower()
            confidence = probs[0][emotion_id].item()
        
        cta = EMOTIONS.get(emotion, {}).get('cta', 'Tell me more about how you feel.')
        full_response = f"{reply} {cta}"
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 3),
            'reply': reply,
            'cta': cta,
            'audio': text_to_speech(full_response)
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)