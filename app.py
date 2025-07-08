from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import io
import re
import threading
import tempfile
import os
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

# Emotion configuration
EMOTIONS = {
    'joy': {'cta': 'Want to share more happy moments?'},
    'love': {'cta': 'Tell me more about your feelings.'},
    'fear': {'cta': 'Would you like to talk about what worries you?'},
    'surprise': {'cta': 'What happened next?'},
    'anger': {'cta': 'Would expressing more help you feel better?'},
    'sadness': {'cta': 'Would sharing more help lighten your burden?'}
}

# Load emotion model
emotion_model_path = "davidvensilinr/emotion_detection"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model.to(device)
emotion_model.eval()

# Load response generator model (move this out of predict)
response_model_path = "davidvensilinr/response_generator"
response_tokenizer = AutoTokenizer.from_pretrained(response_model_path)
response_model = AutoModelForCausalLM.from_pretrained(response_model_path)
if response_tokenizer.pad_token is None:
    response_tokenizer.pad_token = response_tokenizer.eos_token
    response_model.config.pad_token_id = response_tokenizer.pad_token_id
response_model.to(device)
response_model.eval()

def process_response(text):
    """
    Post-process the raw model output so that the reply:
    1. Has no obvious repetitions.
    2. Contains meaningful content (up to 2-3 sentences, ≤200 chars).
    3. Always ends with proper punctuation.
    4. Makes conversational sense.
    """
    # Clean up the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove obvious repetitions and nonsensical patterns
    text = re.sub(r'\b(\w+)( \1\b){2,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'(.{1,20})( \1){2,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'[?]{2,}', '?', text)  # Remove multiple question marks
    text = re.sub(r'[!]{2,}', '!', text)  # Remove multiple exclamation marks
    
    # Remove common nonsensical patterns from model output
    text = re.sub(r'\b(you\?|you\s+\?|you\s*[?])\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(ive been|ive|ive\s+been)\b', 'I\'ve been', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(you\?s|you\'s)\b', 'you\'re', text, flags=re.IGNORECASE)
    
    # Break into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
    
    # Filter out very short or nonsensical sentences
    meaningful_sentences = []
    for s in sentences:
        # Skip sentences that are too short or contain mostly punctuation
        if len(s) < 5 or re.match(r'^[?!.,\s]+$', s):
            continue
        # Skip sentences that are just repeated words
        if len(set(s.lower().split())) < 2:
            continue
        meaningful_sentences.append(s)
    
    # Keep up to 2 meaningful sentences / 200 chars
    summary_parts, total_len = [], 0
    for s in meaningful_sentences:
        if total_len + len(s) <= 200 and len(summary_parts) < 2:
            summary_parts.append(s)
            total_len += len(s)
        else:
            break
    
    summary = ' '.join(summary_parts).strip()
    
    # Fallback: if no meaningful sentences, take first reasonable chunk
    if not summary:
        # Take first 150 chars that end with punctuation
        summary = text[:150].strip()
        if summary and summary[-1] not in '.!?':
            # Find last punctuation mark
            for i in range(len(summary)-1, -1, -1):
                if summary[i] in '.!?':
                    summary = summary[:i+1]
                    break
            else:
                summary += '.'
    
    # Ensure terminal punctuation
    if summary and summary[-1] not in '.!?':
        summary += '.'
    
    return summary

def text_to_speech(text):
    """Convert text to speech using gTTS and return Base64-encoded mp3 data."""
    if not text or not text.strip():
        print("TTS: Empty text provided")
        return None

    try:
        print(f"TTS: Generating audio for: '{text[:50]}...'")
        
        # Create a temporary file for the audio
        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        
        # Verify that the file was created and has content
        if not os.path.exists(temp_path):
            print("Warning: gTTS file was not created.")
            return None
            
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            print("Warning: gTTS generated empty audio file.")
            return None
        
        print(f"TTS: Generated audio file of size: {file_size} bytes")
        
        # Read the audio file and convert to base64
        with open(temp_path, 'rb') as f:
            audio_data = f.read()
        
        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except OSError:
            pass
        
        # Return base64 encoded audio data
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        print(f"TTS: Successfully generated base64 audio of length: {len(base64_data)}")
        return base64_data
        
    except Exception as e:
        print(f"Error in TTS conversion: {e}")
        # Clean up temp file if it exists
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return None

def predict_emotion(text):
    """Predict emotion from text with proper attention mask"""
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_label_id = torch.argmax(probs, dim=1).item()
        pred_label = emotion_model.config.id2label[pred_label_id].lower()
        confidence = probs[0][pred_label_id].item()
    
    return pred_label if pred_label in EMOTIONS else 'joy', confidence

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    print(f"\nReceived message: {message}")  # Debug print

    try:
        # Use preloaded response generator model
        tokenizer = response_tokenizer
        model = response_model

        # Generate with attention mask
        inputs = tokenizer(message + tokenizer.eos_token, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            no_repeat_ngram_size=2  # Prevent repeating n-grams
        )

        raw_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply_only = raw_reply[len(message):].strip()

        print(f"Raw model output: {raw_reply}")  # Debug print
        print(f"Extracted reply: {reply_only}")  # Debug print

        # Process the response
        processed_reply = process_response(reply_only)
        print(f"Processed reply: {processed_reply}")  # Debug print

        # Get emotion and CTA
        emotion, confidence = predict_emotion(message)
        cta = EMOTIONS.get(emotion, {}).get('cta', 'Tell me more about how you feel.')

        # Generate audio for the complete response (reply + CTA)
        full_response = f"{processed_reply} {cta}"
        audio_base64 = text_to_speech(full_response)

        response_data = {
            'emotion': emotion,
            'confidence': round(confidence, 3),
            'reply': processed_reply,
            'cta': cta,
            'audio': audio_base64
        }

        print(f"Returning response: {response_data}")  # Debug print
        return jsonify(response_data)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # fallback to 10000 on Render
    print(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port)
