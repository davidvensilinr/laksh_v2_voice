function showchat() {
    document.getElementById("intro").style.display = "none";
    document.getElementById("chat").style.display = "block";
    document.getElementById("userinput").focus();
}

function appendMessage(sender, text, audioBase64 = null, emotion = 'joy', cta = '') {
    if (!text) return;

    const chatMessages = document.getElementById("chat-messages");

    // Main message
    const msgDiv = document.createElement("div");
    msgDiv.className = sender === "user" ? "chat-bubble user" : "chat-bubble bot";
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);

    // Add CTA if present (bot only)
    if (cta && sender === 'bot') {
        const ctaDiv = document.createElement("div");
        ctaDiv.className = "cta-message";
        ctaDiv.textContent = cta;
        chatMessages.appendChild(ctaDiv);
    }

    // Add TTS button for bot messages with audio
    if (sender === "bot" && audioBase64 && audioBase64 !== 'null' && audioBase64 !== '') {
        const ttsButton = document.createElement("div");
        ttsButton.className = "tts-button playing";
        ttsButton.innerHTML = '<i class="fas fa-volume-up"></i>';
        ttsButton.title = "Speaking...";

        const buttonContainer = document.createElement("div");
        buttonContainer.className = "tts-button-container";
        buttonContainer.appendChild(ttsButton);
        chatMessages.appendChild(buttonContainer);

        // Play audio automatically
        playAudio(audioBase64, ttsButton);
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function playAudio(audioBase64, buttonElement = null) {
    const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);

    if (buttonElement) {
        audio.onplay = () => {
            buttonElement.classList.add("playing");
            buttonElement.innerHTML = '<i class="fas fa-volume-up"></i>';
            buttonElement.title = "Speaking...";
        };

        audio.onended = () => {
            setTimeout(() => {
                if (buttonElement) {
                    buttonElement.classList.remove("playing");
                    buttonElement.innerHTML = '<i class="fas fa-volume-mute"></i>';
                    buttonElement.title = "Click to replay";
                    buttonElement.style.cursor = "pointer";

                    // Make it clickable for replay
                    buttonElement.onclick = () => {
                        playAudio(audioBase64, buttonElement);
                        buttonElement.innerHTML = '<i class="fas fa-volume-up"></i>';
                        buttonElement.title = "Speaking...";
                    };
                }
            }, 500);
        };
    }

    audio.play().catch(e => {
        console.error("Audio playback failed:", e);
        if (buttonElement) {
            buttonElement.classList.remove("playing");
            buttonElement.innerHTML = '<i class="fas fa-volume-mute"></i>';
            buttonElement.title = "Playback failed - click to retry";
            buttonElement.style.cursor = "pointer";
            buttonElement.onclick = () => playAudio(audioBase64, buttonElement);
        }
    });
}

function showTypingIndicator() {
    const chatMessages = document.getElementById("chat-messages");
    const typingDiv = document.createElement("div");
    typingDiv.className = "chat-bubble bot typing-indicator";
    typingDiv.id = "typing-indicator";

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("span");
        dot.className = "typing-dot";
        typingDiv.appendChild(dot);
    }

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById("typing-indicator");
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function sendMessage() {
    const inputElem = document.getElementById("userinput");
    const input = inputElem.value.trim();

    if (!input) return;

    appendMessage("user", input);
    inputElem.value = "";

    // Show typing indicator
    showTypingIndicator();

    fetch("/predict", {
        method: "POST",
        headers: { 
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        body: JSON.stringify({ message: input })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        hideTypingIndicator();
        
        if (data.error) {
            appendMessage("bot", "Sorry, I encountered an error: " + data.error);
        } else {
            const emotion = data.emotion.toLowerCase();
            const imagePath = `static/${emotion}.png`;
            document.getElementById("emotion-img").src = imagePath;
            document.getElementById("emotion").textContent = emotion;
            document.getElementById("confidence").textContent = `- ( ${data.confidence} )`;
            
            if (data.reply) {
                appendMessage("bot", data.reply, data.audio, emotion, data.cta);
            } else {
                appendMessage("bot", "I'm not sure how to respond to that.");
            }
        }
    })
    .catch(error => {
        hideTypingIndicator();
        console.error("Error:", error);
        appendMessage("bot", "Sorry, I'm having trouble responding right now.");
    });
}

document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("userinput").addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
    });
    
    document.getElementById("user-input").addEventListener("click", sendMessage);
});