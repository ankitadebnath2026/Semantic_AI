import streamlit as st
import joblib
import time

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Sentiment AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# 2. Load trained model and vectorizer
# -------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("logistic_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("The mind of the AI is currently resting. Please ensure 'logistic_model.pkl' and 'tfidf_vectorizer.pkl' are available.")
    st.stop()

# -------------------------------
# 3. Emotion label & UI mapping
# -------------------------------
emotion_data = {
    0: {"name": "Sadness",  "emoji": "🌧️", "color": "#4A90E2"},
    1: {"name": "Anger",    "emoji": "🔥", "color": "#E74C3C"},
    2: {"name": "Love",     "emoji": "🤍", "color": "#E91E63"},
    3: {"name": "Surprise", "emoji": "✨", "color": "#F39C12"},
    4: {"name": "Fear",     "emoji": "🌒", "color": "#8E44AD"},
    5: {"name": "Joy",      "emoji": "☀️", "color": "#2ECC71"}
}

# -------------------------------
# 4. Custom Modern CSS
# -------------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 5px;
        padding-top: 20px;
        letter-spacing: -0.5px;
    }
    
   .subtitle {
    text-align: center;
    color: var(--text-color);
    font-size: 19px;
    margin-bottom: 35px;
    font-weight: 300;
    font-style: italic;
    opacity: 0.85;
}
    
    .result-card {
        padding: 35px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
        margin: 25px 0;
    }
    
    .emotion-emoji {
        font-size: 75px;
        margin-bottom: 15px;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .emotion-text {
        font-size: 34px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        height: 55px;
        font-size: 18px;
        font-weight: 500;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3);
        color: white;
    }
    
    .stTextArea textarea {
        border-radius: 18px;
        border: 1px solid rgba(200, 200, 200, 0.2);
        padding: 20px;
        font-size: 17px;
        line-height: 1.6;
        transition: all 0.3s ease;
        background-color: rgba(255,255,255,0.02);
    }
    
    .stTextArea textarea:focus {
        border-color: #FF6B6B;
        box-shadow: 0 0 15px rgba(255, 107, 107, 0.1);
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: #777;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid rgba(200, 200, 200, 0.1);
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 5. Header Section
# -------------------------------
st.markdown('<div class="main-title">Sentiment AI </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Uncover the unspoken heartbeat behind your words.</div>', unsafe_allow_html=True)

# -------------------------------
# 6. User Input
# -------------------------------
user_text = st.text_area(
    label="Your text",
    placeholder="Pour your thoughts onto the page... \n\n(e.g., 'The quiet morning rain brought a sense of peace I hadn't felt in years.')",
    height=140,
    label_visibility="collapsed"
)

# -------------------------------
# 7. Prediction Logic & Layout
# -------------------------------
if st.button("Reveal the Emotion", icon="✨"):
    if user_text.strip():
        with st.spinner("Listening to the nuances of your words..."):
            time.sleep(0.8) # Slightly longer pause for a thoughtful, ambient feel
            
            try:
                # Prediction
                text_tfidf = vectorizer.transform([user_text])
                prediction_idx = int(model.predict(text_tfidf)[0])
                probabilities = model.predict_proba(text_tfidf)[0]
                
                # Fetch emotion details
                emotion_info = emotion_data.get(prediction_idx, {"name": "Unknown", "emoji": "🌫️", "color": "#888888"})
                confidence = max(probabilities) * 100

                # -------------------------------
                # 8. Display Results 
                # -------------------------------
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="emotion-emoji">{emotion_info['emoji']}</div>
                        <div class="emotion-text" style="color: {emotion_info['color']};">
                            {emotion_info['name']}
                        </div>
                        <div style="color: #999; margin-top: 12px; font-size: 16px; font-weight: 300;">
                            Felt with <b>{confidence:.1f}%</b> certainty
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.write("")
                st.write("### 🌌 Emotional Resonance")
                st.caption("A deeper look into the blend of feelings present in your text.")
                st.write("")
                
                col1, col2 = st.columns(2)
                
                for idx, (label_idx, prob) in enumerate(zip(model.classes_, probabilities)):
                    info = emotion_data.get(label_idx, {"name": str(label_idx), "emoji": "◦"})
                    percentage = float(prob) * 100
                    
                    target_col = col1 if idx % 2 == 0 else col2
                    
                    with target_col:
                        st.markdown(f"<span style='font-size:15px;'>{info['emoji']} <b>{info['name']}</b> &nbsp; {percentage:.1f}%</span>", unsafe_allow_html=True)
                        st.progress(float(prob))
                        st.write("")

            except Exception as e:
                st.error(f"The analysis was interrupted. A quiet error occurred: {e}")
    else:
        st.warning("The canvas is empty. Please weave a few words for me to feel.")

# -------------------------------
# 9. Footer
# -------------------------------
st.markdown(
    """
    <div class="footer">
        Crafted with intention • Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)