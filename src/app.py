import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import masking_pipeline

st.set_page_config(
    page_title="Socratic-OT Tutor",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #1a1a2e; }
    
    .header { text-align: center; padding: 2rem 0 1rem 0; }
    .header h1 { color: #e0e0ff; font-size: 1.6rem; font-weight: 600; margin: 0; }
    .header p { color: #8888aa; font-size: 0.85rem; margin-top: 0.3rem; }
    
    .user-msg {
        display: flex;
        justify-content: flex-end;
        margin: 0.5rem 0;
    }
    .user-msg span {
        background: #7c3aed;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 16px 16px 4px 16px;
        max-width: 70%;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .bot-msg {
        display: flex;
        justify-content: flex-start;
        margin: 0.5rem 0;
        gap: 0.5rem;
        align-items: flex-start;
    }
    .bot-avatar {
        background: #2d2d4e;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .bot-msg span {
        background: #2d2d4e;
        color: #e0e0ff;
        padding: 0.6rem 1rem;
        border-radius: 16px 16px 16px 4px;
        max-width: 70%;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .stChatInput textarea {
        background: #2d2d4e !important;
        color: #e0e0ff !important;
        border: 1px solid #444466 !important;
        border-radius: 12px !important;
    }
    .stChatInput button {
        background: #7c3aed !important;
    }
    
    #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🧬 Socratic-OT Anatomy Tutor</h1>
    <p>I won't give you the answer but I'll guide you to it!</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_number" not in st.session_state:
    st.session_state.turn_number = 0

if "hidden_answer" not in st.session_state:
    st.session_state.hidden_answer = None

# render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg"><span>{msg["content"]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"><div class="bot-avatar">🧬</div><span>{msg["content"]}</span></div>', unsafe_allow_html=True)

def is_new_topic(new_question, history):
    if not history:
        return True
    previous_questions = [m["content"] for m in history if m["role"] == "user"]
    if not previous_questions:
        return True
    last_question = previous_questions[-1]
    # if the new question shares very few words with the last one, it's a new topic
    new_words = set(new_question.lower().split())
    last_words = set(last_question.lower().split())
    overlap = new_words & last_words
    # remove common filler words
    filler = {"what", "is", "the", "a", "an", "how", "does", "do", "can", "you", "me", "about", "of", "in", "and"}
    overlap = overlap - filler
    return len(overlap) == 0

if user_input := st.chat_input("Ask an anatomy question..."):
    if is_new_topic(user_input, st.session_state.messages):
        st.session_state.turn_number = 0
        st.session_state.hidden_answer = None
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.turn_number += 1
    with st.spinner("Thinking..."):
        hint, hidden_answer = masking_pipeline(
            user_input,
            st.session_state.turn_number,
            st.session_state.messages,
            st.session_state.hidden_answer
        )
        st.session_state.hidden_answer = hidden_answer
    st.session_state.messages.append({"role": "assistant", "content": hint})
    st.rerun()