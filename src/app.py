import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manager import ManagerAgent
from vlm import handle_diagram_upload

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
    .user-msg { display: flex; justify-content: flex-end; margin: 0.5rem 0; }
    .user-msg span {
        background: #7c3aed; color: white;
        padding: 0.6rem 1rem;
        border-radius: 16px 16px 4px 16px;
        max-width: 70%; font-size: 0.9rem; line-height: 1.5;
    }
    .bot-msg { display: flex; justify-content: flex-start; margin: 0.5rem 0; gap: 0.5rem; align-items: flex-start; }
    .bot-avatar {
        background: #2d2d4e; border-radius: 50%;
        width: 30px; height: 30px; display: flex;
        align-items: center; justify-content: center;
        font-size: 0.85rem; flex-shrink: 0;
    }
    .bot-msg span {
        background: #2d2d4e; color: #e0e0ff;
        padding: 0.6rem 1rem;
        border-radius: 16px 16px 16px 4px;
        max-width: 70%; font-size: 0.9rem; line-height: 1.5;
    }
    .stChatInput textarea {
        background: #2d2d4e !important; color: #e0e0ff !important;
        border: 1px solid #444466 !important; border-radius: 12px !important;
    }
    .stChatInput button { background: #7c3aed !important; }
    #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🧬 Socratic-OT Anatomy Tutor</h1>
    <p>I won't give you the answer but I'll guide you to it!</p>
</div>
""", unsafe_allow_html=True)

# ask for student name before starting
if "student_id" not in st.session_state:
    st.session_state.student_id = None

if st.session_state.student_id is None:
    st.markdown("<div style='max-width:400px; margin: 2rem auto;'>", unsafe_allow_html=True)
    name = st.text_input("Enter your name to get started:", placeholder="Your name...")
    if st.button("Start Session"):
        if name.strip():
            st.session_state.student_id = name.strip()
            st.rerun()
        else:
            st.warning("Please enter your name first.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# initialize agent and messages
if "agent" not in st.session_state:
    st.session_state.agent = ManagerAgent(st.session_state.student_id)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

# render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        if msg.get("is_image"):
            st.markdown(f'<div class="user-msg"><span>📎 Uploaded a diagram</span></div>', unsafe_allow_html=True)
            st.image(msg["image_data"], width=300)
        else:
            st.markdown(f'<div class="user-msg"><span>{msg["content"]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"><div class="bot-avatar">🧬</div><span>{msg["content"]}</span></div>', unsafe_allow_html=True)

# image upload
uploaded_image = st.file_uploader(
    "Upload a diagram",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="image_uploader"
)

if uploaded_image is not None and uploaded_image.name != st.session_state.processed_image:
    st.session_state.processed_image = uploaded_image.name
    image_bytes = uploaded_image.read()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    
    st.session_state.messages.append({
        "role": "user",
        "content": "Uploaded a diagram",
        "is_image": True,
        "image_data": image_bytes
    })
    
    with st.spinner("Analyzing diagram..."):
        response, image_description, hidden_structure = handle_diagram_upload(tmp_path)
    
    os.remove(tmp_path)
    
    # wire into manager agent tutoring phase
    st.session_state.agent.phase = "tutoring"
    st.session_state.agent.current_topic = hidden_structure
    st.session_state.agent.hidden_answer = hidden_structure
    st.session_state.agent.turn_count = 1
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# text input
if user_input := st.chat_input("Ask an anatomy question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = st.session_state.agent.respond(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()