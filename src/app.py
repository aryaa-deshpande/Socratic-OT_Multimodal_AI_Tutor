import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manager import ManagerAgent
from vlm import handle_diagram_upload
from memory import save_mastery

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
    .home-btn { margin-bottom: 1rem; }
    #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# session state
if "page" not in st.session_state:
    st.session_state.page = "name"
if "student_id" not in st.session_state:
    st.session_state.student_id = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

def go_home():
    agent = st.session_state.agent
    if agent is not None:
        if agent.phase == "tutoring" and agent.current_topic:
            save_mastery(
                student_id=st.session_state.student_id,
                topic=agent.current_topic,
                score="incomplete",
                tutor_note="Student left session before completing topic",
                student_summary="Session ended early on this topic"
            )
        elif agent.phase == "assessment":
            save_mastery(
                student_id=st.session_state.student_id,
                topic=agent.current_topic,
                score="incomplete",
                tutor_note="Student left during assessment",
                student_summary="Session ended during clinical scenario"
            )
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.processed_image = None
    st.session_state.page = "home"
    st.rerun()

def render_name_page():
    st.markdown("""
    <div style='text-align: center; padding: 4rem 0 2rem 0;'>
        <h1 style='color: #e0e0ff; font-size: 2rem; margin-bottom: 0.5rem;'>🧬 Socratic-OT</h1>
        <p style='color: #8888aa; font-size: 0.95rem;'>Your personal anatomy tutor</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name = st.text_input("Name", placeholder="Enter your name...", label_visibility="collapsed")
        if st.button("Get Started", use_container_width=True):
            if name.strip():
                st.session_state.student_id = name.strip()
                st.session_state.page = "home"
                st.rerun()
            else:
                st.warning("Please enter your name first.")

def render_home_page():
    st.markdown(f"""
    <div style='text-align: center; padding: 3rem 0 2rem 0;'>
        <h1 style='color: #e0e0ff; font-size: 1.8rem; margin-bottom: 0.3rem;'>🧬 Socratic-OT</h1>
        <p style='color: #8888aa; font-size: 0.9rem;'>Welcome back, {st.session_state.student_id}!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='max-width: 500px; margin: 0 auto;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8888aa; text-align: center; margin-bottom: 1.5rem;'>How would you like to study today?</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Text Chat", use_container_width=True, help="Ask anatomy questions and get Socratic hints"):
            st.session_state.agent = ManagerAgent(st.session_state.student_id)
            st.session_state.messages = []
            st.session_state.page = "text_chat"
            st.rerun()
    with col2:
        if st.button("🖼️ Diagram Chat", use_container_width=True, help="Upload an anatomical diagram and get guided"):
            st.session_state.agent = ManagerAgent(st.session_state.student_id)
            st.session_state.messages = []
            st.session_state.processed_image = None
            st.session_state.page = "diagram_chat"
            st.rerun()
    with col3:
        if st.button("📊 My Progress", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_chat_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            if msg.get("is_image"):
                st.markdown('<div class="user-msg"><span>📎 Uploaded a diagram</span></div>', unsafe_allow_html=True)
                st.image(msg["image_data"], width=300)
            else:
                st.markdown(f'<div class="user-msg"><span>{msg["content"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg"><div class="bot-avatar">🧬</div><span>{msg["content"]}</span></div>', unsafe_allow_html=True)

def render_text_chat_page():
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("🏠", help="Go home"):
            go_home()
    with col2:
        st.markdown("""
        <div style='padding: 0.5rem 0;'>
            <h1 style='color: #e0e0ff; font-size: 1.4rem; margin: 0;'>🧬 Socratic-OT — Text Chat</h1>
        </div>
        """, unsafe_allow_html=True)
    
    render_chat_history()
    
    if user_input := st.chat_input("Ask an anatomy question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = st.session_state.agent.respond(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

def render_diagram_chat_page():
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("🏠", help="Go home"):
            go_home()
    with col2:
        st.markdown("""
        <div style='padding: 0.5rem 0;'>
            <h1 style='color: #e0e0ff; font-size: 1.4rem; margin: 0;'>🧬 Socratic-OT — Diagram Chat</h1>
        </div>
        """, unsafe_allow_html=True)
    
    render_chat_history()
    
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
        
        st.session_state.agent.phase = "tutoring"
        st.session_state.agent.current_topic = hidden_structure
        st.session_state.agent.hidden_answer = hidden_structure
        st.session_state.agent.turn_count = 1
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    if user_input := st.chat_input("Reply to the tutor..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = st.session_state.agent.respond(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

def render_dashboard_page():
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("🏠", help="Go home"):
            go_home()
    with col2:
        st.markdown(f"""
        <div style='padding: 0.5rem 0;'>
            <h1 style='color: #e0e0ff; font-size: 1.4rem; margin: 0;'>📊 My Progress</h1>
        </div>
        """, unsafe_allow_html=True)
    
    from memory import get_student_summary
    records = get_student_summary(st.session_state.student_id)
    
    if not records:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; color: #8888aa;'>
            <p style='font-size: 1.2rem;'>No sessions yet!</p>
            <p style='font-size: 0.9rem;'>Complete a tutoring session to see your progress here.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # summary stats
    total = len(records)
    strong = sum(1 for r in records if r[1] == "strong")
    partial = sum(1 for r in records if r[1] == "partial")
    weak = sum(1 for r in records if r[1] == "weak" or r[1] == "incomplete")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Topics", total)
    with col2:
        st.metric("Strong", strong)
    with col3:
        st.metric("Needs Work", partial + weak)
    
    st.markdown("<hr style='border-color: #2d2d4e; margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # bar chart
    import pandas as pd
    df = pd.DataFrame(records, columns=["topic", "score", "summary", "date"])
    
    score_order = {"strong": 3, "partial": 2, "weak": 1, "incomplete": 0}
    score_color = {"strong": "#22c55e", "partial": "#f59e0b", "weak": "#ef4444", "incomplete": "#6b7280"}
    
    st.markdown("<p style='color: #e0e0ff; font-weight: 600; margin-bottom: 0.5rem;'>Topic Scores</p>", unsafe_allow_html=True)
    
    for _, row in df.iterrows():
        color = score_color.get(row["score"], "#6b7280")
        st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 1rem; margin: 0.4rem 0;'>
            <div style='flex: 1; color: #e0e0ff; font-size: 0.9rem;'>{row['topic']}</div>
            <div style='background: {color}; color: white; padding: 2px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 500;'>{row['score']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #2d2d4e; margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # weak spots
    weak_records = [r for r in records if r[1] in ["partial", "weak", "incomplete"]]
    if weak_records:
        st.markdown("<p style='color: #e0e0ff; font-weight: 600; margin-bottom: 0.5rem;'>Topics to Revisit</p>", unsafe_allow_html=True)
        for r in weak_records:
            st.markdown(f"""
            <div style='background: #2d2d4e; border-left: 3px solid #f59e0b; padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0;'>
                <p style='color: #e0e0ff; margin: 0; font-size: 0.9rem; font-weight: 500;'>{r[0]}</p>
                <p style='color: #8888aa; margin: 0; font-size: 0.8rem;'>{r[2]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # session history
    st.markdown("<hr style='border-color: #2d2d4e; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #e0e0ff; font-weight: 600; margin-bottom: 0.5rem;'>Session History</p>", unsafe_allow_html=True)
    for r in records:
        date = r[3][:10] if r[3] else "unknown"
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #2d2d4e;'>
            <span style='color: #e0e0ff; font-size: 0.85rem;'>{r[0]}</span>
            <span style='color: #8888aa; font-size: 0.8rem;'>{date}</span>
        </div>
        """, unsafe_allow_html=True)

# page router
if st.session_state.page == "name":
    render_name_page()
elif st.session_state.page == "home":
    render_home_page()
elif st.session_state.page == "text_chat":
    render_text_chat_page()
elif st.session_state.page == "diagram_chat":
    render_diagram_chat_page()
elif st.session_state.page == "dashboard":
    render_dashboard_page()