import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "data/chromadb"))
collection = client.get_or_create_collection("anatomy")

def retrieve_chunks(query, n=5, subject="anatomy"):
    embedding = embed_model.encode(query).tolist()
    
    if subject == "physics":
        col = client.get_or_create_collection("physics")
    else:
        col = collection  
    
    results = col.query(query_embeddings=[embedding], n_results=n)
    return results["documents"][0]



def extract_answer(question, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""You are reading a textbook. Based only on the context below, 
    identify the specific answer to the student's question in 1-2 sentences. 
    Return only the answer, nothing else.

    Context:
    {context}

    Question: {question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def student_is_close(student_response, hidden_answer):
    prompt = f"""You are a fair and experienced anatomy tutor evaluating whether a student understands a concept well enough to move forward.

Hidden answer: {hidden_answer}
Student's response: {student_response}

You are looking for THREE things — but be reasonable, not too strict:
1. Does the student show they know WHAT the structure/system is — even if they use informal language?
2. Does the student show they know its FUNCTION or role — even roughly?
3. Is there at least ONE specific detail that shows genuine understanding — not just a guess?

Important guidance:
- Informal language is fine — "part of the nervous system" counts as identifying it
- Uncertain tone like "I think" or "maybe" is fine — judge the content not the confidence  
- The student does NOT need to use exact medical terminology
- But pure everyday language with no anatomical connection at all is NOT enough
- Ask yourself: "Would a real tutor say this student clearly gets it?" If yes, answer YES

Examples of NO:
- "I think it has something to do with stress" — no anatomical connection, too vague
- "maybe something in the brain?" — no specific detail at all
- "fight or flight means your heart beats faster and you get adrenaline" — everyday language only, no nervous system connection
- "I don't know" — obviously no

Examples of YES:
- "it's the part of the nervous system that prepares you for fight or flight by speeding up heart rate" — informal but has anatomical connection and specific detail
- "I think it controls stress responses, increases heart rate and dilates pupils?" — uncertain tone but three specific details
- "maybe the sympathetic division that activates during danger and redirects blood to muscles?" — close enough

Reply with only YES or NO."""


    try:
        result = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = result.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"student_is_close error: {e}")
        return False
    
def generate_hint(question, chunks, hidden_answer, turn_number, history=None):
    if history is None:
        history = []
    context = "\n\n".join(chunks)

    if turn_number == 1:
        instruction = "This is the student's first message. Do not reference anything they have said before. Ask one opening question that makes the student think about what they already know related to this topic."
    elif turn_number >= 3:
        instruction = "The student is struggling. Give a more explicit hint that narrows the answer significantly but still does not state it directly."
    else:
        instruction = "Ask the student one leading question that guides them toward the answer without revealing it or any synonym of it."

    history_text = ""
    student_messages = [m for m in history if m["role"] == "user"]
    if student_messages:
        history_text = "What the student has said so far:\n"
        for msg in student_messages:
            history_text += f"- {msg['content']}\n"
    
    prompt = f"""You are a Socratic anatomy tutor.
            You know the answer is: [{hidden_answer}] — do NOT reveal it.
            IMPORTANT: Only reference what is explicitly written in the conversation history below. Do NOT invent, assume, or reference any prior discussion that is not shown here.

            {history_text}

            {instruction}

            Textbook context:
            {context}

            Student question: {question}
            Your Socratic response:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def guardrail_check(response_text, hidden_answer):
    prompt = f"""You are checking if a tutor's response reveals the answer to a student.

    Hidden answer: {hidden_answer}
    Tutor's response: {response_text}

    Does the tutor's response either directly state or strongly imply the hidden answer, 
    such that a student reading it would know the answer without having to think?

    Reply with only YES or NO."""

    try:
        result = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = result.choices[0].message.content.strip().upper()
        return "NO" in answer
    except Exception as e:
        print(f"guardrail_check error: {e}")
        return True

def masking_pipeline(question, turn_number, history=None, stored_answer = None):
    if history is None:
        history = []
    chunks = retrieve_chunks(question)
    hidden_answer = stored_answer if stored_answer else extract_answer(question, chunks)
    
    for attempt in range(3):
        hint = generate_hint(question, chunks, hidden_answer, turn_number, history)
        if guardrail_check(hint, hidden_answer):
            return hint, hidden_answer
    
    return "Can you tell me what you already know about this topic?", hidden_answer

if __name__ == "__main__":
    question = "what is the brachial plexus"
    chunks = retrieve_chunks(question)
    print("Retrieved chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:", chunk[:300])
        print("---")
