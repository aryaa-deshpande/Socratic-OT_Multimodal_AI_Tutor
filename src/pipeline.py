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
    prompt = f"""You are evaluating how close a student is to correctly answering a question.

Hidden answer: {hidden_answer}
Student response: {student_response}

Rate the student's response on a scale of 0 to 10 using this rubric:

0-3: No scientific understanding — uses only everyday language, describes observable effects without any mechanism, or makes a vague single-word guess
4-6: Surface level understanding — identifies the general category correctly but missing specific components, correct direction but incomplete
7-8: Solid understanding — correctly identifies what it is, describes its function with at least one specific detail, understands the mechanism not just the observable effect
9-10: Complete understanding — all key components described, mechanism fully explained, specific enough that a tutor would say "yes exactly"

Important:
- Uncertain tone like "I think" or "maybe" is fine — judge content not confidence
- Everyday language describing only observable effects scores 0-3 regardless of topic
- A single vague word that appears in the hidden answer does NOT warrant a high score

Reply with only a number between 0 and 10. Nothing else."""

    try:
        result = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = result.choices[0].message.content.strip().lower()
        
        # handle word numbers
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        
        # try extracting a digit first
        import re
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', raw)
        if numbers:
            score = float(numbers[0])
        else:
            # try word numbers
            for word, num in word_to_num.items():
                if word in raw:
                    score = float(num)
                    break
            else:
                print(f"student_is_close could not parse score from: {raw}")
                return 0
        
        score = min(max(score, 0), 10)  # clamp between 0 and 10
        print(f"student_is_close score: {score}")
        return score
    except Exception as e:
        print(f"student_is_close error: {e}")
        return 0
    
def generate_hint(question, chunks, hidden_answer, turn_number, history=None, attempt=0, subject="anatomy"):
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
    
    tutor_type = "anatomy tutor for Occupational Therapy students" if subject == "anatomy" else "physics tutor"

    prompt = f"""You are a Socratic {tutor_type}.
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

def masking_pipeline(question, turn_number, history=None, stored_answer=None, subject="anatomy"):
    if history is None:
        history = []
    chunks = retrieve_chunks(question, subject=subject)
    hidden_answer = stored_answer if stored_answer else extract_answer(question, chunks)
    
    for attempt in range(3):
        hint = generate_hint(question, chunks, hidden_answer, turn_number, history, subject=subject)
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
