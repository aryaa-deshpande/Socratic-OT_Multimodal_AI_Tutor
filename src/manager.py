import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import masking_pipeline, retrieve_chunks, extract_answer
from memory import init_db, save_mastery, load_weak_spots
from groq import Groq
from dotenv import load_dotenv
from pipeline import masking_pipeline, student_is_close

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ManagerAgent:
    def __init__(self, student_id):
        self.student_id = student_id
        self.phase = "rapport"
        self.turn_count = 0
        self.current_topic = None
        self.hidden_answer = None
        self.session_history = []
        self.assessment_started = False
        init_db()
        self.weak_spots = load_weak_spots(student_id)
    
    def respond(self, student_message):
        self.session_history.append({"role": "user", "content": student_message})
        
        if self.phase == "rapport":
            response = self.handle_rapport(student_message)
        elif self.phase == "tutoring":
            response = self.handle_tutoring(student_message)
        elif self.phase == "assessment":
            response = self.handle_assessment(student_message)
        else:
            response = "I'm not sure what to do next. What would you like to study?"
        
        self.session_history.append({"role": "assistant", "content": response})
        return response
    
    def handle_rapport(self, message):
        if self.is_content_question(message):
            self.phase = "tutoring"
            self.current_topic = message
            self.turn_count = 0
            return self.handle_tutoring(message)
        
        weak_spot_context = ""
        if self.weak_spots:
            weak_spot_context = f"This student previously struggled with: {', '.join(self.weak_spots)}. Reference this naturally if relevant."
        
        prompt = f"""You are a friendly anatomy tutor starting a session with a student.
    {weak_spot_context}
    The student said: "{message}"
    Respond warmly and briefly. If they seem ready to study, ask what they want to work on today."""

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"handle_rapport error: {e}")
            return "Hey! Good to see you. What would you like to study today?"

    def is_content_question(self, message):
        prompt = f"""You are deciding whether a student's message is an anatomy/science content question or just casual conversation.

    Message: "{message}"

    Is this a content question that requires looking up information from a textbook?
    Reply with only YES or NO."""

        try:
            result = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = result.choices[0].message.content.strip().upper()
            return "YES" in answer
        except Exception as e:
            print(f"is_content_question error: {e}")
            return False
        
    def handle_tutoring(self, message):
        self.turn_count += 1
        
        # check if student is close to the answer
        if self.turn_count > 1 and self.hidden_answer:
            if student_is_close(message, self.hidden_answer):
                self.phase = "assessment"
                return "You're very close! Can you now put it all together and tell me the full answer in your own words?"
        
        hint, self.hidden_answer = masking_pipeline(
            self.current_topic,
            self.turn_count,
            self.session_history,
            self.hidden_answer
        )
        
        return hint