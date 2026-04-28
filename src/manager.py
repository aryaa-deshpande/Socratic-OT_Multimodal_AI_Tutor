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
        self.assessment_attempt = 0
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
            acknowledgment = self.get_acknowledgment('rapport_to_tutoring')
            first_hint = self.handle_tutoring(message)
            return f"{acknowledgment}\n\n{first_hint}"
        
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
        print(f"DEBUG hidden_answer: {self.hidden_answer}")
        print(f"DEBUG student message: {message}")
        
        # check if student is close to the answer
        if self.turn_count > 1 and self.hidden_answer:
            if student_is_close(message, self.hidden_answer):
                self.phase = "assessment"
                acknowledgment = self.get_acknowledgment("tutoring_to_assessment")
                return f"{acknowledgment}\n\nYou're very close! Can you now put it all together and tell me the full answer in your own words?"
        
        hint, self.hidden_answer = masking_pipeline(
            self.current_topic,
            self.turn_count,
            self.session_history,
            self.hidden_answer
        )
        
        return hint
        
    def handle_assessment(self, message):
    
        if self.assessment_attempt == 0:
            self.assessment_attempt = 1
            
            chunks = retrieve_chunks(self.current_topic)
            context = "\n\n".join(chunks[:3])
            
            prompt = f"""You are an Occupational Therapy anatomy tutor.

    The student has just correctly identified the following concept:
    Topic: {self.current_topic}
    Answer: {self.hidden_answer}

    Textbook context:
    {context}

    Generate ONE clinical scenario question for an OT student at an introductory level.
    The scenario should:
    - Describe a real patient situation an OT might encounter
    - Ask the student to apply their understanding of {self.current_topic} to explain what is happening or what they would expect
    - Be specific enough that there is a clear correct answer grounded in the textbook content
    - NOT restate or hint at the answer

    Example format: "A patient presents with X. Based on what you know, what would you expect and why?"
    """

            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )
                scenario = response.choices[0].message.content.strip()
                acknowledgment = self.get_acknowledgment("tutoring_to_assessment")
                return f"{acknowledgment}\n\nNow let's see if you can apply what you've learned. Here's a clinical scenario:\n\n{scenario}"
            except Exception as e:
                print(f"handle_assessment scenario error: {e}")
                return f"Now that you have a better understanding of {self.current_topic}, how would you apply this in a clinical setting? Describe a patient scenario and walk me through your reasoning."
            
        elif self.assessment_attempt == 1:
            self.assessment_attempt = 2
            mastery = self.run_llm_judge(message)
            
            if mastery["score"] == "weak":
                return f"Not quite, {mastery['feedback']}\n\nGive it another try! What would you expect to see clinically?"
            
            else:
                save_mastery(
                    student_id=self.student_id,
                    topic=self.current_topic,
                    score=mastery["score"],
                    tutor_note=mastery["tutor_note"],
                    student_summary=mastery["student_summary"]
                )
                self.phase = "rapport"
                self.assessment_attempt = 0
                self.hidden_answer = None
                self.turn_count = 0
                self.current_topic = None
                acknowledgment = self.get_acknowledgment("assessment_to_rapport")
                return f"{acknowledgment}\n\nHere's how you did:\n\n{mastery['feedback']}\n\n**Summary:** {mastery['student_summary']}\n\nFeel free to ask me about another topic whenever you're ready!"
        
        elif self.assessment_attempt == 2:
            mastery = self.run_llm_judge(message)
            save_mastery(
                student_id=self.student_id,
                topic=self.current_topic,
                score=mastery["score"],
                tutor_note=mastery["tutor_note"],
                student_summary=mastery["student_summary"]
            )
            completed_topic = self.current_topic
            self.phase = "rapport"
            self.assessment_attempt = 0
            self.hidden_answer = None
            self.turn_count = 0
            self.current_topic = None
            acknowledgment = self.get_acknowledgment("assessment_to_rapport")
            return f"{acknowledgment}\n\nHere's how you did:\n\n{mastery['feedback']}\n\n**Summary:** {mastery['student_summary']}\n\nThis topic needs a bit more practice. When you start a new session, we'll revisit {completed_topic} to build on what you've learned today."

    def run_llm_judge(self, student_response):
        chunks = retrieve_chunks(self.current_topic)
        context = "\n\n".join(chunks[:3])
        
        prompt = f"""You are evaluating an OT student's clinical reasoning.

    Topic: {self.current_topic}
    Gold standard answer: {self.hidden_answer}
    Textbook context: {context}
    Student's response: {student_response}

    Evaluate the student and return a JSON object with exactly these four keys:
    - score: one of "strong", "partial", or "weak"
    - tutor_note: one sentence for the tutor about what to revisit (student never sees this)
    - student_summary: one sentence telling the student how they did overall
    - feedback: 3-4 sentences of real feedback — what they got right, what they missed, and the correct clinical reasoning explained clearly

    Return only valid JSON, no extra text."""

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"run_llm_judge error: {e}")
            return {
                "score": "partial",
                "tutor_note": f"Could not evaluate student response on {self.current_topic}",
                "student_summary": "Good effort on that topic. Keep practicing!",
                "feedback": "I wasn't able to fully evaluate your response this time. Let's keep going and come back to this topic."
            }
        
    def get_acknowledgment(self, transition_type):
        messages = {
            "rapport_to_tutoring": "Generate ONE short warm sentence (max 10 words) acknowledging that the student asked a good question. Don't answer it.",
            "tutoring_to_assessment": "Generate ONE short encouraging sentence (max 10 words) congratulating the student for figuring out the answer. Keep it natural.",
            "assessment_to_rapport": "Generate ONE short warm closing sentence (max 10 words) wrapping up the topic and encouraging the student to keep going."
        }
        
        prompt = f"""You are a friendly anatomy tutor. {messages[transition_type]}
    Examples of good responses: "Great question!", "Nice work getting there!", "Good session today!"
    Return only the sentence, nothing else."""

        try:
            result = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            print(f"get_acknowledgment error: {e}")
            fallbacks = {
                "rapport_to_tutoring": "Great question! Let's explore this together.",
                "tutoring_to_assessment": "Nice work! Now let's apply what you've learned.",
                "assessment_to_rapport": "Good effort today! Keep it up."
            }
            return fallbacks[transition_type]