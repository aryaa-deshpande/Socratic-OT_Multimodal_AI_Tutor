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
    def __init__(self, student_id, subject="anatomy"):
        self.student_id = student_id
        self.subject = subject
        self.phase = "rapport"
        self.turn_count = 0
        self.current_topic = None
        self.hidden_answer = None
        self.session_history = []
        self.assessment_attempt = 0
        init_db()
        self.weak_spots = load_weak_spots(student_id, subject=subject)
    
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
            self.current_topic = message.strip().strip('"').strip("'")
            self.turn_count = 0
            acknowledgment = self.get_acknowledgment("rapport_to_tutoring")
            first_hint = self.handle_tutoring(message)
            return f"{acknowledgment}\n\n{first_hint}"
        
        subject_label = "anatomy" if self.subject == "anatomy" else "physics"
        
        weak_spot_context = ""
        if self.weak_spots and self.subject == "anatomy":
            weak_spot_context = f"This student previously struggled with: {', '.join(self.weak_spots)}. Mention this naturally in one sentence."
        
        if weak_spot_context:
            prompt = f"""You are a friendly {subject_label} tutor.
    {weak_spot_context}
    The student said: "{message}"
    Respond in 2 sentences maximum. Greet them warmly, mention their weak spot naturally, and ask what they want to work on today. Do not invent any prior sessions or topics."""
        else:
            prompt = f"""You are a friendly {subject_label} tutor.
    The student said: "{message}"
    Respond in 1 sentence. Just greet them warmly and ask what they want to work on today. Do not reference any prior sessions or topics."""

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
        
        if self.turn_count > 1 and self.hidden_answer:
            score = student_is_close(message, self.hidden_answer)
            
            if score >= 7:
                # student is close transition to assessment
                self.phase = "assessment"
                acknowledgment = self.get_acknowledgment("tutoring_to_assessment")
                return f"{acknowledgment}\n\nYou're very close! Can you now put it all together and tell me the full answer in your own words?"
            
            elif score >= 5:
                # student is on the right track encouraging nudge
                subject_label = "anatomy" if self.subject == "anatomy" else "physics"
                prompt = f"""You are a Socratic {subject_label} tutor. The student is on the right track but needs to be more specific.

    Hidden answer (do NOT reveal): {self.hidden_answer}
    Student's response: {message}

    Generate ONE short encouraging response that:
    - Acknowledges they are heading in the right direction
    - Asks them to be more specific about one key aspect they are missing
    - Does not reveal the answer or any synonym of it

    Keep it to 2 sentences maximum."""

                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    print(f"handle_tutoring mid-range error: {e}")
                    return "You're on the right track! Can you be more specific about the key components involved?"
        
        # score 0-4 or turn 1 normal Socratic hint
        hint, self.hidden_answer = masking_pipeline(
            self.current_topic,
            self.turn_count,
            self.session_history,
            self.hidden_answer,
            subject=self.subject
        )
        
        return hint
        
    def handle_assessment(self, message):
    
        if self.assessment_attempt == 0:
            self.assessment_attempt = 1
            
            chunks = retrieve_chunks(self.current_topic)
            context = "\n\n".join(chunks[:3])
            
            if self.subject == "anatomy":
                scenario_instruction = f"""Generate ONE clinical scenario question for an OT student at an introductory level.
            The scenario should:
            - Describe a real patient situation an OT might encounter
            - Ask the student to apply their understanding of {self.current_topic} to explain what is happening or what they would expect
            - Be specific enough that there is a clear correct answer grounded in the textbook content
            - NOT restate or hint at the answer
            - DO NOT include the answer or any explanations of what should happen in the scenario

            Example format: "A patient presents with X. Based on what you know, what would you expect and why?" """
                
            else:
                scenario_instruction = f"""Generate ONE real-world physics application question at an introductory level.
            The question should:
            - Describe a practical everyday scenario where {self.current_topic} applies
            - Ask the student to apply the physics concept to explain or calculate something
            - Be specific enough that there is a clear correct answer grounded in the textbook content
            - NOT restate or hint at the answer
            - DO NOT include the answer, any calculations, or worked solutions in the question

            Example format: "A ball is dropped from X height. Using what you know, what would you expect and why?" """

            prompt = f"""You are a {'clinical Occupational Therapy anatomy' if self.subject == 'anatomy' else 'physics'} tutor.

            The student has just correctly identified the following concept:
            Topic: {self.current_topic}
            Answer: {self.hidden_answer}

            Textbook context:
            {context}

            {scenario_instruction}"""

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
                retry_prompt = "Give it another try! What would you expect to see clinically?" if self.subject == "anatomy" else "Give it another try! How would you apply this concept in a real scenario?"
                return f"Not quite — {mastery['feedback']}\n\n{retry_prompt}"
            else:
                save_mastery(
                    student_id=self.student_id,
                    topic=self.current_topic,
                    score=mastery["score"],
                    tutor_note=mastery["tutor_note"],
                    student_summary=mastery["student_summary"],
                    subject=self.subject
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
                student_summary=mastery["student_summary"],
                subject=self.subject
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
        
        prompt = f"""You are evaluating a {'OT' if self.subject == 'anatomy' else 'physics'} student's reasoning.

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
            result = json.loads(response.choices[0].message.content)

            # handle feedback being a list or a string
            feedback = result.get("feedback", "")
            if isinstance(feedback, list):
                feedback = " ".join(feedback)
            result["feedback"] = feedback

            return result
        except Exception as e:
            print(f"run_llm_judge error: {e}")
            return {
                "score": "partial",
                "tutor_note": f"Could not evaluate student response on {self.current_topic}",
                "student_summary": "Good effort on that topic. Keep practicing!",
                "feedback": "I wasn't able to fully evaluate your response this time. Let's keep going and come back to this topic."
            }
        
    def get_acknowledgment(self, transition_type):
        subject_label = "anatomy" if self.subject == "anatomy" else "physics"
        
        messages = {
            "rapport_to_tutoring": f"Generate ONE short warm sentence (max 10 words) acknowledging that the student asked a good {subject_label} question. Don't answer it.",
            "tutoring_to_assessment": f"Generate ONE short encouraging sentence (max 10 words) congratulating the student for figuring out the {subject_label} answer. Keep it natural.",
            "assessment_to_rapport": f"Generate ONE short warm closing sentence (max 10 words) wrapping up the {subject_label} topic and encouraging the student to keep going."
        }
        
        prompt = f"""You are a friendly {subject_label} tutor. {messages[transition_type]}
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
                "rapport_to_tutoring": f"Great {subject_label} question! Let's explore this together.",
                "tutoring_to_assessment": f"Nice work! Now let's apply what you've learned.",
                "assessment_to_rapport": "Good effort today! Keep it up."
            }
            return fallbacks[transition_type]