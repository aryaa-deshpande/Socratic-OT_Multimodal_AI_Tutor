import ollama
import base64
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image(image_path):
    image_data = load_image(image_path)
    
    response = ollama.chat(
        model="gemma4:e4b",
        messages=[{
            "role": "user",
            "content": """You are looking at an anatomical diagram. Your job is purely descriptive.

Please do the following in order:
1. Read and list EVERY piece of text or label you can see in the image, exactly as written
2. Describe the physical shapes and structures visible — colors, arrangement, connections between parts
3. Describe where in the image each labeled structure appears (top, bottom, left, right, center)
4. Note any arrows, lines, or pointers connecting labels to structures

Do not say what the image 'is'. Just describe what you literally see — text, shapes, colors, spatial arrangement. Be exhaustive.""",
            "images": [image_data]
        }]
    )
    return response["message"]["content"].strip()

def generate_socratic_question(image_description):
    prompt = f"""You are a Socratic anatomy tutor. A student has uploaded an anatomical diagram.

Here is what the diagram shows:
{image_description}

Do two things:
1. Identify the PRIMARY anatomical structure shown — this is your hidden answer
2. Generate ONE Socratic question that guides the student toward identifying it without naming it

Return as JSON with two keys:
- hidden_structure: the primary structure name (keep it short, 1-4 words)
- question: the Socratic question only, no preamble

Return only valid JSON, no extra text."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        result = json.loads(response.choices[0].message.content)
        return result["question"], result["hidden_structure"]
    except Exception as e:
        print(f"generate_socratic_question error: {e}")
        return "What structure do you think is shown in this diagram, and what is its function in the body?", "unknown structure"

def handle_diagram_upload(image_path):
    print("Analyzing image with Gemma...")
    image_description = analyze_image(image_path)
    print(f"Gemma description: {image_description}")
    
    print("Generating Socratic question with Groq...")
    question, hidden_structure = generate_socratic_question(image_description)
    
    fillers = [
        "Interesting diagram! Let me ask you something about it.",
        "Nice image! Here's a question to get you thinking.",
        "Great diagram to work with! Let's explore this together.",
        "Ooh good one! Here's something to consider."
    ]
    import random
    filler = random.choice(fillers)
    
    return f"{filler}\n\n{question}", image_description, hidden_structure

if __name__ == "__main__":
    test_image = "diagrams/train/lumbar_plexus.png"
    if os.path.exists(test_image):
        question, description = handle_diagram_upload(test_image)
        print("\nSocratic question for student:")
        print(question)
    else:
        print("Add a test image to diagrams/train/ to test this")