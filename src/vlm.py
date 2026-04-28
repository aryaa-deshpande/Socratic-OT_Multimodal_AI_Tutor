import ollama
import base64
import os

def load_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def identify_structure(image_path):
    image_data = load_image(image_path)
    
    response = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": "Look at this anatomical diagram. What is the primary anatomical structure shown? Return only the structure name, nothing else.",
            "images": [image_data]
        }]
    )
    return response["message"]["content"].strip()

if __name__ == "__main__":
    # test with a sample image path
    test_image = "diagrams/train/test.jpg"
    if os.path.exists(test_image):
        structure = identify_structure(test_image)
        print("Identified structure:", structure)
    else:
        print("Add a test image to diagrams/train/ to test this")