# Socratic-OT: Multimodal AI Tutor for Occupational Therapy

A Socratic AI tutor for Occupational Therapy students studying Gross Anatomy and Neuroscience. The system guides students toward answers through structured questioning rather than giving answers directly. All responses are grounded in the OpenStax Anatomy and Physiology 2e textbook using Retrieval-Augmented Generation (RAG).

Also supports Physics as a second domain to demonstrate generalizability.

---

## How it works

1. Student enters their name and selects a subject (Anatomy or Physics)
2. Student chooses Text Chat or Diagram Chat
3. **Text Chat:** Student asks a question → system retrieves textbook chunks → extracts hidden answer → generates Socratic hint without revealing it → guardrail checks the response → student sees only the Socratic question
4. **Diagram Chat:** Student uploads an anatomical diagram → Gemma 4 analyzes the image → Groq generates a Socratic question about it → conversation continues through tutoring and assessment
5. After the student reaches the answer, a clinical scenario is presented for application
6. Session results are saved to SQLite memory and visible in the Progress dashboard

---

## Project Structure

```
Socratic-OT_Multimodal_AI_Tutor/
├── src/
│   ├── ingest.py              # PDF extraction, cleaning, chunking
│   ├── embed.py               # Embedding chunks into ChromaDB (anatomy)
│   ├── pipeline.py            # Two-step masking pipeline + guardrail
│   ├── manager.py             # Manager Agent - all three phases
│   ├── memory.py              # SQLite session memory
│   ├── vlm.py                 # Gemma 4 image analysis + Socratic question generation
│   ├── app.py                 # Streamlit multi-page UI
│   ├── evaluate.py            # RAGAS evaluation script
│   └── generalized/
│       ├── ingest_physics.py  # Physics PDF ingestion
│       └── embed_physics.py   # Physics embedding into ChromaDB
├── data/
│   ├── raw/                   # Place PDFs here (not tracked by git)
│   ├── processed/             # Generated chunks (not tracked by git)
│   ├── chromadb/              # Vector database (not tracked by git)
│   └── test_set.json          # 20 Q&A pairs for RAGAS evaluation
├── diagrams/
│   ├── train/                 # 20 training diagrams with metadata JSONs
│   └── blind_test/            # 10 held-out diagrams for evaluation
├── transcripts/               # Sample conversation transcripts
├── .env                       # API keys - create this yourself, not tracked
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### Requirements
- Python 3.10 or above (required - RAGAS has compatibility issues with Python 3.9)
- Ollama 0.20.3 or above (required for Gemma 4)

### 1. Clone the repo

```bash
git clone https://github.com/aryaa-deshpande/Socratic-OT_Multimodal_AI_Tutor.git
cd Socratic-OT_Multimodal_AI_Tutor
```

### 2. Create and activate a virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the root of the repo:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at console.groq.com

### 5. Add the textbook PDFs

**Anatomy (required):**
Download OpenStax Anatomy and Physiology 2e from openstax.org (free) and place it here:
https://openstax.org/details/books/anatomy-and-physiology-2e

```
data/raw/Anatomy_and_Physiology_2e.pdf
```

**Physics (required for generalizability demo):**
Download OpenStax University Physics Volume 1 from openstax.org (free) and place it here:
https://openstax.org/details/books/university-physics-volume-1
```
data/raw/Physics.pdf
```

### 6. Install and start Ollama

Download Ollama from ollama.com/download and install it.

Start the Ollama server:
```bash
ollama serve
```

In a new terminal tab, pull Gemma 4:
```bash
ollama pull gemma4:e4b
```

This will take a few minutes (9.6 GB download).

---

## Building the Knowledge Base

Run these once after cloning. You need the PDFs in `data/raw/` first.

### Anatomy

```bash
python3 src/ingest.py
python3 src/embed.py
```

This produces `data/processed/chunks.json` (6,757 chunks) and stores embeddings in ChromaDB collection "anatomy".

### Physics (for generalizability demo)

```bash
python3 src/generalized/ingest_physics.py
python3 src/generalized/embed_physics.py
```

This produces `data/processed/chunks_physics.json` (3,033 chunks) and stores embeddings in ChromaDB collection "physics".

---

## Running the App

Make sure Ollama is running (`ollama serve`) in a separate terminal tab before starting the app.

```bash
streamlit run src/app.py
```

Open your browser at `http://localhost:8501`.

**App flow:**
1. Enter your name
2. Select subject - 🧬 Anatomy or ⚛️ Physics
3. Choose a mode:
   - 💬 **Text Chat** - ask anatomy or physics questions
   - 🖼️ **Diagram Chat** - upload an anatomical diagram
   - 📊 **My Progress** - view session history and weak spots

---

## Running the Evaluation

### RAGAS Evaluation

Make sure the knowledge base is built and the `.env` file has your Groq API key.

```bash
python3 src/evaluate.py
```

Results are printed at the end.

**Final results:**

| Metric | Baseline | Final | Target |
|---|---|---|---|
| Faithfulness | 0.6964 | 0.8048 | > 0.80 |
| Answer Relevance | 0.6996 | 0.7347 | > 0.75 |

### Multimodal Blind Test

Run Gemma 4 on each of the 10 held-out diagrams in `diagrams/blind_test/` and record whether it correctly identifies the primary structure. Compare against the ground truth in the metadata files.

```bash
python3 -c "
import sys
sys.path.append('src')
from vlm import handle_diagram_upload
import os

blind_test_dir = 'diagrams/blind_test'
images = [f for f in os.listdir(blind_test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img in images:
    path = os.path.join(blind_test_dir, img)
    question, description, hidden_structure = handle_diagram_upload(path)
    print(f'Image: {img}')
    print(f'Identified: {hidden_structure}')
    print('---')
"
```

Compare each identified structure against the ground truth in `diagrams/train/metadata/`.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.1 8B Instant via Groq API |
| VLM | Gemma 4 E4B via Ollama (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (local, persistent) |
| Session Memory | SQLite |
| UI | Streamlit |
| Evaluation | RAGAS |
| PDF processing | PyMuPDF |
| Text splitting | LangChain RecursiveCharacterTextSplitter |

---

## Notes 

- **Python 3.10+ is required** - do not use 3.9
- **Ollama must be running** before starting the app or VLM will fail
- **Gemma 4 E4B requires Ollama 0.20.3+** - update Ollama if needed
- **The `data/` folder is gitignored** - run ingest and embed yourself after cloning
- **The `.env` file is gitignored** - create your own with your Groq API key
- **`diagrams/blind_test/`** - do not look at metadata until after running the blind test
- **`data/memory.db`** is gitignored - each person has their own local session history

