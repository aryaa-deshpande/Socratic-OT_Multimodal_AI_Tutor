# Socratic-OT: Multimodal AI Tutor for Occupational Therapy

A Socratic AI tutor for Occupational Therapy students studying Gross Anatomy and Neuroscience. The system guides students toward answers through structured questioning rather than giving answers directly. All responses are grounded in the OpenStax Anatomy and Physiology 2e textbook using Retrieval-Augmented Generation (RAG).

---

## How it works

1. Student asks an anatomy question
2. System retrieves the most relevant textbook chunks from ChromaDB
3. First LLM call extracts the hidden answer from the chunks, never shown to the student
4. Second LLM call generates a Socratic hint without revealing the answer
5. Guardrail checks the response before sending, rejects and regenerates if the answer leaks
6. Student sees only the Socratic question, not the answer

---

## Project Structure

```
Socratic-OT_Multimodal_AI_Tutor/
├── src/
│   ├── ingest.py        # PDF extraction, cleaning, chunking
│   ├── embed.py         # Embedding chunks and storing in ChromaDB
│   ├── pipeline.py      # Two-step masking pipeline + guardrail
│   ├── app.py           # Streamlit chat UI
│   └── evaluate.py      # RAGAS evaluation script
├── data/
│   ├── raw/             # Place the textbook PDF here (not tracked by git)
│   ├── processed/       # Generated chunks.json (not tracked by git)
│   ├── chromadb/        # Vector database (not tracked by git)
│   └── test_set.json    # 20 Q&A pairs for RAGAS evaluation
├── diagrams/
│   ├── train/           # 20 training diagrams with metadata JSONs
│   └── blind_test/      # 10 held-out diagrams for evaluation
├── transcripts/         # Sample conversation transcripts
├── .env                 # API keys (create this yourself, not tracked by git)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/aryaa-deshpande/Socratic-OT_Multimodal_AI_Tutor.git
cd Socratic-OT_Multimodal_AI_Tutor
```

### 2. Create and activate a virtual environment

Make sure you are using Python 3.10 or above.

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

Get a free API key at console.groq.com. 

### 5. Add the textbook PDF

Download OpenStax Anatomy and Physiology 2e from openstax.org  and place it here:

```
data/raw/Anatomy_and_Physiology_2e.pdf
```

---

## Running the system

### Step 1: Ingest the textbook

Extracts text from the PDF, cleans it, and splits it into chunks.

```bash
python3 src/ingest.py
```

This produces `data/processed/chunks.json` with 6,757 chunks.

### Step 2: Embed and store in ChromaDB

Embeds all chunks using all-MiniLM-L6-v2 and stores them in a local ChromaDB database.

```bash
python3 src/embed.py
```

This will take a few minutes. When done it runs a test retrieval query to confirm everything is working.

### Step 3: Launch the app

```bash
streamlit run src/app.py
```

Open your browser at `http://localhost:8501` and start asking anatomy questions.

---

## Running the evaluation

The evaluation script runs 20 anatomy questions through the pipeline and scores the results using RAGAS faithfulness and answer relevance metrics.

```bash
python3 src/evaluate.py
```

This takes several minutes as it makes LLM calls for each question. Results are printed at the end.

**Baseline results:**

| Metric | Score |
|---|---|
| Faithfulness | 0.6964 |
| Answer Relevance | 0.6996 |

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3.1 8B Instant via Groq API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (local) |
| UI | Streamlit |
| Evaluation | RAGAS |
| PDF processing | PyMuPDF |
| Text splitting | LangChain RecursiveCharacterTextSplitter |

---

## Notes 

- Python 3.10 or above is required, RAGAS has compatibility issues with Python 3.9
- The `data/` folder is gitignored, you need to run `ingest.py` and `embed.py` yourself after cloning
- The `.env` file is gitignored, create your own with your Groq API key
- The `diagrams/blind_test/` folder should not be used until evaluation week

---
