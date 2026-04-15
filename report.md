**Title:** Socratic-OT: A Multimodal AI Tutor for Occupational Therapy Education

---

**1. Problem Statement**

Occupational Therapy students need to master Gross Anatomy and Neuroscience to pass the NBCOT certification exam. These subjects are dense and complex, and students rarely have access to one-on-one support outside of class. Standard AI tools like ChatGPT make this worse , they just give answers directly, which stops students from building the clinical reasoning skills the exam actually tests.

Socratic-OT addresses this by building a tutoring system that refuses to give the answer directly. Instead, it guides students toward the answer through structured questioning, grounded entirely in the OpenStax Anatomy and Physiology 2e textbook. Nothing the system says is made up , every response is derived from retrieved textbook content.

---

**2. Data**

The primary data source is the OpenStax Anatomy and Physiology 2e textbook, which is open-access and freely available as a PDF. The full text was extracted using PyMuPDF, cleaned to remove short lines, page numbers, figure captions, and table of contents entries, then split into chunks using LangChain's RecursiveCharacterTextSplitter with a chunk size of 400 tokens and an overlap of 50 tokens. This produced 6,757 chunks covering the full textbook.

Each chunk was embedded using the sentence-transformers/all-MiniLM-L6-v2 model from HuggingFace and stored in a local ChromaDB vector database with metadata including the chunk index and source identifier. At query time, the student's question is embedded using the same model and the top-5 most semantically similar chunks are retrieved.

For the evaluation, a test set of 20 question-answer pairs was manually constructed covering topics including nerve plexuses, brain regions, the peripheral nervous system, and reflex arcs. Ground truth answers were written by hand based on the textbook content.

A sample of what the retrieved chunks look like for the query "what spinal levels form the brachial plexus":

*Chunk 1: "nerve, which connects to the diaphragm at the base of the thoracic cavity. The other plexus from the cervical level is the brachial plexus. Spinal nerves C4 through T1 reorganize through this plexus to give rise to the nerves of the arms..."*

*Chunk 2: "levels. In this instance, the word plexus is used to describe of the four nerve plexuses, two are found at the cervical level, one at the lumbar level, and one at the sacral level..."*

A sample conversation with the system (Conversation 1 , topic: Brachial Plexus):

*Student: What is Brachial Plexus?*

*System: You seek to understand this term, the Brachial Plexus, which is associated with the cervical region of the spine and the movement of certain structures in the upper body. Have you considered what types of nerves or pathways would logically be involved in controlling the mobility and function of the arms?*

*Student: I think nerves from the spine control the arms, maybe from the neck area?*

*System: You believe that nerves from the spine are involved in controlling the arms and that they originate from the neck area. That's a logical hypothesis given the location and function of the arm. To further explore this idea, can you tell me which specific groups of people might have difficulty using or moving their arms due to a possible problem in the pathway or connection that exists in the neck region?*

*Student: Maybe people with spinal cord injuries in the neck area? Like a cervical injury?*

*System: You have proposed a possibility that involves an injury to the spinal cord specifically in the neck area. To further explore your idea, I'd like to inquire about the types of effects that would be expected if someone were to suffer an injury of this nature. Would you say that such an injury might result in a loss of mobility or sensation in a specific side of the body, rather than affecting the entire body equally?*

The system never revealed the answer directly in any of the turns shown , consistent with the Socratic constraint.

---

**3. Solution Architecture**

The system is composed of five components that form a sequential pipeline from student input to Socratic response.

**Knowledge Base.** The OpenStax textbook is processed offline into 6,757 text chunks, embedded using all-MiniLM-L6-v2, and stored in a local ChromaDB vector database. At inference time, the top-5 most relevant chunks are retrieved for any given student query using cosine similarity between query and chunk embeddings.

**Two-Step Masking Pipeline.** This is the core contribution of the system. When a student submits a question, two sequential LLM calls are made using Llama 3.1 8B via the Groq API. The first call reads the retrieved chunks and extracts the target answer, which is stored as a private variable never shown to the student. The second call takes this hidden answer and the surrounding context and generates a Socratic leading question without revealing the answer or any synonym of it.

**Guardrail.** Before any response is sent to the student, a third LLM call acts as a judge. It checks whether the generated hint directly states or strongly implies the hidden answer. If it does, the response is rejected and regenerated, up to three attempts. This enforces the Socratic constraint at the code level rather than relying solely on prompting.

**Progress Detection.** After the first turn, the system checks whether the student's most recent response is sufficiently close to the hidden answer. If it is, the system pivots to prompt the student to state the full answer in their own words. This check uses a fourth LLM call that evaluates semantic closeness rather than keyword matching.

**Streamlit UI.** The student interacts through a dark-themed Streamlit chat interface. Session state tracks the conversation history, turn number, and stored hidden answer across turns. Student messages are passed to the masking pipeline along with the full conversation history so the system is aware of what has already been said.

The LLM used throughout is Llama 3.1 8B Instant via the Groq API free tier. Embeddings use sentence-transformers/all-MiniLM-L6-v2. The vector database is ChromaDB running locally.

---

**4. Experiments and Results**

Evaluation was conducted using the RAGAS framework, which measures RAG pipeline quality without requiring human annotations. Two metrics were computed on a test set of 20 manually constructed question-answer pairs covering anatomy and neuroscience topics from the OpenStax textbook.

Faithfulness measures whether the system's generated answers are grounded in the retrieved textbook chunks rather than hallucinated from model knowledge. Answer Relevance measures whether the response actually addresses the student's question.

| Metric | Score | Target |
|---|---|---|
| Faithfulness | 0.6964 | > 0.80 |
| Answer Relevance | 0.6996 | > 0.75 |

The evaluation used Llama 3.1 8B via the Groq API as the judge model and all-MiniLM-L6-v2 for embedding-based relevance scoring.

Both scores fall below the target thresholds, which is expected at the baseline stage. The faithfulness score of 0.70 indicates that approximately 70% of the system's answers are directly supported by retrieved textbook chunks, with the remaining 30% showing some degree of hallucination. This is likely caused by noisy chunk retrieval , occasionally off-topic chunks are returned alongside relevant ones, which introduces content not grounded in the correct textbook section.

The answer relevance score of 0.70 reflects that the system's Socratic responses are generally on-topic but sometimes drift, particularly in later turns when the conversation history is longer. This is a known limitation of the current baseline , without a Manager Agent to govern dialogue flow, the system has no mechanism to detect when a conversation is going off track and redirect it.

**Known Limitations of the Baseline**

The current system has no dialogue phase management. There is no Rapport phase to open sessions, no Assessment phase to test clinical reasoning after the student reaches the answer, and no persistent memory across sessions. The conversation continues indefinitely on a single topic with no structured ending. These limitations are architectural , they will be addressed in Milestone 3 through the implementation of a Manager Agent and a SQLite-based session memory system.

Additionally, topic change detection is currently handled by a simple word overlap heuristic, which can fail when a student asks a follow-up question on the same topic using different vocabulary. This will also be replaced with an LLM-based topic change detector in the next milestone.
