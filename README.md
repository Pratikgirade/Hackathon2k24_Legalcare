# 🤖LegalCareBot: Your AI Legal Companion ⚖️📜


#### Team Name : **POWER_CODERS(SPD)**

## 👩‍💻 Team members

- **Pratik Girade** - pratikgirade9999@gmail.com
- **Suman Kayal** - kayalsuman876@gmail.com
- **Dipmay Bisqwas** - dipmay231biswas@gmail.com



----------------------------------------------------------
### Report & Other Document: Available in github Repo [Legalcare.pdf]
----------------------------------------------------------
## What is LegalCareBot?
LegalCareBot 🤖 is a sophisticated chatbot designed to provide personalized legal advice and dispute resolution. Leveraging a user's natural language queries, it offers recommendations on legal matters, case law summaries, and advice directly from a curated legal knowledge base.
## How It Is Built?

**Legalcare** utilizes several advanced technologies:

- **Streamlit**: Delivers an interactive user interface for real-time user interaction.
- **LangChain**: Integrates the language model (LLM) with the vector database.
- **RecursiveTextSplitter**: Splits the PDF containing fashion data into manageable chunks.
- **Hugging Face's sentence-transformers/all-MiniLM-L6-v2 Model**: Creates text embeddings.
- **Chroma Vector Database**: Efficiently stores and retrieves text embeddings for semantic search.
- **Meta-Llama-3-8B-Instruct with HuggingFace API**: Provides the processing power for natural language understanding.
- **Directive-based prompt pattern**: Guides the language model on how to generate appropriate responses based on the query context and user interaction directives.


## Instructions on How to Setup and Run

### Step 1: Install Required Python Libraries

Install the necessary libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```
### Step 2: Generate and Store Embeddings
There are two Python files: `embeddings_generator.py` and `WaLL-E.py`.

1. **Set Up API Keys**: Ensure your HuggingFace API Token is in the .env file
```bash
HUGGINGFACEHUB_API_TOKEN = "<HUGGINGFACEHUB_API_TOKEN>"
```
2. **Generate Embeddings**: Run `vector_embeddings.py` to process the fashion data PDF and store the results in the Chroma Vector Database in the **"data"** directory.
```bash 
python vector_embeddings.py
```

### Step 3: Launch FashionBot
After setting up the embeddings, launch the FashionBot interface by running:
```bash
streamlit run FashionBot.py
```

### Step 4: Testing and Evaluation
LegalCareBot 🤖 can handle diverse queries such as:

Detailed case summaries for specific legal disputes.
Advice on legal proceedings and potential outcomes.
Recommendations for resolving disputes based on provided legal documents.
Legal tips for understanding punishments and fines.

Examples include:

--  ***Simple Query***: "What are the key points of the Partnership Act?"
--  ***Complex Query***: "Can you summarize the case of Appellants vs Abdul Manan Khan?"
--  ***Scenario-Based Query***: "I have a property dispute that involves multiple parties. Could you please suggest the best approach to resolve this?"
--  ***Irrelevant/Out-of-Scope Queries***: "What is the best recipe for a chocolate cake?"
--  ***Misspelled Queries***: "partnrship act lgal deetails?"
