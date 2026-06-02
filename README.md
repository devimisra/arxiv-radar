<a href="https://github.com/devimisra/arxiv-radar/actions/workflows/test.yml"><img src="https://github.com/devimisra/arxiv-radar/actions/workflows/test.yml/badge.svg?branch=main" alt="CI/CD Pipeline"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>

**Taming the ArXiv Firehose: Building a Serverless RAG Pipeline for Paper Discovery**

[Live Demo](https://arxiv-radar.streamlit.app)

**ArXiv Radar** is a domain-agnostic data pipeline designed to filter the daily noise of scientific publications. Moving beyond traditional keyword matching, this tool uses 384-dimensional vector embeddings to score ArXiv papers based on semantic relevance, and open-weights Large Language Models (LLMs) to generate strict, one-sentence technical insights for top-ranking papers.

<p align="center">
  <img src="architecture.png" alt="ArXiv Radar Architecture" width="150">
</p>

## How to Use ArXiv Radar
1. **Provide a Token (BYOK):** Navigate to your Hugging Face account settings, generate a "Read" access token, and securely enter it in the app's sidebar. The application is stateless; your token is only used for the active session and is never stored.
2. **Set Your Target Category:** Enter an ArXiv category code (e.g., `cs.LG` for Machine Learning, `astro-ph.CO` for Cosmology).
3. **Define Research Interests:** Input highly specific technical phrases or methodologies. The system uses semantic search, so descriptive context works better than single-word keywords.
4. **Select an AI Model:** Choose an open-weights model (e.g., Llama 3.1, Gemma 2, or Qwen) to generate the technical hooks.
5. **Scan and Export:** Execute the scan to filter the latest papers. Review the results in the UI and download your curated daily reading list as a clean Markdown (`.md`) file.

## Formulating Your Research Topics
Because this tool uses semantic vector embeddings rather than simple CTRL+F keyword matching, **context matters**. To get the best results, you should be highly specific in your research topics.

* **Bad Example:** `Machine Learning, AI, Biology` 
  *(Too broad. The model will return generic papers with low relevance scores).*
* **Good Example:** `parameter-efficient fine-tuning for large language models, LoRA, QLoRA, quantization techniques` 
  *(Highly specific. The embedding model will capture the mathematical meaning of these concepts and find papers discussing them, even if those exact words are not in the abstract).*

## Architecture & Production Constraints
This project was designed to demonstrate clean separation of concerns, cost-efficient deployment, and rigorous evaluation metrics.

* **Multi-Tiered BYOK Architecture:** Implements a hybrid compute model. Users can rely on zero-configuration defaults or inject personal tokens to access massive 70B parameter models, shifting heavy inference loads back to individual quotas and preventing central API exhaustion.
* **Infrastructure Resilience:** Utilizes aggressive exponential backoff, retry logic, and memory caching (`@st.cache_data`) to prevent `HTTP 429 (Too Many Requests)` blocks from the ArXiv API in shared-IP cloud environments.
* **Decoupled for Orchestration:** The ingestion logic, vector math (NumPy), and presentation layer (`main.py`) are strictly separated. In an enterprise topology, the backend modules can be lifted out of the Streamlit loop and scheduled via workflow orchestrators like **Apache Airflow**.

## Automated CI/CD Testing (`test_pipeline.py`)
A parameterized `pytest` suite runs automatically via GitHub Actions on every code push, treating the RAG pipeline with standard software engineering rigor:

* **Quantifying Retrieval Accuracy:** Evaluates the in-memory NumPy similarity search against an annotated "golden dataset." The pipeline benchmarks ranking performance using **Mean Reciprocal Rank (MRR)** and **Hit Rate@K**, alerting engineers to contextual drift if ground-truth papers fall below rank thresholds.
* **LLM Constraint Adherence:** Verifies the generation phase strictly adheres to structural formatting (`TAGS:` and `HOOK:`) and one-sentence maximums without hallucinating conversational filler.
## Technology Stack
* **Frontend/Hosting:** Streamlit Community Cloud
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), `numpy`
* **Generative AI:** Hugging Face Inference API (Llama 3.1, Gemma 2, Qwen)
* **Data Ingestion:** ArXiv Python API
* **CI/CD:** GitHub Actions, Pytest

## Running Locally

1. Clone the repository and navigate to the project directory:
```bash
   git clone https://github.com/YOUR_USERNAME/arxiv-radar.git
   cd arxiv-radar
```
2. Install the required dependencies:
```bash
   pip install -r requirements.txt
```
3. Set up your local secrets by creating a `.streamlit/secrets.toml` file and adding your Hugging Face API key:
```toml
   HF_TOKEN = "your_hf_token_here"
```
4. Run the Streamlit application:
```bash
   streamlit run main.py
```
5. Run the evaluation suite (`test_pipeline.py`):
```bash
   export HF_TOKEN="your_hf_token_here"
   pytest test_pipeline.py -v
```