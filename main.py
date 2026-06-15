import streamlit as st
import arxiv
from datetime import datetime, timedelta, timezone
from utils import load_embedding_model, get_relevance_score, generate_insights, generate_markdown_report

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="ArXiv Radar", layout="wide")

# --- 2. CACHING ---
@st.cache_resource
def get_cached_model():
    return load_embedding_model()

# Correctly assign the cached model
embedding_model = get_cached_model()

@st.cache_data(ttl=3600) # Cache the raw ArXiv results for 1 hour to prevent 429 errors
def fetch_arxiv_papers(category, max_results=100):
    client = arxiv.Client(page_size=100, delay_seconds=10, num_retries=15)
    search = arxiv.Search(query=f"cat:{category}", max_results=max_results, sort_by=arxiv.SortCriterion.LastUpdatedDate)
    return list(client.results(search))

# --- 3. MAIN INTERFACE ---
st.title("ArXiv Research Radar")
st.markdown("""
Welcome to **ArXiv Radar**, your automated research assistant. This tool scans ArXiv for the latest papers, filters them based on the semantic meaning of your interests, and uses AI to extract technical insights.

**System Reliability:**
To ensure high-signal outputs, this architecture is continuously evaluated via an automated CI/CD pipeline. Using a baseline "golden dataset," the vector retrieval layer is monitored for context drift using **Mean Reciprocal Rank (MRR)** and **Hit Rate@K** metrics. Furthermore, the generative phase is verified to achieve 100% adherence to strict formatting constraints (maximum one-sentence hooks) and zero hallucination of conversational filler. Advanced models (Llama 3.1, Gemma 2) inherit this strict architecture and are seamlessly supported via the BYOK interface.
""")

with st.expander("How to use this tool"):
    st.markdown("""
    1. **Bring Your Own Key (BYOK):** Enter your Hugging Face Token in the sidebar. Your token is never stored or saved by this application.
        * **No Token?** Leave it blank to use the shared demo token (open models only, e.g., Qwen 2.5).
        * **Want Gated Models (e.g., Llama 3.1, Gemma 2)?** 1. Visit the model's page on Hugging Face and click **"Acknowledge License"**.
            2. Go to your HF Profile Settings > Access Tokens and generate a **"Read"** token.
            3. Paste that token into the sidebar.

    2. **Set Your Filters & Topics:** Define your ArXiv category and input your specific research interests. Adjust the **Min Relevance Score** to fine-tune your results. Be careful: set it too high and you might filter out everything; set it too low and the results will lose their specific focus.
        * **Crucial Tip for Topics:** Because this tool uses semantic vector embeddings, you should be as specific as possible. Do not just type "AI" or "Biology". 
        * **Good Example:** "parameter-efficient fine-tuning, LoRA, QLoRA, model quantization"
        * **Good Example:** "exoplanet transit photometry, atmospheric characterization of hot jupiters"
        * The embedding model will capture the mathematical meaning of these phrases and find highly relevant papers even if those exact words are missing from the abstract.

    3. **Choose Your AI:** Select a standard open-weights model or enter a custom Hugging Face Model ID for the summarization step.

    4. **Scan:** Run a live scan and download the filtered results as a Markdown report for your notes.
    """)

with st.expander("ArXiv Category Cheat Sheet"):
    st.markdown("""
    **Computer Science (cs)**
    * cs.AI: Artificial Intelligence
    * cs.LG: Machine Learning
    * cs.CV: Computer Vision
    * cs.CL: Computation & Language (NLP/LLMs)
    * cs.CR: Cryptography & Security
    
    **Physics & Astrophysics (astro-ph, quant-ph, etc.)**
    * astro-ph.HE: High Energy Astrophysical Phenomena
    * astro-ph.CO: Cosmology & Nongalactic Astrophysics
    * quant-ph: Quantum Physics
    * cond-mat.mtrl-sci: Materials Science
    
    **Mathematics (math)**
    * math.PR: Probability
    * math.ST: Statistics Theory
    * math.CO: Combinatorics
    
    **Quantitative Biology (q-bio)**
    * q-bio.NC: Neurons and Cognition
    * q-bio.GN: Genomics
    
    **Finance & Economics (q-fin, econ)**
    * q-fin.ST: Statistical Finance
    * econ.EM: Econometrics

    *Need a highly specific field? Find all codes on the Official ArXiv Taxonomy page.*
    """)

with st.expander("How Relevance Scoring Works (The Math)"):
    st.markdown("""
    To move beyond the limitations of simple keyword matching, ArXiv Radar utilizes **Semantic Search** driven by high-dimensional vector math.

    **1. Vector Embeddings (The Mapping Phase)**
    We use a transformer model (`all-MiniLM-L6-v2`) to convert both your research interests and the ArXiv abstracts into **embeddings**. Think of an embedding as a coordinate in a 384-dimensional space. Words or sentences with similar meanings are placed physically close to each other.

    **2. Cosine Similarity (The Radar Math)**
    To determine relevance, we measure the mathematical angle between your "Interest Vector" and the "Abstract Vector" using **Cosine Similarity**. 
    """)
    
    st.markdown("$$ \\text{similarity} = \\cos(\\theta) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|} $$")
    
    st.markdown("""
    * **$\\mathbf{A} \\cdot \\mathbf{B}$** is the dot product of the vectors.
    * **$\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|$** represents the product of their magnitudes.

    **3. Why this is superior to Keyword Search:**
    * **Synonym Awareness:** If you search for "Large Language Models," the radar will still catch a paper titled "Scaling Transformer-based Architectures."
    * **Contextual Understanding:** The Radar evaluates your interests as a multi-pronged probe. It calculates a similarity score for *every* topic you provide and surfaces the paper based on its strongest match.
    """)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("[View Source Code on GitHub](https://github.com/devimisra/arxiv-radar.git)")
    st.header("Search & Security Settings")
    
    backend_token = st.secrets.get("HF_TOKEN", "")
    user_token = st.text_input(
        "Hugging Face Token", 
        type="password", 
        help="Get one at huggingface.co/settings/tokens",
        key="hf_token_input"
    )
    
    active_token = user_token if user_token else backend_token
    
    if not active_token:
        st.error("Token required for AI insights.")
    elif user_token:
        st.success("Custom AI Engine Authenticated")
    else:
        st.warning("Using shared Demo Token. You may experience rate limits. For heavy use, please enter your own token or clone the repository.")
    

    model_options = [
        "Qwen/Qwen2.5-7B-Instruct", 
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "Custom (Enter Model ID)"
    ]
    
    model_selection = st.selectbox("AI Summary Model", model_options, index=0, key="model_select_box")
    
    if model_selection == "Custom (Enter Model ID)":
        model_choice = st.text_input("Enter Model ID", placeholder="e.g. meta-llama/Llama-3.1-70B-Instruct", key="custom_model_input")
    else:
        model_choice = model_selection

    category = st.text_input("ArXiv Category", value="astro-ph.HE", key="arxiv_cat_input")
    interest_text = st.text_area("Research Interests (Use specific technical phrases)", value="supernova, supernova remnants, gamma ray bursts", key="interests_input")
    user_interests = [i.strip() for i in interest_text.split(",") if i.strip()]
    days_val = st.slider("Days Back", 1, 14, 7, key="days_slider")
    score_threshold = st.slider("Min Relevance Score", 0.0, 1.0, 0.50, 0.05, key="score_slider")
    
    run_btn = st.button("Start Radar Scan", type="primary", key="run_scan_btn", use_container_width=True)

# --- 5. MAIN EXECUTION ---
# Initialize session state to hold our results so they survive button clicks (like downloading)
if "scan_complete" not in st.session_state:
    st.session_state.scan_complete = False
if "filtered_papers" not in st.session_state:
    st.session_state.filtered_papers = []

if run_btn:
    if not active_token:
        st.error("Please provide a Hugging Face Token.")
    elif model_selection == "Custom (Enter Model ID)" and not model_choice:
        st.error("Please enter a Model ID.")
    elif not user_interests:
        st.error("Please enter at least one research interest.")
    else:
        # Reset state for a new scan
        st.session_state.filtered_papers = []
        st.session_state.scan_complete = False
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_val)
        
        try:
            # Use st.status for the "Illusion of Labor" - showing granular enterprise loading states
            with st.status("Initializing ArXiv Radar...", expanded=True) as status:
                
                status.update(label=f"Fetching {category} abstracts from ArXiv...", state="running")
                results = fetch_arxiv_papers(category, max_results=100)
                
                if not results:
                    status.update(label="No papers found.", state="error")
                    st.warning("No papers found in the specified timeframe.")
                else:
                    status.update(label=f"Computing in-memory vector embeddings for {len(results)} papers...", state="running")
                    
                    # Pre-filter papers using the embedding model BEFORE calling the LLM
                    top_candidates = []
                    for paper in results:
                        pub_date = paper.updated.replace(tzinfo=timezone.utc)
                        if pub_date < cutoff: continue
                        
                        score = get_relevance_score(paper.summary, user_interests, embedding_model)
                        if score >= score_threshold:
                            top_candidates.append({
                                "paper_obj": paper,
                                "pub_date": pub_date.strftime('%Y-%m-%d'),
                                "score": score
                            })
                    
                    if not top_candidates:
                        status.update(label="Filtering complete.", state="complete")
                        st.warning("No matches found meeting your relevance threshold. Try lowering the minimum score.")
                    else:
                        status.update(label=f"Synthesizing insights via {model_choice.split('/')[-1]}...", state="running")
                        
                        # Generate insights only for the filtered papers
                        for item in top_candidates:
                            paper = item["paper_obj"]
                            summary = generate_insights(paper.summary, active_token, model_choice)
                            
                            st.session_state.filtered_papers.append({
                                "title": paper.title,
                                "url": paper.pdf_url,
                                "published": item["pub_date"],
                                "summary": paper.summary.replace('\n', ' '),
                                "score": item["score"],
                                "ai_insight": summary
                            })
                            
                        status.update(label="Radar scan complete!", state="complete", expanded=False)
                        st.session_state.scan_complete = True

        except Exception as e:
            # Graceful Rate Limit Handling
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg or "http" in error_msg:
                st.warning("ArXiv servers are currently experiencing high traffic (HTTP 429). Caching is active, but new fetches are temporarily delayed. Please wait a few minutes and try again.")
            else:
                st.error(f"Pipeline Error: {e}")

# --- 6. RENDER RESULTS ---
# By rendering outside the 'if run_btn' block using session state, the UI survives the download button click.
if st.session_state.scan_complete and st.session_state.filtered_papers:
    st.success(f"Found {len(st.session_state.filtered_papers)} highly relevant papers.")
    
    for paper_data in st.session_state.filtered_papers:
        with st.expander(f"({paper_data['score']:.2f}) {paper_data['title']}", expanded=True):
            st.write(f"**Published:** {paper_data['published']}")
            st.write(f"**AI Insights:** {paper_data['ai_insight']}")
            st.link_button("Read Paper", paper_data['url'])
            
    # Markdown Export
    md_report = generate_markdown_report(st.session_state.filtered_papers, category, model_choice.split('/')[-1])
    st.download_button(
        label="Download Full Report (Markdown)",
        data=md_report,
        file_name=f"ArXiv_Radar_{category}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
        use_container_width=True
    )