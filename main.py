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

# --- 3. MAIN INTERFACE ---
st.title("ArXiv Research Radar")
st.markdown("""
Welcome to **ArXiv Radar**, your automated research assistant. This tool scans ArXiv for the latest papers, filters them based on the semantic meaning of your interests, and uses AI to extract technical insights.

**System Reliability:**
To ensure high-signal outputs, our prompt architecture is continuously evaluated via an automated CI/CD pipeline. Using a baseline "golden dataset" and our default open model (Qwen 2.5), the system is verified to achieve **100% adherence** to strict formatting constraints (maximum one-sentence hooks) and zero hallucination of conversational filler. Advanced models (Llama 3.1, Gemma 2) inherit this strict architecture and are seamlessly supported via the BYOK interface.
""")

with st.expander("How to use this tool"):
    st.markdown("""
    1. **Bring Your Own Key (BYOK):** Enter your Hugging Face Token in the sidebar. Your token is never stored or saved by this application.
        * **No Token?** Leave it blank to use the shared demo token (open models only).
        * **Want Gated Models (e.g., Llama 3.1, Gemma 2)?** 
            1. Visit the model's page on Hugging Face and click **"Acknowledge License"**.
            2. Go to your HF Profile Settings > Access Tokens and generate a **"Read"** token.
            3. Paste that token into the sidebar.

    2. **Set Your Filters & Topics:** Define your ArXiv category and input your specific research interests.
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
        st.warning("Token required for AI insights.")
    else:
        st.success("AI Engine Authenticated")
    

    model_options = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2.5-7B-Instruct", 
        "Custom (Enter Model ID)"
    ]
    
    model_selection = st.selectbox("AI Summary Model", model_options, index=0, key="model_select_box")
    
    if model_selection == "Custom (Enter Model ID)":
        model_choice = st.text_input("Enter Model ID", placeholder="e.g. meta-llama/Llama-3.1-70B-Instruct", key="custom_model_input")
    else:
        model_choice = model_selection

    category = st.text_input("ArXiv Category", value="cs.LG", key="arxiv_cat_input")
    interest_text = st.text_area("Research Interests (Use specific technical phrases)", value="Retrieval-Augmented Generation, vector databases, chunking strategies", key="interests_input")
    user_interests = [i.strip() for i in interest_text.split(",") if i.strip()]
    days_val = st.slider("Days Back", 1, 14, 7, key="days_slider")
    score_threshold = st.slider("Min Relevance Score", 0.0, 1.0, 0.50, 0.05, key="score_slider")
    
    run_btn = st.button("Start Radar Scan", type="primary", key="run_scan_btn", use_container_width=True)

# --- 5. MAIN EXECUTION ---
if run_btn:
    if not active_token:
        st.error("Please provide a Hugging Face Token.")
    elif model_selection == "Custom (Enter Model ID)" and not model_choice:
        st.error("Please enter a Model ID.")
    elif not user_interests:
        st.error("Please enter at least one research interest.")
    else:
        st.info(f"Scanning {category}...")
        client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
        search = arxiv.Search(query=f"cat:{category}", max_results=100, sort_by=arxiv.SortCriterion.LastUpdatedDate)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_val)
        
        try:
            results = list(client.results(search))
            if not results:
                st.warning("No papers found in the specified timeframe.")
            else:
                progress = st.progress(0)
                filtered_papers = []
                
                for i, paper in enumerate(results):
                    progress.progress((i + 1) / len(results))
                    pub_date = paper.updated.replace(tzinfo=timezone.utc)
                    if pub_date < cutoff: continue
                    
                    score = get_relevance_score(paper.summary, user_interests, embedding_model)
                    if score >= score_threshold:
                        paper_data = {
                            "title": paper.title,
                            "url": paper.pdf_url,
                            "published": pub_date.strftime('%Y-%m-%d'),
                            "summary": paper.summary.replace('\n', ' '),
                            "score": score,
                            "ai_insight": ""
                        }
                        
                        with st.expander(f"({score:.2f}) {paper.title}", expanded=True):
                            st.write(f"**Published:** {paper_data['published']}")
                            with st.spinner("Analyzing abstract..."):
                                summary = generate_insights(paper.summary, active_token, model_choice)
                                paper_data["ai_insight"] = summary
                            st.write(f"**AI Insights:** {summary}")
                            st.link_button("Read Paper", paper.pdf_url)
                            
                        filtered_papers.append(paper_data)
                        
                progress.empty()
                
                if not filtered_papers:
                    st.warning("No matches found meeting your relevance threshold. Try lowering the minimum score or using different keywords.")
                else:
                    st.success(f"Scan complete. Found {len(filtered_papers)} highly relevant papers.")
                    
                    md_report = generate_markdown_report(filtered_papers, category, model_choice.split('/')[-1])
                    st.download_button(
                        label="Download Full Report (Markdown)",
                        data=md_report,
                        file_name=f"ArXiv_Radar_{category}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"ArXiv API Error: {e}")