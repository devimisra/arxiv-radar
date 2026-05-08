from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import numpy as np
from datetime import datetime, timezone

def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_relevance_score(text, queries, embedding_model): 
    text_vec = embedding_model.embed_query(text.lower())
    text_norm = np.linalg.norm(text_vec)
    scores = []
    for q in queries:
        q_vec = embedding_model.embed_query(q.lower())
        q_norm = np.linalg.norm(q_vec)
        score = np.dot(q_vec, text_vec) / (q_norm * text_norm) if q_norm > 0 else 0
        scores.append(score)
    return max(scores) if scores else 0.0

def generate_insights(abstract, hf_token, model_id):
    client = InferenceClient(api_key=hf_token)
    prompt = f"Abstract: {abstract}\nTask: Provide 3 relevant technical tags and 1 punchy hook (1 sentence max). Format: TAGS: x, y | HOOK: z"
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Insights currently unavailable: {str(e)}"
    
def generate_markdown_report(papers, category, model_name):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    md_content = f"# ArXiv Radar Report\n"
    md_content += f"**Category:** `{category}` | **Date:** {today} | **AI Model:** `{model_name}`\n\n"
    md_content += "---\n\n"
    
    for paper in papers:
        md_content += f"## [{paper['title']}]({paper['url']})\n"
        md_content += f"**Published:** {paper['published']}\n\n"
        md_content += f"**Semantic Relevance:** {paper['score']:.2f}\n\n"
        
        if paper.get('ai_insight'):
            md_content += f"**AI Insight:** {paper['ai_insight']}\n\n"
            
        md_content += f"**Abstract:**\n> {paper['summary']}\n\n"
        md_content += "---\n\n"
        
    return md_content