import os
import pytest
from datetime import datetime, timedelta, timezone
from utils import load_embedding_model, get_relevance_score, generate_insights, filter_papers

# Load the model once for all tests to save time
@pytest.fixture(scope="module")
def embedding_model():
    print("\nLoading Embedding Model for Tests...")
    return load_embedding_model()

# Retrieve the HF Token from the environment or secrets
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- 1. EVALUATING THE RETRIEVER (EMBEDDINGS) ---
def test_retrieval_mrr_and_hit_rate(embedding_model):
    """
    Evaluates the RAG retrieval layer using Mean Reciprocal Rank (MRR) and Hit Rate@K.
    This simulates a search engine ranking to detect context drift over time.
    """
    # 1. A static 'baseline matrix' of papers to search against
    mock_corpus = [
        {"id": "1", "text": "We present a new method for 4-bit quantization using double quantization to reduce memory footprint..."},
        {"id": "2", "text": "This paper explores reinforcement learning from human feedback (RLHF) to align language models..."},
        {"id": "3", "text": "Our research maps the gaseous envelope of a hot Jupiter during transit to detect water vapor signatures..."},
        {"id": "4", "text": "Analyzing the maximum a posteriori log-likelihood of spider pulsar kinematics in the Milky Way..."}
    ]

    # 2. The Golden Dataset: Queries mapped to their expected #1 paper
    golden_dataset = [
        {"query": "model quantization techniques", "expected_id": "1"},
        {"query": "LLM alignment and RLHF", "expected_id": "2"},
        {"query": "exoplanet atmospheric characterization", "expected_id": "3"}
    ]

    mrr_scores = []
    hit_rate_k = 3

    for data in golden_dataset:
        scored_papers = []
        
        # Score every paper in the corpus against the query
        for paper in mock_corpus:
            score = get_relevance_score(paper["text"], [data["query"]], embedding_model)
            scored_papers.append((paper["id"], score))
            
        # Sort papers by score descending (simulating the search ranking)
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Find the rank position of the expected ground-truth paper
        rank = next((i + 1 for i, p in enumerate(scored_papers) if p[0] == data["expected_id"]), 0)
        
        # Evaluate Hit Rate@3
        assert 0 < rank <= hit_rate_k, f"Hit Rate@{hit_rate_k} Failed! Expected paper {data['expected_id']} dropped to rank {rank}."
        
        # Calculate Reciprocal Rank
        if rank > 0:
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)

    # 3. Evaluate aggregate MRR
    total_mrr = sum(mrr_scores) / len(mrr_scores)
    assert total_mrr >= 0.60, f"Context drift detected! Total MRR dropped to {total_mrr:.2f}"


# --- 2. EVALUATING THE GENERATOR (LLM) ---
# List the models you want to evaluate
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-7B-Instruct"
]

@pytest.mark.skipif(not HF_TOKEN, reason="No HF_TOKEN found in environment. Skipping LLM test.")
@pytest.mark.parametrize("model_id", MODELS_TO_TEST)
def test_llm_constraints(model_id):
    """
    Evaluates if different LLMs respect the strict system prompt formatting.
    """
    print(f"\nTesting LLM: {model_id}")
    test_abstract = "We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance."
    
    # Generate the insight using the parameterized model
    insight = generate_insights(test_abstract, HF_TOKEN, model_id)
    
    # 1. Test Length Constraint
    sentences = [s for s in insight.split('.') if s.strip()]
    assert len(sentences) <= 2, f"[{model_id}] Failed: Too wordy. Got {len(sentences)} sentences.\nOutput: {insight}"
    
    # 2. Test Structural Constraints
    assert "TAGS:" in insight, f"[{model_id}] Failed: Missed 'TAGS:' formatting.\nOutput: {insight}"
    assert "HOOK:" in insight, f"[{model_id}] Failed: Missed 'HOOK:' formatting.\nOutput: {insight}"
    assert "|" in insight, f"[{model_id}] Failed: Missed '|' separator.\nOutput: {insight}"
    
    # 3. Test for Conversational Filler
    lower_insight = insight.lower()
    assert not lower_insight.startswith("here is"), f"[{model_id}] Failed: Included conversational filler."
    assert not lower_insight.startswith("sure"), f"[{model_id}] Failed: Included conversational filler."

# --- 3. EVALUATING THE FILTERING LOGIC ---
def test_filter_papers(embedding_model):
    """
    Validates that papers are correctly filtered by both date and relevance threshold.
    """
    class MockPaper:
        def __init__(self, updated, summary):
            self.updated = updated
            self.summary = summary

    now = datetime.now(timezone.utc)
    
    mock_papers = [
        # Paper 1: Highly relevant, published today
        MockPaper(now, "A detailed study on parameter-efficient fine-tuning and LoRA techniques."),
        # Paper 2: Highly relevant, but published 10 days ago
        MockPaper(now - timedelta(days=10), "An old paper about LoRA and model quantization."),
        # Paper 3: Irrelevant, published today
        MockPaper(now, "A sociological study on ancient Roman aqueducts.")
    ]

    cutoff_date = now - timedelta(days=7)
    user_interests = ["LoRA", "quantization"]
    score_threshold = 0.40

    filtered_results = filter_papers(
        mock_papers, cutoff_date, user_interests, embedding_model, score_threshold
    )

    # Assertions
    assert len(filtered_results) == 1, f"Expected 1 paper, got {len(filtered_results)}"
    assert filtered_results[0]["score"] >= score_threshold, "Failed threshold check."