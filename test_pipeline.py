import os
import pytest
from utils import load_embedding_model, get_relevance_score, generate_insights

# Load the model once for all tests to save time
@pytest.fixture(scope="module")
def embedding_model():
    print("\nLoading Embedding Model for Tests...")
    return load_embedding_model()

# Retrieve the HF Token from the environment or secrets
HF_TOKEN = os.environ.get("HF_TOKEN")

# 1. EVALUATING THE RETRIEVER (EMBEDDINGS)
def test_embedding_recall(embedding_model):
    """
    Evaluates if the embedding model correctly scores semantic relevance.
    It tests against a 'Golden Dataset' of known positive and negative matches.
    """
    eval_data = [
        {
            "abstract": "We present a new method for 4-bit quantization using double quantization to reduce memory footprint...",
            "positive_query": "model quantization techniques",
            "negative_query": "black hole event horizons",
            "expected_min_score": 0.50
        },
        {
            "abstract": "This paper explores reinforcement learning from human feedback (RLHF) to align language models...",
            "positive_query": "LLM alignment and RLHF",
            "negative_query": "cellular biology and mitosis",
            # all-MiniLM-L6-v2 is a smaller 384d model, so acronym mapping (RLHF) yields slightly lower baseline scores.
            "expected_min_score": 0.35
        }
    ]

    for data in eval_data:
        # 1. Calculate scores
        pos_score = get_relevance_score(data["abstract"], [data["positive_query"]], embedding_model)
        neg_score = get_relevance_score(data["abstract"], [data["negative_query"]], embedding_model)
        
        # 2. Assertions (If these fail, the test fails)
        assert pos_score >= data["expected_min_score"], \
            f"Positive query scored too low: {pos_score:.2f} (Expected > {data['expected_min_score']})"
            
        assert pos_score > (neg_score + 0.2), \
            f"Model couldn't differentiate! Pos: {pos_score:.2f}, Neg: {neg_score:.2f}"


# 2. EVALUATING THE GENERATOR (LLM)
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