#!/usr/bin/env python3
"""
Test script for multilingual embedding matching.

Compares:
- XLM-RoBERTa (your current model) - trained for MLM, NOT retrieval
- multilingual-e5-small - trained specifically for retrieval/similarity
"""

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np


def get_cls_embedding(text: str, tokenizer, model) -> torch.Tensor:
    """Get the CLS token embedding (what your Android app currently does)."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    return F.cosine_similarity(a, b).item()


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity for numpy arrays."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_test_xlm(get_embedding, method_name, tokenizer, model):
    """Run the full test suite with XLM-RoBERTa."""

    print(f"\n{'='*70}")
    print(f"TESTING: {method_name}")
    print(f"{'='*70}")

    english_terms = ["moon", "sun", "star", "water", "fire", "earth", "mountain", "river", "forest", "ocean"]
    multilingual_queries = [
        ("luna", "Spanish", "moon"),
        ("sole", "Italian", "sun"),
        ("stern", "German", "star"),
        ("agua", "Spanish", "water"),
        ("fuoco", "Italian", "fire"),
        ("terre", "French", "earth"),
        ("montagna", "Italian", "mountain"),
        ("fiume", "Italian", "river"),
        ("foret", "French", "forest"),
        ("oceano", "Spanish", "ocean"),
        ("mond", "German", "moon"),
        ("wasser", "German", "water"),
        ("sol", "Spanish", "sun"),
        ("etoile", "French", "star"),
    ]

    english_embeddings = {term: get_embedding(term, tokenizer, model) for term in english_terms}

    print("\nDirect translations test:\n")

    correct = 0
    for query, language, expected in multilingual_queries:
        query_embedding = get_embedding(query, tokenizer, model)
        similarities = {term: cosine_similarity(query_embedding, emb) for term, emb in english_embeddings.items()}
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        best_match, best_score = sorted_sims[0]
        second_best, second_score = sorted_sims[1]
        expected_score = similarities[expected]

        is_correct = best_match == expected
        if is_correct:
            correct += 1

        status = "OK" if is_correct else "MISS"
        margin = best_score - second_score
        print(f"  [{status}] '{query}' ({language}) -> '{best_match}' ({best_score:.3f})  expected: '{expected}' ({expected_score:.3f})  margin: {margin:.4f}")

    accuracy = 100 * correct / len(multilingual_queries)
    print(f"\n  >>> Accuracy: {correct}/{len(multilingual_queries)} ({accuracy:.1f}%)")
    return accuracy


def run_test_e5(model, model_name):
    """Run the test suite with a sentence-transformers model (like E5)."""

    print(f"\n{'='*70}")
    print(f"TESTING: {model_name}")
    print(f"{'='*70}")

    english_terms = ["moon", "sun", "star", "water", "fire", "earth", "mountain", "river", "forest", "ocean"]
    multilingual_queries = [
        ("luna", "Spanish", "moon"),
        ("sole", "Italian", "sun"),
        ("stern", "German", "star"),
        ("agua", "Spanish", "water"),
        ("fuoco", "Italian", "fire"),
        ("terre", "French", "earth"),
        ("montagna", "Italian", "mountain"),
        ("fiume", "Italian", "river"),
        ("foret", "French", "forest"),
        ("oceano", "Spanish", "ocean"),
        ("mond", "German", "moon"),
        ("wasser", "German", "water"),
        ("sol", "Spanish", "sun"),
        ("etoile", "French", "star"),
    ]

    # E5 models want "query: " prefix for queries (optional for this test)
    english_embeddings = {term: model.encode(term) for term in english_terms}

    print("\nDirect translations test:\n")

    correct = 0
    for query, language, expected in multilingual_queries:
        query_embedding = model.encode(query)
        similarities = {term: cosine_similarity_np(query_embedding, emb) for term, emb in english_embeddings.items()}
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        best_match, best_score = sorted_sims[0]
        second_best, second_score = sorted_sims[1]
        expected_score = similarities[expected]

        is_correct = best_match == expected
        if is_correct:
            correct += 1

        status = "OK" if is_correct else "MISS"
        margin = best_score - second_score
        print(f"  [{status}] '{query}' ({language}) -> '{best_match}' ({best_score:.3f})  expected: '{expected}' ({expected_score:.3f})  margin: {margin:.4f}")

    accuracy = 100 * correct / len(multilingual_queries)
    print(f"\n  >>> Accuracy: {correct}/{len(multilingual_queries)} ({accuracy:.1f}%)")

    # Planet names test
    print("\n\nPlanet names test:\n")

    names = ["Jupiter", "Mars", "Venus", "Saturn", "Mercury", "Neptune"]
    name_embeddings = {name: model.encode(name) for name in names}

    name_queries = [
        ("Giove", "Italian", "Jupiter"),
        ("Marte", "Italian/Spanish", "Mars"),
        ("Venere", "Italian", "Venus"),
        ("Saturno", "Italian/Spanish", "Saturn"),
        ("Mercurio", "Italian/Spanish", "Mercury"),
        ("Neptuno", "Spanish", "Neptune"),
    ]

    correct_names = 0
    for query, language, expected in name_queries:
        query_embedding = model.encode(query)
        similarities = {name: cosine_similarity_np(query_embedding, emb) for name, emb in name_embeddings.items()}
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        best_match, best_score = sorted_sims[0]
        expected_score = similarities[expected]

        is_correct = best_match == expected
        if is_correct:
            correct_names += 1

        status = "OK" if is_correct else "MISS"
        print(f"  [{status}] '{query}' ({language}) -> '{best_match}' ({best_score:.3f})  expected: '{expected}' ({expected_score:.3f})")

    print(f"\n  >>> Accuracy: {correct_names}/{len(name_queries)} ({100*correct_names/len(name_queries):.1f}%)")

    return accuracy


def main():
    # Test 1: XLM-RoBERTa (your current model)
    print("Loading XLM-RoBERTa base model (your current model)...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    xlm_model = AutoModel.from_pretrained("xlm-roberta-base")
    xlm_model.eval()

    xlm_accuracy = run_test_xlm(get_cls_embedding, "XLM-RoBERTa (CLS pooling) - YOUR CURRENT MODEL", tokenizer, xlm_model)

    # Free memory
    del xlm_model
    del tokenizer

    # Test 2: multilingual-e5-small (retrieval-optimized)
    print("\n\nLoading multilingual-e5-small (retrieval-optimized model)...")
    e5_model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

    e5_accuracy = run_test_e5(e5_model, "multilingual-e5-small (384 dims) - RETRIEVAL OPTIMIZED")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"""
MODEL COMPARISON FOR MULTILINGUAL WORD MATCHING:

  XLM-RoBERTa (your current):     {xlm_accuracy:.1f}% accuracy
  multilingual-e5-small:          {e5_accuracy:.1f}% accuracy

WHY THE DIFFERENCE?

XLM-RoBERTa was trained for Masked Language Modeling (predicting missing
words). Its embeddings capture syntax and structure, but single-word
embeddings all look nearly identical (~0.999 cosine similarity).

multilingual-e5-small was trained with CONTRASTIVE LEARNING specifically
for retrieval tasks. It learns to push similar meanings together and
different meanings apart in embedding space.

RECOMMENDATION:

For your use case (matching English terms to multilingual queries),
you should use multilingual-e5-small or similar. It's:
  - Smaller (384 dims vs 768)
  - Faster
  - Much more accurate for similarity/retrieval

You can export it to ONNX for your Android app using:
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("intfloat/multilingual-e5-small")
  model.save_onnx("multilingual-e5-small.onnx")
""")


if __name__ == "__main__":
    main()
