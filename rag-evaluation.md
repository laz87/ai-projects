# RAG Evaluation Framework

**Version:** 1.0  
**Date:** October 2025  
**Author:** laz87

---

## Overview

This framework provides measurable metrics and methodologies for evaluating Retrieval-Augmented Generation (RAG) systems. It helps answer: "Is my RAG system retrieving the right context and generating accurate answers?"

---

## 1. RAG System Components to Evaluate

```
Query → [Retriever] → Context Chunks → [Generator (LLM)] → Answer
          ↓                                    ↓
    Retrieval Metrics                    Generation Metrics
    (precision, recall)                  (faithfulness, relevance)
```

### 1.1 Evaluation Points

| Component | What to Measure | Metrics |
|-----------|----------------|---------|
| **Retriever** | Quality of retrieved context | Recall@k, MRR, nDCG, Precision@k |
| **Generator** | Quality of final answer | Faithfulness, Answer Relevance, F1 Score |
| **End-to-End** | Overall system quality | Accuracy, Latency, Cost |

---

## 2. Retrieval Metrics

### 2.1 Recall@k

**Definition:** Fraction of relevant documents found in top-k results.

```
Recall@k = |Relevant Docs in Top-k| / |Total Relevant Docs|
```

**Example:**
- Query: "What is the company's refund policy?"
- Ground truth: 3 relevant documents exist
- Retriever returns: 5 documents, 2 of which are relevant
- Recall@5 = 2/3 = 0.67

**Target:** > 0.80 for production systems

### 2.2 Mean Reciprocal Rank (MRR)

**Definition:** Average of reciprocal ranks of first relevant document.

```
MRR = (1/|queries|) × Σ (1/rank_i)
```

**Example:**
- Query 1: First relevant doc at position 1 → 1/1 = 1.0
- Query 2: First relevant doc at position 3 → 1/3 = 0.33
- Query 3: First relevant doc at position 2 → 1/2 = 0.50
- MRR = (1.0 + 0.33 + 0.50) / 3 = 0.61

**Target:** > 0.70 for production systems

### 2.3 Normalized Discounted Cumulative Gain (nDCG)

**Definition:** Measures ranking quality, giving more weight to higher-ranked relevant docs.

```
DCG@k = Σ (rel_i / log2(i+1))  for i = 2 to k
nDCG@k = DCG@k / IDCG@k
```

Where `rel_i` is relevance score of document at position i, and `IDCG@k` is ideal DCG.

**Target:** > 0.75 for production systems

### 2.4 Precision@k

**Definition:** Fraction of top-k results that are relevant.

```
Precision@k = |Relevant Docs in Top-k| / k
```

**Target:** > 0.60 for production systems

---

## 3. Generation Metrics

### 3.1 Faithfulness

**Definition:** How much of the generated answer is supported by the retrieved context.

**Method:**
1. Extract claims from the generated answer
2. Check each claim against the retrieved context
3. Calculate: Faithfulness = Supported Claims / Total Claims

```python
def calculate_faithfulness(answer: str, context: str, llm_client) -> float:
    """Calculate faithfulness using LLM-based claim verification."""
    # Step 1: Extract claims from answer
    claims_prompt = f"""
    Extract individual factual claims from this answer. List each claim on a new line.
    
    Answer: {answer}
    
    Claims:
    """
    claims = llm_client.generate(claims_prompt).strip().split('\n')
    
    # Step 2: Verify each claim against context
    supported = 0
    for claim in claims:
        verify_prompt = f"""
        Based ONLY on the following context, is this claim true, false, or unsupported?
        Answer with just "true", "false", or "unsupported".
        
        Context: {context}
        Claim: {claim}
        
        Verdict:
        """
        verdict = llm_client.generate(verify_prompt).strip().lower()
        if verdict == "true":
            supported += 1
    
    return supported / len(claims) if claims else 0.0
```

**Target:** > 0.85 for production systems

### 3.2 Answer Relevance

**Definition:** How well the answer addresses the original question.

**Method:** Use LLM to score relevance on a 1-5 scale.

```python
def calculate_answer_relevance(query: str, answer: str, llm_client) -> float:
    """Score answer relevance on a 1-5 scale."""
    prompt = f"""
    On a scale of 1-5, how relevant is this answer to the question?
    1 = Completely irrelevant
    5 = Directly and fully addresses the question
    
    Question: {query}
    Answer: {answer}
    
    Score (1-5):
    """
    score = int(llm_client.generate(prompt).strip())
    return score / 5.0
```

**Target:** > 0.80 for production systems

### 3.3 Context Precision

**Definition:** Fraction of generated answer that references the retrieved context (not hallucinated).

**Target:** > 0.90 for production systems

---

## 4. End-to-End Metrics

### 4.1 Overall Accuracy

**Definition:** Percentage of queries where the system produces a correct answer.

```
Accuracy = |Correct Answers| / |Total Queries|
```

**Method:** Use a labeled test set with known correct answers.

**Target:** > 0.80 for production systems

### 4.2 Latency

**Definition:** Time from query submission to answer delivery.

```
Total Latency = Retrieval Time + LLM Generation Time
```

**Target:** < 3 seconds for conversational systems

### 4.3 Cost per Query

**Definition:** API cost for processing one query.

```
Cost = (Input Tokens + Output Tokens) × Price per Token
```

**Target:** Track and optimize over time

---

## 5. Evaluation Dataset

### 5.1 Creating a Test Set

Create a labeled dataset with:

| Field | Description |
|-------|-------------|
| `query` | User question |
| `expected_answer` | Ground truth answer |
| `relevant_docs` | List of relevant document IDs |
| `context` | Full relevant context text |

**Minimum size:** 50 queries for initial evaluation, 200+ for production.

### 5.2 Sample Test Set Structure

```json
[
  {
    "id": 1,
    "query": "What is the company's refund policy?",
    "expected_answer": "The company offers a 30-day money-back guarantee for all products.",
    "relevant_doc_ids": ["doc_001", "doc_045"],
    "category": "policies"
  },
  {
    "id": 2,
    "query": "How do I reset my password?",
    "expected_answer": "Go to Settings > Security > Change Password, or click the 'Forgot Password' link.",
    "relevant_doc_ids": ["doc_012"],
    "category": "account"
  }
]
```

---

## 6. Evaluation Implementation

### 6.1 Complete Evaluation Script

```python
import json
import time
from typing import Dict, List

class RAGEvaluator:
    """Evaluates a RAG system across multiple metrics."""

    def __init__(self, retriever, generator, test_set: List[Dict]):
        self.retriever = retriever
        self.generator = generator
        self.test_set = test_set
        self.results = {}

    def evaluate(self, k: int = 5) -> Dict:
        """Run full evaluation on test set."""
        retrieval_scores = {
            "recall": [], "precision": [], "mrr": [], "ndcg": []
        }
        generation_scores = {"faithfulness": [], "relevance": []}
        latencies = []
        costs = []

        for item in self.test_set:
            query = item["query"]
            relevant_ids = set(item["relevant_doc_ids"])

            # Measure retrieval
            start = time.time()
            retrieved = self.retriever.search(query, k=k)
            retrieval_time = time.time() - start

            # Measure generation
            context = " ".join(doc["text"] for doc in retrieved)
            answer = self.generator.generate(query, context)
            gen_time = time.time() - start - retrieval_time

            # Retrieval metrics
            retrieved_ids = set(doc["id"] for doc in retrieved)
            relevant_found = retrieved_ids & relevant_ids

            recall = len(relevant_found) / len(relevant_ids) if relevant_ids else 0
            precision = len(relevant_found) / k if k > 0 else 0

            # MRR: find rank of first relevant doc
            rr = 0
            for i, doc in enumerate(retrieved):
                if doc["id"] in relevant_ids:
                    rr = 1.0 / (i + 1)
                    break

            retrieval_scores["recall"].append(recall)
            retrieval_scores["precision"].append(precision)
            retrieval_scores["mrr"].append(rr)
            latencies.append(retrieval_time + gen_time)

        # Aggregate
        import statistics
        self.results = {
            "retrieval": {
                "recall@k": statistics.mean(retrieval_scores["recall"]),
                "precision@k": statistics.mean(retrieval_scores["precision"]),
                "mrr": statistics.mean(retrieval_scores["mrr"]),
                "avg_latency_ms": statistics.mean(latencies) * 1000,
            },
            "test_size": len(self.test_set),
            "k": k,
        }

        return self.results

    def print_report(self) -> None:
        """Print evaluation report."""
        r = self.results["retrieval"]
        print("=" * 50)
        print("RAG EVALUATION REPORT")
        print("=" * 50)
        print(f"Test size: {self.results['test_size']} queries")
        print(f"Top-k: {self.results['k']}")
        print("-" * 50)
        print(f"Recall@{self.results['k']}:     {r['recall@k']:.3f}")
        print(f"Precision@{self.results['k']}:  {r['precision@k']:.3f}")
        print(f"MRR:               {r['mrr']:.3f}")
        print(f"Avg Latency:       {r['avg_latency_ms']:.0f}ms")
        print("=" * 50)
```

---

## 7. Retrieval Strategy Comparison

Compare different retrieval approaches:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **BM25** | Keyword-based ranking | Fast, no embeddings needed |
| **Dense (Embeddings)** | Vector similarity search | Semantic understanding |
| **Hybrid** | BM25 + Embeddings combined | Best of both worlds |
| **Reranking** | Cross-encoder reranking | Highest accuracy, slower |

### 7.1 Comparison Template

| Strategy | Recall@5 | Precision@5 | MRR | Latency (ms) |
|----------|----------|-------------|-----|-------------|
| BM25 | | | | |
| Dense (embeddings) | | | | |
| Hybrid (BM25 + Dense) | | | | |
| Hybrid + Reranking | | | | |

---

## 8. Chunking Strategy Evaluation

Different chunking strategies affect retrieval quality:

| Strategy | Chunk Size | Overlap | Pros | Cons |
|----------|-----------|---------|------|------|
| Fixed-size | 500 tokens | 50 tokens | Simple, consistent | May split context |
| Semantic | Variable | Variable | Respects meaning | Complex to implement |
| Paragraph | Variable | 0 tokens | Natural boundaries | Variable quality |
| Sentence | ~50 tokens | 1 sentence | Fine-grained | May lack context |

### 8.1 Chunking Comparison Template

| Strategy | Recall@5 | Precision@5 | Avg Chunk Size |
|----------|----------|-------------|---------------|
| Fixed 500 tokens | | | |
| Fixed 1000 tokens | | | |
| Paragraph-based | | | |
| Semantic | | | |

---

## 9. Continuous Monitoring

### 9.1 Production Metrics Dashboard

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Recall@5 | | > 0.80 | |
| Precision@5 | | > 0.60 | |
| MRR | | > 0.70 | |
| Faithfulness | | > 0.85 | |
| Answer Relevance | | > 0.80 | |
| Avg Latency | | < 3000ms | |
| Error Rate | | < 1% | |

### 9.2 Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Recall@5 drops below | 0.70 | 0.60 |
| Latency exceeds | 4000ms | 6000ms |
| Error rate exceeds | 2% | 5% |
| Faithfulness drops below | 0.80 | 0.70 |

---

## 10. Improvement Checklist

- [ ] Evaluate baseline retrieval strategy
- [ ] Test different chunking strategies
- [ ] Compare BM25 vs. embeddings vs. hybrid
- [ ] Add reranking layer if needed
- [ ] Optimize embedding model choice
- [ ] Tune similarity threshold
- [ ] Evaluate faithfulness of generated answers
- [ ] Set up continuous monitoring
- [ ] Create alerting for metric degradation
- [ ] Schedule regular re-evaluation (monthly)

---

*This framework provides the foundation for RAG evaluation. Adapt metrics and thresholds to your specific use case and quality requirements.*
