#!/usr/bin/env python3
"""Prompt builders for STOP annotation stages."""

from __future__ import annotations

from pipeline.common import parse_jsonl_objects


def build_judge_prompt(target: object, response: object, tail_n: int = 100) -> str:
    response_tail = str(response)[-tail_n:]
    return f"""You are a strict answer verifier. Determine whether the model answer is correct.
If it is correct and in boxed format (e.g., boxed{{28}}), return True, else return False.

Correct Answer:
{target}

Model Answer:
{response_tail}

Return True or False.
"""


def build_taxonomy_prompt(row) -> str:
    paragraphs = row["paragraphs"]
    return f"""You will be given a list of reasoning segment nodes, each labeled (e.g., N1, N2), representing steps in a reasoning trace.

Your task has to strictly follow three steps:
First, analyze each node and provide a brief analysis of each node's function in the reasoning process.
Second, assign reasoning strategies using the taxonomy below. Some longer nodes may involve multiple strategies. If no secondary strategy applies, leave it as "None".
Finally, strictly format your output as jsonl as below:

```jsonl
{{"id": "N1", "taxonomy_primary_type": "Verification", "taxonomy_secondary_type": "None"}}
{{"id": "N2", "taxonomy_primary_type": "Exploration", "taxonomy_secondary_type": "Verification"}}
```

Reasoning Taxonomy:

Backtracking:
The node revisits and modifies a previous step or assumption to correct an error, resolve a conflict, or incorporate a new insight that alters the reasoning path.

Verification:
The node tests or confirms the correctness of a specific claim, assumption, or result without modifying the reasoning path.

Exploration:
The node proposes new hypotheses, possibilities, or approaches to the problem in an open-ended manner without committing to a definitive solution.

Clarification:
The node rephrases, restates, or defines terms, assumptions, or problem constraints to reduce ambiguity and enhance understanding.

Conclusion:
The node synthesizes prior reasoning to assert a final or intermediate solution, judgment, or result.

Node:
{paragraphs}
"""


def conclusion_ids_from_taxonomy(taxonomy: object) -> list[object]:
    conclusion_ids = []
    for item in parse_jsonl_objects(taxonomy):
        primary = str(item.get("taxonomy_primary_type", "")).lower()
        if primary == "conclusion":
            conclusion_ids.append(item.get("id"))
    return conclusion_ids


def build_conclusion_prompt(row) -> str:
    conclusion_ids = conclusion_ids_from_taxonomy(row["taxonomy"])
    question = row["question"]
    target = row["target"]
    paragraphs = row["paragraphs"]

    return f"""Task:
You are given a reasoning trace divided into nodes (e.g., N1, N2). Each node represents a reasoning step.
You are also given this list of conclusion nodes identified by serial number:
{conclusion_ids}

For each conclusion node, do the following step by step:

1. Determine whether the conclusion node matches the Correct Answer.
   - If it exactly matches the Correct Answer, mark it as 1.
   - If it does not match, mark it as 0.

2. Determine the type of the conclusion node:
   - Intermediate Conclusion: supports later reasoning but does not directly answer the question.
   - Answering Conclusion: intended to answer the question, regardless of whether it is correct.

3. Include the question and the Correct Answer:
   - Question: {question}
   - Correct Answer: {target}

4. Export the result in JSONL format:

```jsonl
{{"conclusion_node": "N5", "is_correct": 0, "type": "Intermediate Conclusion"}}
{{"conclusion_node": "N10", "is_correct": 1, "type": "Answering Conclusion"}}
```

Complete Nodes:
{paragraphs}
"""
