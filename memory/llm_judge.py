import argparse
import json
from collections import defaultdict

import numpy as np
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

ACCURACY_PROMPT = """
Score the answer on a scale from 0.0 to 1.0 based on semantic correctness.

Scoring Scale:
- 1.0: Perfect match - contains all key information from gold answer, semantically equivalent
- 0.8: Mostly correct - captures main point but may have minor differences in wording or detail
- 0.6: Partially correct - has some correct information but incomplete or missing key details
- 0.4: Somewhat related - touches on the topic but misses significant information
- 0.2: Barely related - answer is mostly incorrect but has some connection to the topic
- 0.0: Completely wrong - answer is unrelated or contradicts gold answer

The point of the question is to ask about something one user should know about the other user based on their prior conversations.

For time-related questions:
- Be generous with date formats (e.g., "May 7th" vs "7 May" should both score highly)
- Accept relative time references if they refer to the same period
- Penalize if the time period is significantly different

For factual questions:
- Focus on semantic equivalence, not exact wording
- Partial credit for partial answers
- Consider whether key entities and relationships are preserved

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON with:
- "score": A float between 0.0 and 1.0
- "reasoning": One sentence explaining the score
"""

class LLMJudge:
    """
    Continuous LLM judge for semantic answer evaluation.

    Scores answers on a 0.0-1.0 scale with partial credit.
    """
    def __init__(self, llm_controller=None):
        """
        Args:
            llm_controller: LLM controller used by the memory/query pipeline.
                Passing it keeps evaluation on the same backend (including local
                OpenAI-compatible vLLM) instead of making a separate cloud call.
        """
        self.llm_controller = llm_controller

    def evaluate_answer(self, question: str, gold_answer: str, predicted_answer: str,
                       question_category: int = None) -> dict:
        """
        Evaluate answer with continuous scoring and category awareness.

        Args:
            question: The question asked
            gold_answer: Ground truth answer
            predicted_answer: Generated answer
            question_category: Question category (optional, for category-aware scoring)

        Returns:
            dict with keys: score (float 0.0-1.0), reasoning (str)
        """
        if question_category == 5:
            pred_is_unans = self._is_unanswerable(predicted_answer)

            if pred_is_unans:
                return {
                    'score': 1.0,
                    'reasoning': "Category 5: Correctly identified as unanswerable (adversarial question)"
                }
            else:
                return {
                    'score': 0.0,
                    'reasoning': "Category 5: Hallucinated answer for adversarial question (should be unanswerable)"
                }

        score = evaluate_llm_judge(
            question, gold_answer, predicted_answer,
            llm_controller=self.llm_controller
        )
        return {
            'score': score,
            'reasoning': f"Continuous LLM judge score: {score:.2f}"
        }

    def _is_unanswerable(self, text: str) -> bool:
        """
        Check if text represents an unanswerable response.

        Args:
            text: Text to check

        Returns:
            True if text indicates unanswerable/no information
        """
        if not text:
            return True

        text_lower = text.strip().lower()

        if text_lower in {"", "n/a", "na", "none", "null", "unanswerable"}:
            return True

        patterns = [
            "not mentioned",
            "not in the conversation",
            "cannot answer",
            "can't answer",
            "insufficient",
            "unknown",
            "no information",
            "not provided",
            "information not found",
            "not found",
            "not available",
            "no data"
        ]

        return any(pattern in text_lower for pattern in patterns)

def _parse_json_object(text):
    """Parse a JSON object, tolerating markdown fences around local-model output."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def evaluate_llm_judge(question, gold_answer, generated_answer, llm_controller=None):
    """
    Evaluate the generated answer against the gold answer using an LLM judge.

    Returns:
        float: Score between 0.0 and 1.0 representing semantic correctness
    """
    prompt = ACCURACY_PROMPT.format(
        question=question, gold_answer=gold_answer,
        generated_answer=generated_answer
    )
    if llm_controller is not None:
        content = llm_controller.llm.get_completion(
            prompt,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
    else:
        # Standalone CLI compatibility. The experiment path always passes its
        # configured controller, so local runs never enter this cloud branch.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM judge requires an llm_controller or OPENAI_API_KEY"
            )
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert grader that scores answers from 0.0 to 1.0."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content

    result = _parse_json_object(content)
    score = float(result.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return score

def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )

    args = parser.parse_args()

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(f"  Category {cat}: {np.mean(results):.4f} " f"({sum(results)}/{len(results)})")
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))

if __name__ == "__main__":
    main()
