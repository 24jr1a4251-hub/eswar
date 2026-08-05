# ==============================================================================
# PROMPTIQ: Iterative Prompt Refinement Engine
# Description: Refines input prompts to effectively guide generative AI models.
# ==============================================================================

import sys
import subprocess
import os

# ------------------------------------------------------------------------------
# 1. AUTO-DEPENDENCY INSTALLER
# ------------------------------------------------------------------------------
required_packages = ["nltk", "matplotlib", "rouge-score", "openai"]
bert_score_fn = None
try:
    import nltk
    import matplotlib.pyplot as plt
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
except ImportError:
    print("Missing packages detected. Installing dependencies now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)
    print("Installation complete! Re-importing libraries...")
    import nltk
    import matplotlib.pyplot as plt
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
except (ImportError, OSError) as exc:
    print("BERTScore is unavailable in this environment. Using fallback semantic scoring.")
    bert_score_fn = None

# Download necessary NLTK data (runs quietly)
nltk.download("punkt", quiet=True)


# ------------------------------------------------------------------------------
# 2. CONFIGURATION & PARAMETERS
# ------------------------------------------------------------------------------
WEIGHTS = {"bleu": 0.20, "rouge": 0.30, "bert": 0.50}
MAX_ITERATIONS = 5
TARGET_SCORE_THRESHOLD = 0.85
CONVERGENCE_DELTA = 0.01
DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 1000


# ------------------------------------------------------------------------------
# 3. EVALUATION METRICS (evaluator.py)
# ------------------------------------------------------------------------------
def compute_bleu(hypothesis: str, reference: str) -> float:
    ref_tokens = [reference.split()]
    hyp_tokens = hypothesis.split()
    smoother = SmoothingFunction().method4
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoother)

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure

def compute_bert_score(hypothesis: str, reference: str) -> float:
    if bert_score_fn is None:
        # Fallback approximation when BERTScore isn't available
        common_tokens = set(hypothesis.lower().split()) & set(reference.lower().split())
        overlap_ratio = len(common_tokens) / max(len(set(reference.lower().split())), 1)
        return min(1.0, overlap_ratio + 0.1)

    _, _, F1 = bert_score_fn(
        [hypothesis], [reference], lang="en",
        model_type="microsoft/deberta-xlarge-mnli", verbose=False
    )
    return F1.mean().item()

def evaluate_response(hypothesis: str, reference: str) -> dict:
    bleu = compute_bleu(hypothesis, reference)
    rouge = compute_rouge_l(hypothesis, reference)
    bert = compute_bert_score(hypothesis, reference)
    total = (WEIGHTS["bleu"] * bleu) + (WEIGHTS["rouge"] * rouge) + (WEIGHTS["bert"] * bert)

    return {
        "bleu": round(bleu, 4),
        "rouge_l": round(rouge, 4),
        "bert_score": round(bert, 4),
        "total_score": round(total, 4),
    }


# ------------------------------------------------------------------------------
# 4. PROMPT BUILDER (prompt_builder.py)
# ------------------------------------------------------------------------------
class PromptBuilder:
    def __init__(self, task_type: str, domain: str = "general"):
        self.task_type = task_type
        self.domain = domain

    def build_initial_prompt(self, base_task: str, constraints: list = None) -> str:
        prompt_parts = [
            f"Role: You are an expert AI assistant specializing in {self.domain}.",
            f"Task ({self.task_type}): {base_task}"
        ]
        if constraints:
            formatted_constraints = "\n".join([f"- {c}" for c in constraints])
            prompt_parts.append(f"Constraints:\n{formatted_constraints}")
        prompt_parts.append("Instructions: Provide a clear, accurate, and concise response.")
        return "\n\n".join(prompt_parts)

    def refine_prompt(self, current_prompt: str, weakness_feedback: str) -> str:
        refinement_block = (
            f"\n\n[Refinement Feedback]\n"
            f"Previous output fell short in these areas:\n{weakness_feedback}\n"
            f"Adjust the response tone, detail level, and structural correctness."
        )
        return current_prompt + refinement_block


# ------------------------------------------------------------------------------
# 5. MODEL CALLER (model_caller.py)
# ------------------------------------------------------------------------------
class ModelCaller:
    def __init__(self, provider: str = "mock"):
        self.provider = provider

    def generate_response(self, prompt: str) -> str:
        if self.provider == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                return response.choices[0].message.content
            except Exception:
                return "API Error. Simulating response based on prompt instructions."
        else:
            # Deterministic simulation for local testing without API keys
            return "Iterative prompt engineering improves AI outputs by systematically refining instructions based on scoring metrics, resulting in highly accurate responses."


# ------------------------------------------------------------------------------
# 6. WEAKNESS ANALYZER (weakness_analyzer.py)
# ------------------------------------------------------------------------------
class WeaknessAnalyzer:
    @staticmethod
    def analyze(scores: dict) -> str:
        issues = []
        if scores["bleu"] < 0.35:
            issues.append("- Low n-gram exact overlap: Ensure exact phrase structures are present.")
        if scores["rouge_l"] < 0.45:
            issues.append("- Low sequence structure: Reorder information to match the target sequence.")
        if scores["bert_score"] < 0.65:
            issues.append("- Low semantic fidelity: The underlying meaning diverges from expected ground truth.")
        if not issues:
            issues.append("- High quality response: Fine-tune formatting and tone.")
        return "\n".join(issues)


# ------------------------------------------------------------------------------
# 7. OPTIMIZER PIPELINE (optimizer.py)
# ------------------------------------------------------------------------------
class PromptOptimizer:
    def __init__(self, task_type: str, domain: str = "general", provider: str = "mock"):
        self.builder = PromptBuilder(task_type, domain)
        self.caller = ModelCaller(provider)
        self.history = []

    def run(self, base_task: str, reference: str, constraints: list = None) -> dict:
        current_prompt = self.builder.build_initial_prompt(base_task, constraints)
        previous_score = 0.0

        for i in range(1, MAX_ITERATIONS + 1):
            response = self.caller.generate_response(current_prompt)
            scores = evaluate_response(response, reference)
            total_score = scores["total_score"]

            self.history.append({
                "iteration": i, "prompt": current_prompt, 
                "response": response, "scores": scores
            })

            print(f"Iteration {i} | Composite Score: {total_score:.4f} | BLEU: {scores['bleu']:.4f}")

            if total_score >= TARGET_SCORE_THRESHOLD:
                print(f"Target score reached ({total_score:.4f} >= {TARGET_SCORE_THRESHOLD}).")
                break
            if i > 1 and abs(total_score - previous_score) < CONVERGENCE_DELTA:
                print(f"Convergence met at iteration {i}.")
                break

            previous_score = total_score
            feedback = WeaknessAnalyzer.analyze(scores)
            current_prompt = self.builder.refine_prompt(current_prompt, feedback)

        return {"final_scores": scores, "history": self.history}


# ------------------------------------------------------------------------------
# 8. VISUALIZER (visualizer.py)
# ------------------------------------------------------------------------------
def plot_metrics(history: list):
    iterations = [h["iteration"] for h in history]
    bleu = [h["scores"]["bleu"] for h in history]
    rouge = [h["scores"]["rouge_l"] for h in history]
    bert = [h["scores"]["bert_score"] for h in history]
    total = [h["scores"]["total_score"] for h in history]

    plt.figure(figsize=(9, 5))
    plt.plot(iterations, bleu, marker='o', label='BLEU')
    plt.plot(iterations, rouge, marker='s', label='ROUGE-L')
    plt.plot(iterations, bert, marker='^', label='BERTScore')
    plt.plot(iterations, total, color='black', linewidth=2.5, linestyle='--', label='Total Weighted')

    plt.title("PromptIQ Metric Improvement Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Score [0 - 1]")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# 9. MAIN EXECUTION BLOCK
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing PromptIQ Optimization Pipeline...\n")
    
    # Define project inputs
    task = "Summarize the benefits of iterative prompt engineering."
    reference_truth = "Iterative prompt engineering improves AI outputs by systematically refining instructions based on scoring metrics like BLEU and BERTScore, resulting in highly accurate, context-aware, and aligned responses."
    constraints = ["Keep it under 50 words", "Mention specific metrics"]
    
    # Run optimization (Using "mock" provider for local testing without API key)
    optimizer = PromptOptimizer(task_type="summarization", domain="AI Engineering", provider="mock")
    results = optimizer.run(base_task=task, reference=reference_truth, constraints=constraints)
    
    print("\n--- Optimization Complete ---")
    print(f"Final Composite Score: {results['final_scores']['total_score']}")
    
    # Plot results
    plot_metrics(results["history"])