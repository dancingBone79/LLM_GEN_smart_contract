# Evaluation Smart Contracts

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import coverage

def compute_bleu_score(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)

def compute_rouge_scores(reference, candidate):
    rouge = Rouge()
    return rouge.get_scores(candidate, reference, avg=True)

def compute_code_coverage(script_path):
    cov = coverage.Coverage()
    cov.start()
    exec(open(script_path).read())
    cov.stop()
    cov.save()
    return cov.report()

def evaluate_generated_code(reference_code, generated_code, script_path):
    # BLEU Score
    bleu_score = compute_bleu_score(reference_code, generated_code)
    
    # ROUGE Scores
    rouge_scores = compute_rouge_scores(reference_code, generated_code)
    
    # Code Coverage
    coverage_score = compute_code_coverage(script_path)
    
    return {
        "BLEU Score": bleu_score,
        "ROUGE Scores": rouge_scores,
        "Code Coverage": coverage_score
    }

# Example usage
reference_code = "function transfer(address to, uint256 value) public"
generated_code = "function transfer(address to, uint256 value) public"
script_path = "generated_code.py"  # Replace with the path to the script

results = evaluate_generated_code(reference_code, generated_code, script_path)
print("Evaluation Results:")
print(results)
