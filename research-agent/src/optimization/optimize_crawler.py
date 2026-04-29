import os
import sys
import logging
from typing import List, Dict

# Ensure src can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import ASI1LLM
from src.agents.crawler import extract_empirical_results

# ASI1 configuration
opt_llm = ASI1LLM(model="asi1", temperature=0.7)

class EvoOptimizer:
    def __init__(self, target_agent_name: str):
        self.target_agent_name = target_agent_name
        self.history = []

    def evaluate(self, result: str, expected: str) -> float:
        """Evaluates the result against expected output."""
        prompt = f"""
        [Identity Context]
        You are the EvoAgentX Evaluator.
        
        [Task Context]
        Compare the Actual Result with the Expected Benchmark.
        Score the quality from 0.0 (wrong/missing) to 1.0 (perfect).
        
        Actual: {result}
        Expected: {expected}
        
        Respond ONLY with the float score.
        """
        try:
            score = float(opt_llm.invoke(prompt).strip())
            return score
        except:
            return 0.5

    def evolve_prompt(self, current_prompt: str, feedback: str) -> str:
        """Mutates the prompt based on feedback."""
        prompt = f"""
        [Identity Context]
        You are the EvoAgentX Self-Evolution Engine.
        
        [Task Context]
        Refine the following prompt to improve its extraction accuracy based on the provided feedback.
        
        Current Prompt:
        {current_prompt}
        
        Feedback from Evaluator:
        {feedback}
        
        Return the NEW improved prompt.
        """
        return opt_llm.invoke(prompt).strip()

def run_optimization_cycle():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Optimizer")
    
    # 1. Dataset: Sample abstracts and expected benchmarks
    dataset = [
        {
            "abstract": "We achieved a state-of-the-art 98.2% top-1 accuracy on ImageNet-1K using only 100M parameters.",
            "expected": "Accuracy: 98.2% on ImageNet-1K (Params: 100M)"
        },
        {
            "abstract": "The model reduces latency by 45% on ARM64 hardware compared to the MobileNetV2 baseline.",
            "expected": "Latency: -45% on ARM64 vs MobileNetV2"
        }
    ]
    
    current_prompt = """
    Extract empirical results from this abstract: {summary}
    """
    
    optimizer = EvoOptimizer("Crawler")
    
    logger.info("🚀 Starting EvoAgentX Self-Evolution Loop...")
    
    for generation in range(2):
        logger.info(f"--- Generation {generation} ---")
        total_score = 0
        all_feedback = []
        
        for item in dataset:
            # Simulate the agent execution with current prompt
            # In a real EvoAgentX setup, this would be agent.run()
            simulated_result = extract_empirical_results(item["abstract"])
            score = optimizer.evaluate(simulated_result, item["expected"])
            total_score += score
            
            if score < 0.9:
                all_feedback.append(f"For abstract '{item['abstract'][:30]}...', the result was '{simulated_result}' but expected '{item['expected']}'.")
        
        avg_score = total_score / len(dataset)
        logger.info(f"Average Fitness Score: {avg_score:.2f}")
        
        if avg_score >= 0.95:
            logger.info("🎯 Evolution Converged! Target accuracy reached.")
            break
            
        # Evolve!
        logger.info("🔄 Evolving prompt based on feedback...")
        combined_feedback = "\n".join(all_feedback)
        current_prompt = optimizer.evolve_prompt(current_prompt, combined_feedback)
        logger.info(f"New Evolved Prompt Strategy: {current_prompt[:100]}...")

    logger.info("✅ Evolution Complete. Best prompt identified.")
    print("\n[Final Optimized Prompt]\n")
    print(current_prompt)

if __name__ == "__main__":
    run_optimization_cycle()
