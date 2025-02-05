import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from datasets import load_dataset
from token_scoring import calculate_mutual_information, retain_top_k_tokens
from openai import OpenAI

def generate_cot_prompt(question: str) -> str:
        """
        Generates a CoT prompt for the given question
        """
        return f"Solve this problem step by step. Problem: {question}"

def extract_answer_from_cot(cot_response: str) -> str:
    """
    Extracts the final answer from a CoT response
    """
    # Look for patterns like "The answer is", "Final answer:", etc.
    patterns = ["the answer is", "final answer:", "result is", "solution is"]
    
    # Split into lines and reverse to find the last occurrence
    lines = cot_response.lower().split('\n')
    for line in reversed(lines):
        for pattern in patterns:
            if pattern in line:
                # Extract everything after the pattern
                return line.split(pattern)[-1].strip()
    
    # If no pattern found, return the last line as fallback
    return lines[-1].strip()

def generate_responses(dataset, model):
    """
    Generates CoT and answer-only responses for a dataset
    """
    results = []
    
    for example in dataset:
        # Generate CoT response
        cot_prompt = generate_cot_prompt(example['question'])
        cot_response = model.generate(cot_prompt)
        
        # Extract answer from CoT
        cot_answer = extract_answer_from_cot(cot_response)
        
        # Generate answer-only response
        answer_response = model.generate(example['question'])
        
        results.append({
            'question': example['question'],
            'cot_response': cot_response,
            'cot_answer': cot_answer,
            'answer_response': answer_response
        })
    
    return results

class cot_generation:
    def __init__(self):
        pass
    
    def generate_responses_with_importance(dataset, model):
        """
        Generates CoT responses with token importance analysis
        """
        # First pass: collect all CoT responses and answers
        cot_responses = []
        answers = []
        for example in dataset:
            cot_prompt = generate_cot_prompt(example['question'])
            cot_response = model.generate(cot_prompt)
            cot_answer = extract_answer_from_cot(cot_response)
            cot_responses.append(cot_response)
            answers.append(cot_answer)
        
        # Calculate mutual information scores
        mi_scores = calculate_mutual_information(cot_responses, answers)
        
        # Second pass: generate responses with token importance
        results = []
        for example, cot_response, answer in zip(dataset, cot_responses, answers):
            # Retain top-k important tokens
            pruned_cot = retain_top_k_tokens(cot_response, mi_scores)
            
            results.append({
                'question': example['question'],
                'original_cot': cot_response,
                'pruned_cot': pruned_cot,
                'answer': answer
            })
        
        return results

class DeepSeekR1Wrapper:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
    def generate(self, prompt, max_tokens=1024):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return ""

if __name__ == "__main__":
    gsm8k_dataset = load_dataset("openai/gsm8k", "main")
    arc_agi_dataset = load_dataset("dataartist/arc-agi")
    humaneval_dataset = load_dataset("openai/openai_humaneval")
    
    # Initialize with your OpenAI API key as temporary solution
    your_model = DeepSeekR1Wrapper(api_key="sk-your-openai-key-here")
    
    # Generate responses with token importance analysis
    gsm8k_results = generate_responses_with_importance(gsm8k_dataset, your_model)
    arc_agi_results = generate_responses_with_importance(arc_agi_dataset, your_model)
    humaneval_results = generate_responses_with_importance(humaneval_dataset, your_model)