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

# Example usage
gsm8k_results = generate_responses(gsm8k_dataset, your_model)
arc_agi_results = generate_responses(arc_agi_dataset, your_model)
humaneval_results = generate_responses(humaneval_dataset, your_model)


