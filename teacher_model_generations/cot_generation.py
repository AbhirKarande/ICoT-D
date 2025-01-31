import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def calculate_mutual_information(cot_responses, answers):
    """
    Calculate mutual information between CoT tokens and answers
    """
    # Vectorize CoT responses and answers
    cot_vectorizer = CountVectorizer(analyzer='word')
    answer_vectorizer = CountVectorizer(analyzer='word')
    
    # Fit and transform the data
    cot_matrix = cot_vectorizer.fit_transform(cot_responses)
    answer_matrix = answer_vectorizer.fit_transform(answers)
    
    # Calculate mutual information
    mi_scores = defaultdict(float)
    vocab = cot_vectorizer.get_feature_names_out()
    
    for i, token in enumerate(vocab):
        # Calculate H(Y)
        p_y = np.mean(answer_matrix, axis=0)
        h_y = -np.sum(p_y * np.log2(p_y + 1e-10))
        
        # Calculate H(Y|T_i)
        clf = MultinomialNB()
        clf.fit(cot_matrix[:, i].toarray(), answer_matrix)
        h_y_given_t = clf.score(cot_matrix[:, i].toarray(), answer_matrix)
        
        # Calculate MI
        mi_scores[token] = h_y - h_y_given_t
    
    return mi_scores

def retain_top_k_tokens(cot_response, mi_scores, k=0.5):
    """
    Retain top-k important tokens based on MI scores
    """
    tokens = cot_response.split()
    # Get scores for tokens in this response
    token_scores = [(token, mi_scores.get(token, 0)) for token in tokens]
    # Sort by score descending
    token_scores.sort(key=lambda x: x[1], reverse=True)
    # Calculate number of tokens to retain
    num_to_retain = int(len(tokens) * k)
    # Get top tokens
    top_tokens = [t[0] for t in token_scores[:num_to_retain]]
    return ' '.join(top_tokens)

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

# Generate responses with token importance analysis
gsm8k_results = generate_responses_with_importance(gsm8k_dataset, your_model)
arc_agi_results = generate_responses_with_importance(arc_agi_dataset, your_model)
humaneval_results = generate_responses_with_importance(humaneval_dataset, your_model)