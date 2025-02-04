

#make a class around this
class TokenScorer:
    def __init__(self):
        pass
    
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