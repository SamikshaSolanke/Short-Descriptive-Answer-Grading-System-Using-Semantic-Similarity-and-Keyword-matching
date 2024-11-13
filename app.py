import nltk
from nltk.corpus import wordnet
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_md')

# Function to calculate keyword matching score
def keyword_matching_score(expected_answer, student_answer, expected_keywords):
    expected_tokens = set(nltk.word_tokenize(expected_answer.lower()))
    student_tokens = set(nltk.word_tokenize(student_answer.lower()))
    common_tokens = expected_tokens.intersection(student_tokens)
    
    # Check for synonyms
    keyword_score = (len(common_tokens) / len(expected_tokens))*100
    return keyword_score

# Function to calculate semantic similarity score
def semantic_similarity_score(expected_answer, student_answer):
    expected_doc = nlp(expected_answer)
    student_doc = nlp(student_answer)
    similarity_score = (expected_doc.similarity(student_doc))*100
    return similarity_score

# Function to calculate final score
def calculate_final_score(expected_answer, student_answer, expected_keywords):
    sem_sim_score = semantic_similarity_score(expected_answer, student_answer)
    keyword_score = keyword_matching_score(expected_answer, student_answer, expected_keywords)
    final_score = 0.5 * keyword_score + 0.5 * sem_sim_score
    return final_score

# Function to load text files and calculate score
def load_files_and_calculate_score(expected_file, student_file, expected_keywords):
    with open(expected_file, 'r') as f:
        expected_answer = f.read()
    with open(student_file, 'r') as f:
        student_answer = f.read()
    
    score = calculate_final_score(expected_answer, student_answer, expected_keywords)
    
    # Provide detailed feedback
    print('Keyword Matching Score:', keyword_matching_score(expected_answer, student_answer, expected_keywords))
    print('Semantic Similarity Score:', semantic_similarity_score(expected_answer, student_answer))
    print('Final Score:', score)
    
    return score

# Example usage
expected_file = 'expected_answer.txt'
student_file = 'student_answer.txt'
expected_keywords = ['keyword1', 'keyword2', 'keyword3']
score = load_files_and_calculate_score(expected_file, student_file, expected_keywords)
