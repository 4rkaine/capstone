import streamlit as st
import json
import random
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from siamese_network import SiameseLSTM
import pickle

import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')


if 'siamese_model' not in st.session_state:
    vocab_path = './siamese_network_vocab.pkl'
    try:
        with open(vocab_path, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
    except:
        vocab = {}

    vocab_size = len(vocab)
    hidden_dim = 64
    embedding_dim = 100

    st.session_state.vocab = vocab

    siamese_model = SiameseLSTM(vocab_size, embedding_dim, hidden_dim)

    model_weights = torch.load('./siamese_model.pt')
    siamese_model.load_state_dict(model_weights)
    siamese_model.eval()
    st.session_state.siamese_model = siamese_model

# load the questions and answers from the JSON file
def load_questions_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return random.sample(data, 10)  # select 10 random samples - TODO remove this

def get_random_samples(data, sample_size=10):
    return random.sample(data, sample_size)

def modify_question_with_option(question, option):
    question_words = ["what", "who", "when", "where", "why", "how"]
    for word in question_words:
        if word in question.lower():
            modified_question = question.replace("?", "")  # Remove question mark only when replacing
            return modified_question.lower().replace(word, option).capitalize(), option
    return question + " " + option, None

def get_embedding_sbert(text):
    sbert_model = st.session_state.sbert_model
    return sbert_model.encode([text])[0]

def preprocess_text_with_siamese(text):
    # Tokenize and preprocess the text as needed for Siamese model
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english') 
    tokens = tokenizer(text)
    
    # Convert tokens to numerical tokens using the vocabulary
    vocab = st.session_state.vocab
    numerical_tokens = [vocab[token] for token in tokens if token in vocab]

    # Pad or truncate the numerical tokens to a fixed length (max_sequence_length)
    max_sequence_length = 128 
    if len(numerical_tokens) < max_sequence_length:
        numerical_tokens += [0] * (max_sequence_length - len(numerical_tokens))
    else:
        numerical_tokens = numerical_tokens[:max_sequence_length]

    # Convert the numerical tokens to a PyTorch tensor
    numerical_tensor = torch.tensor(numerical_tokens)
    
    return numerical_tensor

# Function to compute similarity using Siamese network
def compute_similarity_siamese(text1, text2):
    processed_text1 = [preprocess_text_with_siamese(text1)]
    processed_text2 = [preprocess_text_with_siamese(text2)]
    
    processed_text1 = pad_sequence(processed_text1, batch_first=True, padding_value=0)
    processed_text2 = pad_sequence(processed_text2, batch_first=True, padding_value=0)

    siamese_model = st.session_state.siamese_model
    # Make predictions using Siamese model
    with torch.no_grad():
        output1, output2 = siamese_model(processed_text1, processed_text2)
        similarity_scores = torch.abs(output1 - output2)
    return similarity_scores

def compute_similarity_sbert(text1, text2):
    emb1 = get_embedding_sbert(text1)
    emb2 = get_embedding_sbert(text2)
    emb1_2d = emb1.reshape(1, -1)
    emb2_2d = emb2.reshape(1, -1)
    similarity = cosine_similarity(emb1_2d, emb2_2d)[0][0]
    return similarity

def main():
    st.title("Automated Answer Evaluation")

    # Load all data initially if not already loaded
    if 'all_data' not in st.session_state:
        st.session_state.all_data = load_questions_from_json("./data/sciq/train.json")

    # Display a button to go to the next question
    if st.button('Next'):
        # Update current_question and correct_answer with a new random question
        current_question = random.choice(st.session_state.all_data)
        st.session_state.current_question = current_question
        st.session_state.correct_answer = current_question["correct_answer"]

    # Ensure all_data is available before proceeding
    if 'all_data' in st.session_state:
        # Retrieve the current question and correct answer
        if 'current_question' not in st.session_state:
            current_question = random.choice(st.session_state.all_data)
            st.session_state.current_question = current_question
            st.session_state.correct_answer = current_question["correct_answer"]
        else:
            current_question = st.session_state.current_question

        # Define correct_answer here so it's accessible outside of the if block
        correct_answer = st.session_state.correct_answer

        # Display the current question
        st.write("Question:", current_question["question"])

        # Display the supporting text
        support_text = current_question["support"]
        st.write("Supporting Text:", support_text)

    options = [
        current_question["correct_answer"],
        current_question["distractor1"],
        current_question["distractor2"],
        current_question["distractor3"]
    ]

    max_similarity_sbert = 0
    predicted_option_sbert = ""
    max_similarity_siamese = 0
    predicted_option_siamese = ""

    column_headers = ["", "SBERT Cosine Similarity", "Siamese Networks"]
    option_scores = []

    for option in options:
        modified_option, used_option = modify_question_with_option(current_question["question"], option)

        # Compute similarity using SBERT
        similarity_score_sbert = compute_similarity_sbert(modified_option, support_text) * 100

        # Compute similarity using Siamese network
        similarity_score_siamese = compute_similarity_siamese(modified_option, support_text) * 100

        option_scores.append([option, f"{float(similarity_score_sbert):.2f}%", f"{float(similarity_score_siamese):.2f}%"])

        if similarity_score_sbert > max_similarity_sbert:
            max_similarity_sbert = similarity_score_sbert
            predicted_option_sbert = option

        if similarity_score_siamese > max_similarity_siamese:
            max_similarity_siamese = similarity_score_siamese
            predicted_option_siamese = option

    st.write("Options and Scores:")

    option_scores.append(["Predicted Answer", predicted_option_sbert, predicted_option_siamese])

    if correct_answer == predicted_option_sbert:
        sbert_prediction = "Correct!"
    else:
        sbert_prediction = "Wrong!"

    if correct_answer == predicted_option_siamese:
        siamese_prediction = "Correct!"
    else:
        siamese_prediction = "Wrong!"

    option_scores.append(["Evaluation", sbert_prediction, siamese_prediction])
    option_table = pd.DataFrame(option_scores, columns=column_headers)
    st.table(option_table)

    st.write("Correct Answer:", correct_answer)

    # Add a text area for user input and evaluation
    st.write("\n\nEvaluate Your Answer")
    user_answer = st.text_area("Type your answer here:")
    evaluate_button = st.button("Evaluate")

    if evaluate_button:
        if user_answer == correct_answer:
            st.markdown('<span style="color:green">Your Answer is Correct!</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:red">Your Answer is Wrong!</span>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

