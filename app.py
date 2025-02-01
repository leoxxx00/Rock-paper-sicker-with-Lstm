import os
import random
import csv
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Disable GPU for Hugging Face Spaces
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# File paths
csv_filename = "game_moves.csv"
model_filename = "lstm_model.h5"

# Mapping choices to numerical values
choices = {'rock': 0, 'paper': 1, 'scissors': 2}
rev_choices = {0: 'rock', 1: 'paper', 2: 'scissors'}

# Ensure CSV exists
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Player Choice", "Computer Choice", "Result"])

def get_computer_choice(model, past_moves):
    """ Predicts player's next move and counteracts it. """
    if len(past_moves) < 5:  # Adjusted sequence length to 5
        return random.choice(["rock", "paper", "scissors"])
    
    # Prepare input data for prediction
    sequence = [choices[move] for move in past_moves[-5:]]
    sequence = pad_sequences([sequence], maxlen=5)
    
    prediction = model.predict(sequence, verbose=0)
    predicted_choice = rev_choices[np.argmax(prediction)]
    
    # Counteract the predicted choice
    counter_choices = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    return counter_choices[predicted_choice]

def get_winner(player, computer):
    """ Determines the winner of the game. """
    if player == computer:
        return "It's a tie!"
    elif (player == "rock" and computer == "scissors") or \
         (player == "scissors" and computer == "paper") or \
         (player == "paper" and computer == "rock"):
        return "You win!"
    else:
        return "Computer wins!"

def save_move(player, computer, result):
    """ Saves game move to CSV file. """
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([player, computer, result])

def load_data():
    """ Loads past player moves from CSV file. """
    try:
        with open(csv_filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            return [row[0] for row in reader]
    except FileNotFoundError:
        return []

def train_lstm_model(data):
    """ Trains an LSTM model to predict the player's next move. """
    if len(data) < 6:  # Adjusted for longer training sequences
        return None  # Not enough data for meaningful training
    
    tokenizer = Tokenizer(num_words=3)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    
    X = pad_sequences(sequences, maxlen=5)  # Using longer sequences
    y = np.array([choices[move] for move in data[1:]])

    model = Sequential([
        Embedding(input_dim=3, output_dim=10, input_length=5),
        LSTM(30, return_sequences=False),  # Increased LSTM units for better learning
        Dense(3, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    if len(X) > 1:
        model.fit(X[:-1], y, epochs=15, verbose=0)  # Increased epochs for better training
        model.save(model_filename)  # Save trained model

    return model

# Load past moves
past_moves = load_data()

# Load existing model if available
if os.path.exists(model_filename):
    model = load_model(model_filename)
else:
    model = train_lstm_model(past_moves) if len(past_moves) >= 6 else None

def play_game(player_choice):
    """ Handles the game logic and returns the result. """
    global past_moves, model

    if player_choice not in choices:
        return "Invalid choice. Choose rock, paper, or scissors."

    # Ensure model exists before predicting
    if model is None:
        computer_choice = random.choice(["rock", "paper", "scissors"])
    else:
        computer_choice = get_computer_choice(model, past_moves)

    result = get_winner(player_choice, computer_choice)

    # Save the move
    save_move(player_choice, computer_choice, result)
    past_moves.append(player_choice)

    # Retrain the model only when enough new data is available
    if len(past_moves) >= 6:
        model = train_lstm_model(past_moves)

    return f"Computer chose: {computer_choice}\n{result}"

# Gradio UI
iface = gr.Interface(
    fn=play_game,
    inputs=gr.Radio(["rock", "paper", "scissors"], label="Choose your move"),
    outputs="text",
    title="Rock, Paper, Scissors AI",
    description="Play against an AI that learns from your moves and tries to beat you!"
)

if __name__ == "__main__":
    iface.launch()
