import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.builder import RLModel
from environment.env import Environment as ChessEnv
from utils import config
import lib

class ChessDataset(Dataset):
    """Custom Dataset for chess positions."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, probs, winner = self.data[idx]
        # Convert FEN to input tensor (19, 8, 8)
        input_tensor = ChessEnv.fen_to_input(state)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        # Convert move probabilities to output vector (4672,)
        board = chess.Board(state)
        prob_vector = lib.moves_to_output_vector(probs, board)
        # Convert probability distribution to class index
        prob_vector = torch.tensor(prob_vector, dtype=torch.float32)  # Shape: (73, 8, 8)
        prob_vector = prob_vector.flatten()  # Shape: (4672,)
        class_index = torch.argmax(prob_vector, dim=0)  # Shape: () (scalar)
        # Convert winner to float
        value = float(winner)
        value = torch.tensor(value, dtype=torch.float32)
        return input_tensor, class_index, value


class Trainer:
    def __init__(self, model):
        self.model = model.to(config.DEVICE)
        self.device = config.DEVICE
        self.batch_size = config.BATCH_SIZE
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def train_random_batches(self, dataset, num_epochs=2):
        """Train the model on random batches from the dataset for num_epochs passes."""
        history = {'loss': [], 'policy_head_loss': [], 'value_head_loss': []}
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Number of batches per epoch, with a minimum of 5 (as in original)
        batches_per_epoch = max(5, len(dataset) // self.batch_size)
        print(f"Training for {num_epochs} epochs, {batches_per_epoch} batches per epoch...")

        self.model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            # Reset DataLoader iterator for each epoch to ensure fresh shuffling
            data_iterator = iter(data_loader)
            for _ in tqdm(range(batches_per_epoch)):
                try:
                    # Sample a random batch
                    batch = next(data_iterator)
                except StopIteration:
                    # If iterator is exhausted, reset it
                    data_iterator = iter(data_loader)
                    batch = next(data_iterator)
                inputs, prob_targets, value_targets = [x.to(self.device) for x in batch]

                # Forward pass
                self.optimizer.zero_grad()
                policy_logits, value_pred = self.model(inputs)

                # Compute losses
                policy_loss = self.policy_loss_fn(policy_logits, prob_targets)
                value_loss = self.value_loss_fn(value_pred.squeeze(), value_targets)
                total_loss = policy_loss + value_loss  # Equal weighting, as in AlphaZero

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Record losses
                history['loss'].append(total_loss.item())
                history['policy_head_loss'].append(policy_loss.item())
                history['value_head_loss'].append(value_loss.item())

        print('returning history')
        return history

    def plot_loss(self, history):
        """Plot and save loss curves."""
        df = pd.DataFrame(history)
        plt.plot(df['loss'], label='Total Loss')
        plt.plot(df['policy_head_loss'], label='Policy Head Loss')
        plt.plot(df['value_head_loss'], label='Value Head Loss')
        plt.legend()
        plt.title(f"Loss over time\nLearning rate: {config.LEARNING_RATE}")
        os.makedirs(config.LOSS_PLOTS_FOLDER, exist_ok=True)
        plt.savefig(f"{config.LOSS_PLOTS_FOLDER}/loss-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png")
        plt.close()

    def save_model(self):
        import shutil
        """Save the model state dict."""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        path = f"{config.MODEL_DIR}/model-{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pth"
        shutil.move(f'{config.MODEL_DIR}/model.pt', path)
        print('moving model.pt to', path)
        torch.save(self.model.state_dict(), f'{config.MODEL_DIR}/model.pt')
        print(f"Model trained. Saved model to {path}")

    def train_all(self) -> RLModel:
        # Load data
        folder = config.MEM_DIR
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        print(f"Loading all games in {folder}...")
        data = []
        for file in files:
            data.append(np.load(os.path.join(folder, file), allow_pickle=True))
        data = np.concatenate(data)
        print(f"{len(data[data[:,2] > 0])} positions won by white")
        print(f"{len(data[data[:,2] < 0])} positions won by black")
        print(f"{len(data[data[:,2] == 0])} positions drawn")
        print(f"Training with {len(data)} positions")

        # Create dataset and trainer
        dataset = ChessDataset(data)

        # Train and plot
        history = self.train_random_batches(dataset, num_epochs=10)
        print("Training complete.")
        self.save_model()
        print("Model saved.")
        return self.model
