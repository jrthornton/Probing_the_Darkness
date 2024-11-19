import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from confection import Config
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import pandas as pd

def save_combined_logs(loss_logs, generated_sequences):
    # Convert loss logs and generated sequences to DataFrames
    loss_logs_df = pd.DataFrame(loss_logs)
    
    gen_seq_df = pd.DataFrame(np.vstack(generated_sequences))
    # Define the Excel file path
    file_path = f'combined_logs.xlsx'

    # Write to a single Excel file with multiple sheets
    with pd.ExcelWriter(file_path) as writer:
        loss_logs_df.to_excel(writer, sheet_name='Loss Logs', index=False)
        gen_seq_df.to_excel(writer, sheet_name='Generated Sequences', index=False)

class Generator_LSTM(nn.Module):
    # Generator class
    def __init__(self, config):
        super(Generator_LSTM, self).__init__()
        self.config = config
        # Define hyper parameters:
        self.output_size = self.config['dataset']['len_dict']
        self.input_size = self.config['generator']['input_size']
        self.hidden_size = self.config['generator']['hidden_size']
        self.max_sequence_length = self.config['dataset']['max_sequence_length']
        self.num_layers = self.config['generator']['num_layers']
        
        # Define layers: Embedding, input, lstm_cell, output
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, noise_vector, sequence_lengths):
        # Input is a noise_vector with the first dimension of each example equal to the sequence length
        batch_size = noise_vector.size(0)

        # Initialize h_t, c_t, x_t
        h_t = torch.zeros(batch_size, self.hidden_size, device=noise_vector.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=noise_vector.device)

        # Pass x_t through input layer to get embedded input
        x_t = self.input_layer(noise_vector) 

        # Initialize tensor for storing output
        outputs = torch.zeros(batch_size, self.max_sequence_length, device=noise_vector.device, dtype=torch.long)

        # Loop through LSTMCell for each sequence value to be created
        for t in range(self.max_sequence_length):
            # LSTM forward for sequence element t
            h_t, c_t = self.lstm(x_t, (h_t, c_t))

            # Map hidden state to class logits
            logits = self.output_layer(h_t)
            probs = F.softmax(logits, dim=-1)

            # Sample output token
            x_t_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Mask to avoid writing over padded elements
            valid_mask = t < sequence_lengths
            valid_mask = valid_mask.to(noise_vector.device)
            outputs[:, t] = torch.where(valid_mask, x_t_indices, torch.zeros_like(x_t_indices, device=noise_vector.device))
            #outputs[:, t] = x_t_indices
            # Map class token to hidden size for next step
            x_t = self.embedding(x_t_indices)
        return outputs

class Discriminator_LSTM(nn.Module):
    # Discriminator with LSTM network
    def __init__(self, config):
        super(Discriminator_LSTM, self).__init__()
        self.config = config
        # Define hyper parameters:
        self.embedding_dim = self.config['discriminator']['embedding_dim']
        self.hidden_size = self.config['discriminator']['hidden_size']
        self.num_layers = self.config['discriminator']['num_layers']
        self.len_dict = self.config['dataset']['len_dict'] + 1 # Plus one to include padding token
        
        # Define layers: embedding, LSTM, fully connected
        self.embedding = nn.Embedding(self.len_dict, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        
        # Sigmoid activation for classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences):
        # sequences: batch_size x seq_len
        x = self.embedding(sequences)
        outputs, (h_n, c_n) = self.lstm(x)

        h_last = h_n[-1]
        out = self.fc(h_last)
        probs = self.sigmoid(out)
        return probs

class Discriminator_Linear(nn.Module):
    # Discriminator with linear architecture
    def __init__(self, config):
        super(Discriminator_Linear, self).__init__()
        self.config = config
        # Define layers: embedding, fc1, fc2, fc3
        self.embedding = nn.Embedding(self.config['dataset']['len_dict'], self.config['discriminator']['embedding_dim'])
        self.fc1 = nn.Linear(self.config['discriminator']['embedding_dim'] * (self.config['dataset']['max_sequence_length']), self.config['discriminator']['hidden_size_1'])
        self.fc2 = nn.Linear(self.config['discriminator']['hidden_size_1'], self.config['discriminator']['hidden_size_2'])  # Binary classification: real or fake
        self.fc3 = nn.Linear(self.config['discriminator']['hidden_size_2'], 1)  # Binary classification: real or fake
        
        # Sigmoid activation for classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences):
        x = sequences
        x = self.embedding(x)
        # Reshape x based on embedding size:
        x = x.view(-1, self.config['discriminator']['embedding_dim'] * (self.config['dataset']['max_sequence_length']))
        x = self.fc1(x)
        x = self.fc2(x)
        probs = self.sigmoid(x)
        return probs

class GAN(nn.Module):
    # Define GAN network composed of generator and discriminator networks
    def __init__(self, config):
        super(GAN, self).__init__() 
        self.config = config
        self.Generator = Generator_LSTM(config)
        self.Discriminator = Discriminator_Linear(config)

def Train(model, config):
    # Training function
    # Get hyperparameters from config:
    epochs = config['training']['epochs']
    seed = config['training']['seed']
    
    # BCE Loss for binary classification
    criterion = nn.BCELoss()

    # Optimizers for each part of GAN
    optimizer_G = optim.Adam(model.Generator.parameters(), lr = config['training']['learning_rate_gen'], betas = (0.5, 0.999))
    optimizer_D = optim.Adam(model.Discriminator.parameters(), lr = config['training']['learning_rate_discrim'], betas = (0.5, 0.999))
    
    # Load data (set up for only single chain proteins currently)
    if config['dataset']['single_chain']:
        real = torch.load('single_chain_protein_sequences.pt')
        real_lengths = torch.tensor(pd.read_excel("single_chain_protein_sequences.xlsx")["Length"].to_numpy())
    
    # Preprocess sequences limiting them to max_sequence_length from config
    mask = real_lengths < config['dataset']['max_sequence_length']

    real = real[mask]
    real_lengths = real_lengths[mask]

    real = real[:, :config['dataset']['max_sequence_length']]
    
    real = real - 5
    real = torch.clamp(real, min = 0)

    # Create dataset dropping last examples for easier handling
    dataset = TensorDataset(real, real_lengths)
    dataloader = DataLoader(dataset, batch_size = config['training']['batch_size'], shuffle = True, drop_last=True)
    
    # Set cuda if avaliable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)

    # Initialize logs
    loss_logs = []
    generated_sequences = []
    batch_idx = 0
    # Training Loop:
    for epoch in range(epochs):
        for batch_data, lengths in dataloader:
            batch_data = batch_data.to(device)

            # Initialize input to random noise with first dimension containing the sequence length
            input = torch.zeros((config['training']['batch_size'], config['generator']['input_size']))
            input[:, 0] = lengths
            input = input.to(device)
            
            # Generate synthetic sequences
            gen = model.Generator(input, lengths)

            # Get probabilities of real and synthetic from discriminator
            probs_real = model.Discriminator(batch_data)
            probs_gen = model.Discriminator(gen.detach())

            # Calculate Loss for discriminator
            loss_D_real = criterion(probs_real, torch.ones_like(probs_real))
            loss_D_fake = criterion(probs_gen, torch.zeros_like(probs_gen))
            loss_discrim = loss_D_real + loss_D_fake
            
            # Step optimizer for discriminator
            optimizer_D.zero_grad()
            loss_discrim.backward()
            optimizer_D.step()
            
            # Calc loss for generator
            probs_gen = model.Discriminator(gen)
            loss_gen = criterion(probs_gen, torch.ones_like(probs_gen))  # Target labels are ones

            # Step optimizer for generator
            optimizer_G.zero_grad()
            loss_gen.backward()
            optimizer_G.step()

            # Log loss values
            loss_logs.append({'Epoch': epoch + 1,
            'Batch': batch_idx + 1,
            'Discriminator Loss': loss_discrim.item(),
            'Generator Loss': loss_gen.item()
            })
            batch_idx += 1
            print(f"Epoch [{epoch+1}/{epochs}], d_loss: {loss_discrim.item()}, g_loss: {loss_gen.item()}")
        
        # At the end of each epoch create 5 sampled amino acid sequences of random lengths
        with torch.no_grad():
            input = torch.zeros((5, config['generator']['input_size'])).to(device)
            
            gen_sequences = model.Generator(input, torch.randint(low=30, high=config['dataset']['max_sequence_length'], size=(5,)))
            generated_sequences.append(gen_sequences.cpu().numpy())
    
    # Log results
    save_combined_logs(loss_logs, generated_sequences)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Parser for specifying config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to config file")
    args = parser.parse_args()
    config = Config().from_disk(args.c)

    # Define GAN
    gan = GAN(config)

    # Train GAN
    Train(gan, config)