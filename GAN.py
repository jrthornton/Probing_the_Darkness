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

    #print(f"Combined logs saved to {file_path}")

class Generator_LSTM(nn.Module):
    def __init__(self, config):
        super(Generator_LSTM, self).__init__()
        self.config = config
        self.output_size = self.config['dataset']['len_dict']
        self.input_size = self.config['dataset']['len_dict']
        self.hidden_size = self.config['generator']['hidden_size']
        self.max_sequence_length = self.config['dataset']['max_sequence_length']
        self.num_layers = self.config['generator']['num_layers']

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, noise_vector, sequence_lengths):
        batch_size = noise_vector.size(0)

        h_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=noise_vector.device)
        c_t = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=noise_vector.device)
        
        x_t = noise_vector 

        outputs = torch.zeros(batch_size, self.max_sequence_length, device=noise_vector.device, dtype=torch.long)

        for t in range(self.max_sequence_length):
            x_t = x_t.unsqueeze(0)  # Shape: (1, batch_size, input_size)

            output, (h_t, c_t) = self.lstm(x_t, (h_t, c_t))  # output: (1, batch_size, hidden_size)
            output = output.squeeze(0)  # Shape: (batch_size, hidden_size)

            logits = self.output_layer(output)
            probs = F.softmax(logits, dim=-1)

            x_t_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

            valid_mask = t < sequence_lengths
            valid_mask = valid_mask.to(noise_vector.device)
            outputs[:, t] = torch.where(valid_mask, x_t_indices, torch.zeros_like(x_t_indices, device=noise_vector.device))
            x_t = F.one_hot(x_t_indices, num_classes=probs.size(-1)).float()

            #h_t, c_t = self.lstm(x_t, (h_t, c_t))

            #logits = self.output_layer(h_t)  
            #probs = F.softmax(logits, dim=-1)  

            #x_t_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

            #valid_mask = t < sequence_lengths
            #valid_mask = valid_mask.to(noise_vector.device)
            #outputs[:, t] = torch.where(valid_mask, x_t_indices, torch.zeros_like(x_t_indices, device=noise_vector.device))
            #x_t = F.one_hot(x_t_indices, num_classes=probs.size(-1)).float()

        
        outputs = torch.transpose(outputs, 0, 1)
        return outputs

class Discriminator_Linear(nn.Module):
    def __init__(self, config):
        super(Discriminator_Linear, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config['dataset']['len_dict'], self.config['discriminator']['embedding_dim'])
        self.fc1 = nn.Linear(self.config['discriminator']['embedding_dim'] * (self.config['dataset']['max_sequence_length']), self.config['discriminator']['hidden_size_1'])
        self.fc2 = nn.Linear(self.config['discriminator']['hidden_size_1'], self.config['discriminator']['hidden_size_2'])  # Binary classification: real or fake
        self.fc3 = nn.Linear(self.config['discriminator']['hidden_size_2'], 1)  # Binary classification: real or fake
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences):
        x = sequences
        x = self.embedding(x)
        x = x.view(-1, self.config['discriminator']['embedding_dim'] * (self.config['dataset']['max_sequence_length']))
        x = self.fc1(x)
        x = self.fc2(x)
        probs = self.sigmoid(x)
        return probs

class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__() 
        self.config = config
        self.Generator = Generator_LSTM(config)
        self.Discriminator = Discriminator_Linear(config)

def Train(model, config):
    epochs = config['training']['epochs']
    seed = config['training']['seed']
    
    criterion = nn.BCELoss()

    optimizer_G = optim.Adam(model.Generator.parameters(), lr = config['training']['learning_rate_gen'], betas = (0.5, 0.999))
    optimizer_D = optim.Adam(model.Discriminator.parameters(), lr = config['training']['learning_rate_discrim'], betas = (0.5, 0.999))
    
    if config['dataset']['single_chain']:
        real = torch.load('single_chain_protein_sequences.pt')
        real_lengths = torch.tensor(pd.read_excel("single_chain_protein_sequences.xlsx")["Length"].to_numpy())
    
    mask = real_lengths < config['dataset']['max_sequence_length']

    real = real[mask]
    real_lengths = real_lengths[mask]

    real = real[:, :config['dataset']['max_sequence_length']]
    print(real.shape)
    dataset = TensorDataset(real, real_lengths)
    dataloader = DataLoader(dataset, batch_size = config['training']['batch_size'], shuffle = True, drop_last=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    loss_logs = []
    generated_sequences = []
    

    for epoch in range(epochs):
        batch_idx = 0
        for batch_data, lengths in dataloader:
            batch_data = batch_data.to(device)

            input = torch.randn((config['training']['batch_size'], config['dataset']['len_dict']))
            input = input.to(device)
            
            gen = model.Generator(input, lengths)

            probs_real = model.Discriminator(batch_data)
            probs_gen = model.Discriminator(gen.detach())

            loss_D_real = criterion(probs_real, torch.ones_like(probs_real))
            loss_D_fake = criterion(probs_gen, torch.zeros_like(probs_gen))
            loss_discrim = loss_D_real + loss_D_fake
            
            optimizer_D.zero_grad()
            loss_discrim.backward()
            optimizer_D.step()
            
            probs_gen = model.Discriminator(gen)
            loss_gen = criterion(probs_gen, torch.ones_like(probs_gen))  # Target labels are ones

            optimizer_G.zero_grad()
            loss_gen.backward()
            optimizer_G.step()

            loss_logs.append({'Epoch': epoch + 1,
            'Batch': batch_idx + 1,
            'Discriminator Loss': loss_discrim.item(),
            'Generator Loss': loss_gen.item()
            })
            batch_idx += 1
            print(f"Epoch [{epoch+1}/{epochs}], d_loss: {loss_discrim.item()}, g_loss: {loss_gen.item()}")
        
        with torch.no_grad():
            input = torch.randn((5, config['dataset']['len_dict'])).to(device)
            gen_sequences = model.Generator(input, torch.randint(low=30, high=500, size=(1,5)))
            gen_sequences = torch.transpose(gen_sequences, 0, 1)
            generated_sequences.append(gen_sequences.cpu().numpy())
        
    save_combined_logs(loss_logs, generated_sequences)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to config file")
    
    args = parser.parse_args()

    config = Config().from_disk(args.c)
    gan = GAN(config)

    Train(gan, config)