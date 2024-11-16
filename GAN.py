import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from confection import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

class Generator_LSTM(nn.Module):
    def __init__(self, config):
        super(Generator_LSTM, self).__init__()
        self.config = config
        self.output_size = self.config['dataset']['len_dict']
        self.input_size = self.config['generator']['noise_size']
        self.hidden_size = self.config['generator']['hidden_size']
        self.max_sequence_length = self.config['dataset']['max_sequence_length']
        self.num_layers = self.config['generator']['num_layers']

        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, noise_vector):
        batch_size = noise_vector.size(0)

        h_t = torch.zeros(batch_size, self.hidden_size, device=noise_vector.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=noise_vector.device)
        
        x_t = noise_vector  
        
        outputs = []
        for _ in range(self.max_sequence_length):

            h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))

            logits = self.output_layer(h_t)  
            probs = F.softmax(logits, dim=-1)  
            outputs.append(probs.unsqueeze(0))  

            x_t_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

            x_t = F.one_hot(x_t_indices, num_classes=probs.size(-1)).float()

        outputs = torch.cat(outputs, dim=0)
        outputs = torch.transpose(torch.argmax(outputs, dim=-1))
        
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
        x = self.Embedding(x)
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

    optimizer_G = optim.Adam(model.Generator.parameters(), lr = config['training']['learning_rate'], betas = (0.5, 0.999))
    optimizer_D = optim.Adam(model.Discriminator.parameters(), lr = config['training']['learning_rate'], betas = (0.5, 0.999))
    
    if config['dataset']['single_chain']:
        real = torch.load('single_chain_protein_sequences.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    real.to(device)

    for epoch in range(epochs):
        input = torch.randn((config['dataset']['dataset_size'], config['generator']['noise_size']))
        gen = model.Generator(input)

        input.to(device)
        gen.to(device)

        probs_real = model.Discriminator(real)
        probs_gen = model.Discriminator(gen)

        loss_gen = criterion(probs_gen, torch.ones_like(probs_gen))
        loss_discrim = criterion(probs_real, torch.ones_like(probs_real)) + loss_gen

        optimizer_D.zero_grad()
        loss_discrim.backward()
        optimizer_D.steP()

        optimizer_G.zero_grad()
        loss_gen.backward()
        optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}], d_loss: {loss_discrim.item()}, g_loss: {loss_gen.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to config file")

    args = parser.parse_args()

    config = Config().from_disk(args.c)
    gan = GAN(config)

    Train(gan, config)