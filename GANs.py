import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_CNN(nn.Module):
    def __init__(self, config):
        super(Generator_CNN, self).__init__()
        self.config = config
        # Define hyperparameters:
        self.output_size = self.config['generator']['len_dict']
        self.input_size = self.config['generator']['input_size']
        self.hidden_size = self.config['generator']['hidden_size']
        self.max_sequence_length = self.config['dataset']['max_sequence_length']
        self.num_layers = self.config['generator']['num_layers']
        
        # Input layer to map noise vector to hidden representation
        self.input_layer = nn.Linear(self.input_size, self.hidden_size * self.max_sequence_length)
        self.input_layer = nn.Linear(self.input_size, self.hidden_size * self.max_sequence_length)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            # Additional layers can be added here
        )
        
        # Output layer to map hidden representation to output vocabulary size
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, noise_vector, sequence_lengths):
        batch_size = noise_vector.size(0)
        sequence_lengths = sequence_lengths.to(noise_vector.device)
        # Map noise vector to hidden representation
        x = self.input_layer(noise_vector)  # Shape: (batch_size, hidden_size * max_sequence_length)
        
        # Reshape to (batch_size, hidden_size, max_sequence_length)
        x = x.view(batch_size, self.hidden_size, self.max_sequence_length)
        
        # Apply convolutional layers
        x = self.conv_layers(x)  # Shape remains (batch_size, hidden_size, max_sequence_length)
        
        # Transpose to (batch_size, max_sequence_length, hidden_size)
        x = x.transpose(1, 2)
        
        # Apply output layer to each time step
        logits = self.output_layer(x)  # Shape: (batch_size, max_sequence_length, output_size)
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample output tokens
        x_t_indices = torch.multinomial(probs.view(-1, self.output_size), num_samples=1).squeeze(1)
        x_t_indices = x_t_indices.view(batch_size, self.max_sequence_length)
        
        # Mask to avoid writing over padded elements
        valid_mask = torch.arange(self.max_sequence_length, device=noise_vector.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        outputs = torch.where(valid_mask, x_t_indices, torch.zeros_like(x_t_indices))
        
        return outputs

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

class Discriminator_CNN(nn.Module):
    def __init__(self, config):
        super(Discriminator_CNN, self).__init__()
        self.config = config
        # Define hyperparameters
        self.embedding_dim = self.config['discriminator']['embedding_dim']
        self.num_filters = self.config['discriminator'].get('num_filters', 128)
        self.filter_sizes = self.config['discriminator'].get('filter_sizes', [3, 4, 5])
        self.len_dict = self.config['dataset']['len_dict'] + 1  # Plus one to include padding token

        # Embedding layer
        self.embedding = nn.Embedding(self.len_dict, self.embedding_dim)

        # Convolutional layers with different filter sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(filter_size, self.embedding_dim))
            for filter_size in self.filter_sizes
        ])

        # Fully connected layer
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), 1)

        # Sigmoid activation for classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, sequences):
        # sequences: (batch_size, seq_len)
        x = self.embedding(sequences)  # Shape: (batch_size, seq_len, embedding_dim)

        # Add channel dimension for convolution: (batch_size, 1, seq_len, embedding_dim)
        x = x.unsqueeze(1)

        # Apply convolution and pooling
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))  # Shape: (batch_size, num_filters, seq_len - filter_size + 1, 1)
            conv_out = conv_out.squeeze(3)  # Remove the last dimension: (batch_size, num_filters, seq_len - filter_size + 1)
            # Global max pooling over the sequence length
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # Shape: (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # Shape: (batch_size, num_filters)
            conv_outputs.append(pooled)

        # Concatenate pooled features from all filter sizes
        x = torch.cat(conv_outputs, dim=1)  # Shape: (batch_size, num_filters * len(filter_sizes))

        # Apply fully connected layer and sigmoid activation
        logits = self.fc(x)  # Shape: (batch_size, 1)
        probs = self.sigmoid(logits).squeeze(1)  # Shape: (batch_size)

        return probs

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

class GAN_LSTM(nn.Module):
    def __init__(self, config):
        super(GAN_LSTM, self).__init__() 
        self.config = config
        self.Generator = Generator_LSTM(config)
        self.Discriminator = Discriminator_LSTM(config)
    
class GAN_CNN(nn.Module):
    def __init__(self, config):
        super(GAN_CNN, self).__init__() 
        self.config = config
        self.Generator = Generator_CNN(config)
        self.Discriminator = Discriminator_CNN(config)