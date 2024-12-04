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
import Attention_GAN as Att
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer
from augmented_loss import aug_loss
import torch.autograd as autograd
from utils import save_combined_logs, gen_operators, gen_input, sample
from GANs import GAN, GAN_LSTM, GAN_CNN

def Train(model, config):
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    # Training function
    # Get hyperparameters from config:
    epochs = config['training']['epochs']
    
    # BCE Loss for binary classification
    #criterion = nn.BCELoss()

    # Optimizers for each part of GAN
    optimizer_G = optim.Adam(model.Generator.parameters(), lr = config['training']['learning_rate_gen'], betas = (0, 0.9))
    optimizer_D = optim.Adam(model.Discriminator.parameters(), lr = config['training']['learning_rate_discrim'], betas = (0, 0.9))
    
    # Load data (set up for only single chain proteins currently)
    if config['dataset']['single_chain']:
        real = torch.load('single_chain_protein_sequences.pt')
        real_lengths = torch.tensor(pd.read_excel("single_chain_protein_sequences.xlsx")["Length"].to_numpy())
    
    # Preprocess sequences limiting them to max_sequence_length from config
    mask = real_lengths < config['dataset']['max_sequence_length']

    real = real[mask]
    real_lengths = real_lengths[mask]

    real = real[:, :config['dataset']['max_sequence_length']]

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
    criterion = nn.BCELoss()
    # Training Loop:
    for epoch in range(epochs):
        for batch_data, lengths in dataloader:
            batch_data = batch_data.to(device)

            # Initialize input to gaussian random noise with first dimension containing the sequence length if embed
            # length is specified
            input = gen_input(config, lengths, device)
            
            # Generate synthetic sequences
            gen = model.Generator(input, lengths)

            # Sample fom distribution if ProteinGAN is used otherwise it is done automatically
            gen = sample(config, gen)

            # Do other generated sample manipulations depending on config
            gen = gen_operators(config, gen, lengths)

            if config['discriminator']['reg_flag']== "False":
                reg_flag = False
            else:
                reg_flag = True

            if reg_flag:
                # Regularization for the discriminator
                optimizer_D.zero_grad()
                probs_gen = model.Discriminator(gen.detach())
                embedded_real = model.Discriminator.embedding(batch_data.long())  # (batch_size, seq_length, embedding_dim)
                embedded_real = embedded_real.detach().requires_grad_(True)
                D_real = model.Discriminator(embedded_input=embedded_real)
                gamma = config['discriminator']['r1_gamma']  # Regularization coefficient
                gradients = autograd.grad(
                    outputs=D_real.sum(),
                    inputs=embedded_real,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                gradients = gradients.view(gradients.size(0), -1)  # Flatten per sample
                gradients = torch.clamp(gradients, min=-1.0, max=1.0)
                gradient_penalty = ((gradients.norm(2, dim=1) + 1e-8)** 2).mean()
                gradient_penalty = torch.clamp(gradient_penalty, min = -100.0, max = 100.0)
                R1_reg = (gamma / 2) * gradient_penalty
                # Total Discriminator Loss
                loss_discrim = torch.mean(probs_gen) - torch.mean(D_real)
                loss_discrim_total = loss_discrim + R1_reg
                
                # Backward and optimize
                loss_discrim_total.backward()
            else:
                if config["MAIN"]["TYPE"] == "ProteinGAN":
                    optimizer_D.zero_grad()
                    probs_gen = model.Discriminator(gen.detach())
                    probs_real = model.Discriminator(batch_data)
                    loss_discrim = torch.mean(probs_gen) - torch.mean(probs_real)
                    # Backward and optimize
                    loss_discrim.backward()
                else:
                    optimizer_D.zero_grad()
                    probs_gen = model.Discriminator(gen.detach())
                    probs_gen = torch.clamp(probs_gen, min=-10.0, max=10.0)
                    probs_real = model.Discriminator(batch_data)
                    probs_real = torch.clamp(probs_real, min=-10.0, max=10.0)
                    loss_discrim = criterion(probs_gen, torch.zeros_like(probs_gen)) + criterion(probs_real, torch.ones_like(probs_real))

                    # Backward and optimize
                    loss_discrim.backward()
            
            torch.nn.utils.clip_grad_norm_(model.Discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            # Generator update
            optimizer_G.zero_grad()

            # Generate new sequences
            gen = model.Generator(input, lengths)
            gen = sample(config, gen)
            gen = gen_operators(config, gen, lengths)
            probs_gen = model.Discriminator(gen)
            probs_gen = torch.clamp(probs_gen, min=-10.0, max=10.0)
            # Calculate loss
            if config["MAIN"]["TYPE"] == "ProteinGAN":
                loss_gen = -torch.mean(probs_gen)
                loss_gen = torch.clamp(loss_gen, min= -100.0, max = 100.0)
            else:
                loss_gen = criterion(probs_gen, torch.ones_like(probs_gen))
            
            torch.nn.utils.clip_grad_norm_(model.Generator.parameters(), max_norm=1.0)
            # Add augmented loss
            if bool(config['generator']['similarity_loss']) and epoch % config['generator']['similarity_loss_frequency'] == 0 and epoch != 0:
                loss_gen = loss_gen + config['generator']['aug_lambda']*aug_loss(config, gen, lengths)

            # Backward and optimize
            loss_gen.backward()
            optimizer_G.step()
            
            if reg_flag:
                # Log loss values
                loss_logs.append({'Epoch': epoch + 1,
                'Batch': batch_idx + 1,
                'Discriminator Loss': loss_discrim_total.item(),
                'Generator Loss': loss_gen.item()
                })
                batch_idx += 1
                print(f"Epoch [{epoch+1}/{epochs}], d_loss: {loss_discrim_total.item()}, g_loss: {loss_gen.item()}")
            else:
                # Log loss values
                loss_logs.append({'Epoch': epoch + 1,
                'Batch': batch_idx + 1,
                'Discriminator Loss': loss_discrim.item(),
                'Generator Loss': loss_gen.item()
                })
                batch_idx += 1
                print(f"Epoch [{epoch+1}/{epochs}], d_loss: {loss_discrim.item()}, g_loss: {loss_gen.item()}")
        
        # At the end of each epoch create 5 sampled amino acid sequences of random lengths
        if epoch % 20 == 0:
            with torch.no_grad():
                
                ls = torch.randint(low=30, high=config['dataset']['max_sequence_length'], size=(5,))
                if config['MAIN']['TYPE'] == "ProteinGAN":
                    input = torch.randn((5, config['network']['z_dim']))
                else:
                    input = torch.randn((5, config['generator']['input_size']))
                
                if config['generator']['embed_length']:
                    input[:, 0] = ls
                input = input.to(device)

                gen_sequences = model.Generator(input, ls)
                gen_sequences = sample(config, gen_sequences)
                gen_sequences = gen_operators(config, gen_sequences, ls)
                list_ = []

                for t in range(gen_sequences.shape[0]):
                    seq = gen_sequences[t, :].tolist()
                    list_.append(tokenizer.convert_ids_to_tokens(seq))

                generated_sequences.append(list_)
    
    # Log results
    save_combined_logs(loss_logs, generated_sequences)
    discrim_scores = []
    sequences = []

    with torch.no_grad():
        for t in range(config['training']['final_batches']):
            ls = torch.randint(low=30, high=config['dataset']['max_sequence_length'], size=(config['training']['batch_size'],))
            
            input = gen_input(config, ls, device)
            gen_sequences = model.Generator(input, ls)
            
            gen_sequences = sample(gen_sequences)
            gen_sequences = gen_operators(config, gen_sequences, ls)
            
            probs_gen = model.Discriminator(gen_sequences.detach())
            discrim_scores.extend(probs_gen.cpu().numpy().flatten().tolist())
            
            for t in range(gen_sequences.shape[0]):
                seq = gen_sequences[t, :].tolist()
                seq = tokenizer.convert_ids_to_tokens(seq)
                sequences.append(''.join(seq[:ls[t]]))

    pd.DataFrame({"Discriminator Score":discrim_scores, "Generated Sequence":sequences}).to_excel("Final_Generated_Sequences.xlsx")
            


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Parser for specifying config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to config file")
    args = parser.parse_args()
    config = Config().from_disk(args.c)

    # Define GAN
    if config['MAIN']['TYPE'] == "ProteinGAN":
        gan = Att.GAN(config)
    elif config['MAIN']['TYPE'] == "LSTM":
        gan = GAN_LSTM(config)
    elif config['MAIN']['TYPE'] == "CNN":
        gan = GAN_CNN(config)
    else:
        gan = GAN(config)

    # Train GAN
    Train(gan, config)