import pandas as pd
import torch

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

def gen_operators(config, gen, lengths):
    gen += config['generator']['offset']
    if config['generator']['force_pad']:
        for t in range(gen.shape[0]):
            gen[t, lengths[t]:] = torch.zeros(gen[t, lengths[t]:].shape[0])
    return gen

def gen_input(config, lengths, device):
    if config['MAIN']['TYPE'] == "ProteinGAN":
        input = torch.randn((config['training']['batch_size'], config['network']['z_dim']))
    else:
        input = torch.randn((config['training']['batch_size'], config['generator']['input_size']))
    
    if config['generator']['embed_length']:
        input[:, 0] = lengths
    input = input.to(device)
    return input

def sample(config, gen):
    if config['MAIN']['TYPE'] == 'ProteinGAN':
        gen = torch.distributions.Categorical(gen.permute(0, 2, 1)).sample()
    return gen