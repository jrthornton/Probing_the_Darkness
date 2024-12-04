import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math
# ProteinGAN paper based GAN acrchitecture
# See: https://github.com/Biomatter-Designs/ProteinGAN

NUM_AMINO_ACIDS = 25

# self-attention layer
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W

        proj_query = self.query_conv(x).view(batch_size, -1, N)
        proj_key = self.key_conv(x).view(batch_size, -1, N)
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(batch_size, -1, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        out = self.gamma * out + x
        return out
    
class SNConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNConv2d, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)
    
class SNLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SNLinear, self).__init__()
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)

# residual block for discriminator
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ResidualBlock2D, self).__init__()
        padding = (
            ((kernel_size[0] - 1) // 2) * dilation[0],
            ((kernel_size[1] - 1) // 2) * dilation[1],
        )

        self.conv1 = SNConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SNConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                SNConv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.shortcut(residual)
        out += residual
        out = self.activation(out)
        return out

# residual block for generator
class ResidualBlockUp2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ResidualBlockUp2D, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(1, stride[1]), mode='nearest')
        padding = (
            math.ceil(((kernel_size[0] - 1) * dilation[0]) / 2),
            math.ceil(((kernel_size[1] - 1) * dilation[1]) / 2),
        )
        self.conv1 = SNConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SNConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=(1, stride[1]), mode='nearest'),
            SNConv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.upsample(x)
        out = self.activation(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.activation(out)
        return out

def minibatch_stddev_layer(x):
    batch_std = torch.std(x, dim=0, keepdim=True)
    mean_std = batch_std.mean()
    shape = list(x.size())
    shape[1] = 1
    stddev_feature_map = mean_std.expand(shape)
    x = torch.cat([x, stddev_feature_map], dim=1)
    return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.embedding_dim = config['network']['embedding_dim']
        self.vocab_size = config['network']['vocab_size']
        self.seq_length = config['network']['seq_length']

        self.dim = config['network']['df_dim']
        self.kernel_size = (
            config['network']['kernel_height'],
            config['network']['kernel_width'],
        )
        self.dilations = config['network']['dilation_rate']
        self.pooling = config['network']['pooling']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.initial_conv = SNConv2d(self.embedding_dim, self.dim, kernel_size=(1, 1), stride=1)

        layers = []
        hidden_dim = self.dim
        num_blocks = 4
        for layer in range(num_blocks):
            dilation = self.dilations ** max(0, layer - 2)
            stride = (1, 2)  # downsample
            layers.append(
                ResidualBlock2D(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim * stride[1],
                    kernel_size=self.kernel_size,
                    stride=stride,
                    dilation=(1, dilation),
                )
            )
            hidden_dim *= stride[1]
            if layer == 1:
                layers.append(SelfAttention(hidden_dim))
        self.main = nn.Sequential(*layers)

        self.final_conv = SNConv2d(hidden_dim + 1, hidden_dim // 16, kernel_size=1, stride=1)
        self.activation = nn.ReLU()
        self.seq_length_reduced = self.seq_length // (2 ** num_blocks)
        self.fc_in_features = (hidden_dim // 16) * self.seq_length_reduced
        self.fc = SNLinear(self.fc_in_features, 1)

    def forward(self, x=None, embedded_input=None):
        # Nones are for implementing regularization
        # Handled embedded input:
        if embedded_input is not None:
            embedded = embedded_input
        else:
            x = x.long()
            embedded = self.embedding(x)
        if embedded.is_sparse:
            embedded = embedded.to_dense()

        embedded = embedded.permute(0, 2, 1).unsqueeze(2)
        h = self.initial_conv(embedded)
        h = self.main(h)
        h = self.activation(h)
        # minibatch std
        h = minibatch_stddev_layer(h)
        h = self.final_conv(h)
        h = h.view(h.size(0), -1)
        output = self.fc(h)
        return output

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.latent_dim = config['network']['z_dim']
        self.embedding_dim = config['network']['embedding_dim']
        self.vocab_size = config['generator']['len_dict']
        self.seq_length = config['network']['seq_length']
        self.dim = config['network']['gf_dim']
        self.kernel_size = (
            config['network']['kernel_height'],
            config['network']['kernel_width'],
        )
        self.dilations = config['network']['dilation_rate']
        self.pooling = config['network']['pooling']

        self.number_of_layers = 4
        self.starting_dim = self.dim * (2 ** self.number_of_layers)

        seq_length_reduced = int(self.seq_length / (2 ** self.number_of_layers))
        self.initial_fc = SNLinear(self.latent_dim, self.starting_dim * 1 * seq_length_reduced)

        # ResNet Blocks with upsampling
        layers = []
        hidden_dim = self.starting_dim
        for layer in range(self.number_of_layers):
            stride = (1, 2)  # upsample along sequence length
            dilation = self.dilations ** max(0, self.number_of_layers - (layer + 3))
            layers.append(
                ResidualBlockUp2D(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // stride[1],
                    kernel_size=self.kernel_size,
                    stride=stride,
                    dilation=(1, dilation),
                )
            )
            hidden_dim = hidden_dim // stride[1]
            if layer == self.number_of_layers - 2:
                # self-attention layer
                layers.append(SelfAttention(hidden_dim))
        self.main = nn.Sequential(*layers)

        self.final_bn = nn.BatchNorm2d(hidden_dim)
        self.activation = nn.ReLU()
        self.final_conv = SNConv2d(hidden_dim, self.vocab_size, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z, lengths):
        # lengths as input for consistency across GAN models
        h = self.initial_fc(z)
        seq_length_reduced = int(self.seq_length / (2 ** self.number_of_layers))
        h = h.view(-1, self.starting_dim, 1, seq_length_reduced)
        h = self.main(h)
        h = self.activation(self.final_bn(h))
        logits = self.final_conv(h)

        batch_size, vocab_size, _, seq_length = logits.shape
        logits = logits.view(batch_size, vocab_size, seq_length)
        # softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        return probs  # Return probabilities of amino acids at each position

# Define GAN the same as GANs in GANs module
class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        self.Discriminator = Discriminator(config)
        self.Generator = Generator(config)
