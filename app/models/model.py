import os
import math
import pickle
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from collections import Counter


class Vocabulary:

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.token2idx = {}
        self.idx2token = {}

    def build(self, formulas):
        counter = Counter()
        for formula in formulas:
            tokens = formula.split()
            counter.update(tokens)

        vocab_tokens = [token for token, freq in counter.items() if freq >= self.min_freq]
        all_tokens = self.special_tokens + sorted(vocab_tokens)

        self.token2idx = {token: idx for idx, token in enumerate(all_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

    def encode(self, formula):
        tokens = formula.split()
        encoded = [self.token2idx["<SOS>"]]
        for token in tokens:
            encoded.append(self.token2idx.get(token, self.token2idx["<UNK>"]))
        encoded.append(self.token2idx["<EOS>"])
        return encoded

    def decode(self, indices):
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, "<UNK>")
            if token == "<EOS>":
                break
            if token not in ["<SOS>", "<PAD>"]:
                tokens.append(token)
        return " ".join(tokens)

    def __len__(self):
        return len(self.token2idx)


class PositionalEncoding(nn.Module):


    def __init__(self, d_model=256, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Encoder(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((4, 16))
        self.projection = nn.Conv2d(1280, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.projection(x)
        x = x.flatten(2)       # (B, C, 64)
        x = x.transpose(1, 2)  # (B, 64, C)
        return x


class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position = PositionalEncoding(embed_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, memory, tgt, tgt_mask=None, tgt_padding_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.position(tgt)
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        out = self.fc(out)
        return out


class FormulaRecognizer(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, dropout=0.1, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx
        )

    def forward(self, images, tgt, tgt_mask=None, tgt_padding_mask=None):
        memory = self.encoder(images)
        return self.decoder(memory, tgt, tgt_mask, tgt_padding_mask)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_vocab_file(vocab_path="weights/vocab.pkl"):
   
    if not os.path.exists(vocab_path):
        if os.path.exists("vocab.pkl"):
            vocab_path = "vocab.pkl"
        else:
            raise FileNotFoundError(f"Vocabulary file not found at '{vocab_path}'.")

    with open(vocab_path, "rb") as f:
        try:
            vocab = CustomUnpickler(f).load()
        except Exception:
            f.seek(0)
            vocab = pickle.load(f)
    return vocab
