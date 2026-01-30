# DANmodels.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples
from BPE import BPE


class SentimentDatasetDAN(Dataset):
    """
    Dataset class for Deep Averaging Network (DAN).
    Converts text examples into padded tensors of word indices.
    """

    def __init__(self, infile, word_embeddings, max_len=None):
        """
        Args:
            infile: Path to the data file.
            word_embeddings: The WordEmbeddings object (from sentiment_data.py).
            max_len: Fixed sequence length for padding. If None, it uses the max length in the dataset.
        """
        self.examples = read_sentiment_examples(infile)
        self.word_embeddings = word_embeddings
        self.indexer = word_embeddings.word_indexer

        # Convert words to indices
        self.indexed_examples = []
        self.labels = []

        # We need to handle UNK (Unknown) and PAD (Padding)
        # Assuming from sentiment_data.py: PAD is index 0, UNK is index 1
        unk_idx = self.indexer.index_of("UNK")
        if unk_idx == -1: unk_idx = 1  # Fallback if not found, though read_word_embeddings adds it

        for ex in self.examples:
            indices = []
            for word in ex.words:
                idx = self.indexer.index_of(word)
                if idx == -1:
                    indices.append(unk_idx)
                else:
                    indices.append(idx)
            self.indexed_examples.append(indices)
            self.labels.append(ex.label)

        # Determine max length for padding
        if max_len is None:
            self.max_len = max(len(s) for s in self.indexed_examples)
        else:
            self.max_len = max_len

        # Pad sequences to max_len
        self.padded_data = []
        for indices in self.indexed_examples:
            # Truncate if too long (rare if max_len is auto-calculated)
            if len(indices) > self.max_len:
                indices = indices[:self.max_len]

            # Pad with 0 (PAD token)
            padding_len = self.max_len - len(indices)
            padded_indices = indices + [0] * padding_len
            self.padded_data.append(padded_indices)

        # Convert to tensors
        self.padded_data = torch.tensor(self.padded_data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the word indices and the label
        return self.padded_data[idx], self.labels[idx]


class DAN(nn.Module):
    """
    Deep Averaging Network implementation.
    """

    def __init__(self, word_embeddings, hidden_size=100, num_classes=2, dropout=0.3):
        super(DAN, self).__init__()

        # 1. Load Pretrained Embeddings
        # get_initialized_embedding_layer returns a torch.nn.Embedding layer
        # frozen=False allows fine-tuning the embeddings during training
        self.embeddings = word_embeddings.get_initialized_embedding_layer(frozen=False)
        embedding_dim = self.embeddings.embedding_dim

        # 2. Feedforward Network
        # Architecture: Average Embedding -> Linear -> ReLU -> Dropout -> Linear -> LogSoftmax
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Additional hidden layer for capacity
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, max_seq_len) containing word indices
        """
        # Create a mask for non-padding tokens (assume index 0 is PAD)
        # Shape: (batch_size, seq_len)
        mask = (x != 0).float()

        # Look up embeddings
        # Shape: (batch_size, seq_len, embedding_dim)
        embeds = self.embeddings(x)

        # Compute the average embedding, ignoring padding
        # Sum embeddings across the sequence length dimension (dim 1)
        # Shape: (batch_size, embedding_dim)
        sum_embeds = torch.sum(embeds * mask.unsqueeze(-1), dim=1)

        # Count non-padding tokens
        # Shape: (batch_size, 1)
        counts = mask.sum(dim=1).unsqueeze(-1)

        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)

        # Calculate Average
        avg_embeds = sum_embeds / counts

        # Pass through feedforward network
        out = self.fc1(avg_embeds)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)

        logits = self.fc3(out)

        # Return log_softmax to be compatible with NLLLoss
        return F.log_softmax(logits, dim=1)


class SubwordEmbeddings:
    """
    A wrapper class to mimic WordEmbeddings but with random initialization.
    This allows us to reuse the DAN model class without changing it.
    """

    def __init__(self, vocab_size, embedding_dim=50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # We don't have an indexer object, but DAN usually just needs the layer.

    def get_initialized_embedding_layer(self, frozen=False):
        # Initialize random embeddings [cite: 96]
        return nn.Embedding(self.vocab_size, self.embedding_dim)


class SentimentDatasetBPE(Dataset):
    """
    Dataset class that uses BPE for tokenization.
    """

    def __init__(self, infile, bpe_tokenizer, max_len=None):
        self.examples = read_sentiment_examples(infile)
        self.bpe = bpe_tokenizer

        self.indexed_examples = []
        self.labels = []

        for ex in self.examples:
            # Use BPE to encode the sentence
            # We reconstruct the sentence from words to pass to BPE
            sentence = " ".join(ex.words)
            indices = self.bpe.encode(sentence)
            self.indexed_examples.append(indices)
            self.labels.append(ex.label)

        if max_len is None:
            self.max_len = max(len(s) for s in self.indexed_examples)
        else:
            self.max_len = max_len

        self.padded_data = []
        for indices in self.indexed_examples:
            if len(indices) > self.max_len:
                indices = indices[:self.max_len]
            padding_len = self.max_len - len(indices)
            padded_indices = indices + [0] * padding_len  # 0 is PAD
            self.padded_data.append(padded_indices)

        self.padded_data = torch.tensor(self.padded_data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.padded_data[idx], self.labels[idx]


class RandomEmbeddings:
    """
    Mimics the WordEmbeddings class but initializes vectors randomly.
    Used for Part 1b.
    """

    def __init__(self, vocab_size, embedding_dim=50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def get_initialized_embedding_layer(self, frozen=False):
        # Return a fresh Embedding layer with random weights
        # frozen is ignored because we MUST train random embeddings
        return nn.Embedding(self.vocab_size, self.embedding_dim)