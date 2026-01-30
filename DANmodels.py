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
        
        self.indexed_examples = []
        self.labels = []

        unk_idx = self.indexer.index_of("UNK")
        if unk_idx == -1: unk_idx = 1

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

        if max_len is None:
            self.max_len = max(len(s) for s in self.indexed_examples)
        else:
            self.max_len = max_len

        self.padded_data = []
        for indices in self.indexed_examples:
            if len(indices) > self.max_len:
                indices = indices[:self.max_len]

            padding_len = self.max_len - len(indices)
            padded_indices = indices + [0] * padding_len
            self.padded_data.append(padded_indices)

        self.padded_data = torch.tensor(self.padded_data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.padded_data[idx], self.labels[idx]


class DAN(nn.Module):
    """
    Deep Averaging Network implementation.
    """

    def __init__(self, word_embeddings, hidden_size=100, num_classes=2, dropout=0.3):
        super(DAN, self).__init__()

        self.embeddings = word_embeddings.get_initialized_embedding_layer(frozen=False)
        embedding_dim = self.embeddings.embedding_dim

        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, max_seq_len) containing word indices
        """
        mask = (x != 0).float()

        embeds = self.embeddings(x)

        sum_embeds = torch.sum(embeds * mask.unsqueeze(-1), dim=1)

        counts = mask.sum(dim=1).unsqueeze(-1)
        counts = torch.clamp(counts, min=1.0)
        avg_embeds = sum_embeds / counts

        out = self.fc1(avg_embeds)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)

        logits = self.fc3(out)

        return F.log_softmax(logits, dim=1)


class SubwordEmbeddings:
    """
    A wrapper class to mimic WordEmbeddings but with random initialization.
    This allows us to reuse the DAN model class without changing it.
    """

    def __init__(self, vocab_size, embedding_dim=50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def get_initialized_embedding_layer(self, frozen=False):
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
        return nn.Embedding(self.vocab_size, self.embedding_dim)