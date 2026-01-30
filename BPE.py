# bpe.py
import re
import collections


class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # Maps pair -> new_token
        self.ranks = {}  # Maps pair -> priority (lower is better)
        self.token_to_idx = {"PAD": 0, "UNK": 1}
        self.idx_to_token = {0: "PAD", 1: "UNK"}
        self.cache = {}  # Cache for fast encoding

    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        # Escape the pair for regex safely
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        replacement = ''.join(pair)

        for word in v_in:
            w_out = p.sub(replacement, word)
            v_out[w_out] = v_in[word]
        return v_out

    def train(self, data_path):
        """
        Trains BPE on the provided text file.
        """
        print(f"Training BPE with vocab size {self.vocab_size}...")

        word_freqs = collections.defaultdict(int)
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    sentence = parts[1].lower()
                else:
                    sentence = " ".join(line.strip().split()[1:]).lower()

                words = sentence.split()
                for word in words:
                    # Add end-of-word token </w>
                    word_freqs[' '.join(list(word)) + ' </w>'] += 1

        self.vocab = word_freqs

        num_merges = self.vocab_size

        for i in range(num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break

            # Find most frequent pair
            best = max(pairs, key=pairs.get)

            # Record the merge
            self.merges[best] = best[0] + best[1]
            self.ranks[best] = i  # Store rank for fast encoding later

            # Update the vocab
            self.vocab = self.merge_vocab(best, self.vocab)

            # Add to indexer
            new_token = best[0] + best[1]
            if new_token not in self.token_to_idx:
                idx = len(self.token_to_idx)
                self.token_to_idx[new_token] = idx
                self.idx_to_token[idx] = new_token

            if i % 100 == 0:
                print(f"BPE Merge {i}/{num_merges} complete")

        print("BPE Training Complete.")

    def encode_word(self, word):
        """
        Helper: Encodes a single word using learned ranks.
        Uses greedy application of the earliest learned merge.
        """
        if word in self.cache:
            return self.cache[word]

        # Start as characters
        word_split = list(word) + ['</w>']

        while len(word_split) > 1:
            pairs = [(word_split[i], word_split[i + 1]) for i in range(len(word_split) - 1)]

            best_pair = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))

            if best_pair not in self.ranks:
                break

            # Merge that pair in the word_split list
            new_split = []
            i = 0
            while i < len(word_split):
                # Check if we found the pair to merge
                if i < len(word_split) - 1 and (word_split[i], word_split[i + 1]) == best_pair:
                    new_split.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_split.append(word_split[i])
                    i += 1
            word_split = new_split

        # Convert final subwords to indices
        indices = []
        for token in word_split:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                indices.append(self.token_to_idx["UNK"])

        self.cache[word] = indices
        return indices

    def encode(self, sentence):
        """
        Tokenizes a sentence into BPE indices.
        """
        words = sentence.lower().split()
        indices = []
        for word in words:
            indices.extend(self.encode_word(word))
        return indices