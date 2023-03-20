from dataclasses import dataclass
import numpy as np

BOUNDARY = "$"

class Vocab:
    def __init__(self):
        self.index = {}
        self.rev_index = {}


    def add(self, token):
        if token in self.index:
            return
        self.index[token] = len(self.index)
        self.rev_index[self.index[token]] = token


    def get(self, token):
        return self.index[token]


    def get_rev(self, token_id):
        return self.rev_index[token_id]


    def __post_init__(self):
        self.rev_index = {v: k for k, v in index.items()}


    def encode(self, tokens, add=False):
        if add:
            for token in tokens:
                self.add(token)
        return tuple(self.index[t] for t in tokens)


    def decode(self, token_ids):
        return tuple(self.rev_index[i] for i in token_ids)


    def __len__(self):
        return len(self.index)


@dataclass
class Dataset:
    data: list
    vocab: Vocab
    onset: bool = False
    random: np.random.RandomState = np.random.RandomState(0)

    def random_seq(self):
        length = self.random.randint(2, 5)
        #length = 2
        seq = []
        for i in range(length):
            seq.append(self.random.randint(1, len(self.vocab)))
            #seq.append(np.random.choice([
            #    self.vocab.get("D"),
            #    self.vocab.get("AA")
            #]))
        if self.onset:
            seq = [0] + seq[:2]
        else:
            seq = [0] + seq + [0]
        return tuple(seq)

    def permute(self, random, seq):
        if self.onset:
            return (0,) + tuple(random.permutation(seq[1:]))
        else:
            return (0,) + tuple(random.permutation(seq[1:-1])) + (0,)

    def random_example(self):
        #return self.data[np.random.randint(len(self.data))]
        return self.data[self.random.randint(len(self.data))]


def load_cmu():
    vocab = Vocab()
    data = []
    with open("data/hw/words.txt") as reader:
        for line in reader:
            phonemes = [BOUNDARY] + line.strip().split() + [BOUNDARY]

            #if not all(
            #    p in {"B", "T", "M", "L", "IY", "AH", "UW", BOUNDARY}
            #    for p in phonemes
            #):
            #    continue

            data.append(vocab.encode(phonemes, add=True))
    return Dataset(data, vocab)

def load_lexicon(file_name):
    vocab = Vocab()
    data = []
    with open(file_name) as reader:
        for line in reader:
            phonemes = [BOUNDARY] + line.strip().split() + [BOUNDARY]

            #if not all(
            #    p in {"B", "T", "M", "L", "IY", "AH", "UW", BOUNDARY}
            #    for p in phonemes
            #):
            #    continue

            data.append(vocab.encode(phonemes, add=True))
    return Dataset(data, vocab)



def load_cmu_onsets():
    vocab = Vocab()
    data = []
    with open("data/hw/words.txt") as reader:
        for line in reader:
            phonemes = [BOUNDARY] + line.strip().split()[:2]
            data.append(vocab.encode(phonemes, add=True))
    return Dataset(data, vocab, onset=True)


def load_dummy():
    vocab = load_cmu().vocab
    data = [
        vocab.encode([BOUNDARY, "P"]),
        vocab.encode([BOUNDARY, "T"]),
        vocab.encode([BOUNDARY, "B"]),
    ]
    return Dataset(data, vocab, onset=True)
