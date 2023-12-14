from dataclasses import dataclass, field
from decimal import Decimal, getcontext
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


# @dataclass
# class Dataset:
#     data: list
#     vocab: Vocab
#     onset: bool = False
#     random: np.random.RandomState = np.random.RandomState(0)
#     min_length: int = 2 
#     max_length: int = 5
#     probs: list = field(default_factory=list) # Only used
#     lamb: float = 1.31 # 1.31 is the lambda value of the Poisson distribution that best fits the CMU data (with weighting by freq)

class Dataset:
    def __init__(self, data, vocab, onset=False, random=np.random.RandomState(0), min_length=2, max_length=5, probs=np.array([]), lamb=1.31):
        print("Initializing Dataset...")
        self.data = data
        self.vocab = vocab
        self.onset = onset
        self.random = random
        self.min_length = min_length
        self.max_length = max_length
        self.probs = probs
        self.lamb = lamb
        print("Probs:", len(probs))

    def random_seq(self):
        length = self.random.randint(self.min_length, self.max_length)
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

    def perturb(self, 
            seq, 
            ops=['swap', 'add', 'delete'],
            ):
        assert seq[0] == 0
        assert seq[-1] == 0
        new_seq = list(seq[1:-1])
        # randomly sample the operation type
        op_idx = self.random.randint(0, len(ops))
        op = ops[op_idx]
        # randomly sample an index between 0 and length (np.random.randint is exclusive)
        idx = self.random.randint(0, len(new_seq))
        if op == 'swap':
            new_phoneme = self.random.randint(1, len(self.vocab))
            new_seq[idx] = new_phoneme
        elif op == 'add':
            new_phoneme = self.random.randint(1, len(self.vocab))
            new_seq = new_seq[:idx] + [new_phoneme] + new_seq[idx:]
        elif op == 'delete':
            new_seq = new_seq[:idx] + new_seq[idx+1:] 
        else:
            raise NotImplementedError()
        
        new_seq = (0,) + tuple(new_seq) + (0,)
#        print(f'op:\t{op}')
#        print(f'orig seq:\t{seq}')
#        print(f'new seq:\t{new_seq}')

        if op == 'swap':
            assert len(new_seq) == len(seq)
        elif op == 'add':
            assert len(new_seq) == len(seq) + 1
        elif op == 'delete':
            assert len(new_seq) == len(seq) - 1
        return new_seq

    def random_example(self):
        #return self.data[np.random.randint(len(self.data))]
        return self.data[self.random.randint(len(self.data))]

    def random_seq_poisson(self):
        """ Generate length with Poisson distribution, then generate sequence """
        length = self.random.poisson(self.lamb)
        seq = []
        for i in range(length):
            seq.append(self.random.randint(1, len(self.vocab)))
        if self.onset:
            seq = [0] + seq[:2]
        else:
            seq = [0] + seq + [0]
        return tuple(seq)

    # sample with probs 
    def random_example_by_probs(self):
        return self.data[self.random.choice(len(self.data), p=self.probs)]

def load_cmu(min_length, max_length):
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
    return Dataset(data, vocab, min_length=min_length, max_length=max_length)

def load_lexicon(file_name, min_length, max_length):
    vocab = Vocab()
    data = []
    with open(file_name) as reader:
        for line in reader:
#            # TOOD: this is for only including words with fewerthan 5 phonemes 
#            if len(line.strip().split()) >= 5:
#                continue
            phonemes = [BOUNDARY] + line.strip().split() + [BOUNDARY]

            #if not all(
            #    p in {"B", "T", "M", "L", "IY", "AH", "UW", BOUNDARY}
            #    for p in phonemes
            #):
            #    continue

            data.append(vocab.encode(phonemes, add=True))
    print(f"Loading lexicon with min_length={min_length}, max_length={max_length}...")
    return Dataset(data, vocab, min_length=min_length, max_length=max_length)

def load_cmu_words_and_probs(file_name='data/SmallCMUWithFrequencies_Prepped.txt'):
    # load SmallCMUWithFrequencies_Prepped.txt and create a dictionary
    # with words as keys and frequencies as values
    word_freq_dict = {}
    with open(file_name, 'r') as f:
        for line in f:
            # counts are per million
            word, counts = line.strip().split('\t')
            freq = float(counts) 
            word_freq_dict[word] = freq

    # Need to use Decimal to avoid floating point errors
    total_freq = Decimal(sum(word_freq_dict.values()))
    # print("Total frequency:", total_freq)

    # Go back through and normalize the probabilities
    new_word_freq_dict = {}
    for word in word_freq_dict:
        new_word_freq_dict[word] = float(Decimal(word_freq_dict[word]) / total_freq)

    word_freq_dict = new_word_freq_dict
    # Check that probabilities sum to 1
    probs = list(word_freq_dict.values())
    words = list(word_freq_dict.keys())

    probs_sum = np.sum(probs)
    print("Probs sum:", probs_sum)
    assert np.isclose(probs_sum, 1.0), f"Probabilities do not sum to 1.0: {np.sum(list(word_freq_dict.values()))}"

    return words, probs

def load_lexicon_with_probabilities(min_length, max_length, file_name='data/SmallCMUWithFrequencies_Prepped.txt'):
    vocab = Vocab()
    words, probs = load_cmu_words_and_probs(file_name=file_name)
    data = []
    for word in words:
        word = word.split()
        phonemes = [BOUNDARY] + list(word) + [BOUNDARY]
        data.append(vocab.encode(phonemes, add=True))
    print(f"Loading lexicon with probabilities...")
    print('data:', data)
    print('probs:', probs)
    # probs_np = np.array(probs, dtype=np.float128)
    # assert np.isclose(sum(probs), 1.0), f"Probabilities do not sum to 1.0: {np.sum(probs)}"
    assert np.isclose(np.sum(np.array(probs, dtype=np.float128)), 1.0), f"Probabilities do not sum to 1.0: {np.sum(list(word_freq_dict.values()))}"
    return Dataset(data, vocab, probs=probs, min_length=min_length, max_length=max_length)

def load_cmu_onsets(min_length, max_length):
    vocab = Vocab()
    data = []
    with open("data/hw/words.txt") as reader:
        for line in reader:
            phonemes = [BOUNDARY] + line.strip().split()[:2]
            data.append(vocab.encode(phonemes, add=True))
    return Dataset(data, vocab, onset=True, min_length=min_length, max_length=max_length)


def load_dummy(min_length, max_length):
    vocab = load_cmu().vocab
    data = [
        vocab.encode([BOUNDARY, "P"]),
        vocab.encode([BOUNDARY, "T"]),
        vocab.encode([BOUNDARY, "B"]),
    ]
    return Dataset(data, vocab, onset=True, min_length=min_length, max_length=max_length)
