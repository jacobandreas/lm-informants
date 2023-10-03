from util import ngrams

class NGramCountInformant:
    def __init__(self, dataset, order=2):
        self.dataset = dataset
        self.order = order
        self.allowed_transitions = set()
        for seq in dataset.data:
            for ngram in ngrams(seq, order):
                self.allowed_transitions.add(ngram)

    def judge(self, seq):
        cost = 0
        for ngram in ngrams(seq, self.order):
            if ngram not in self.allowed_transitions:
                cost += 1
        return cost == 0

class InteractiveInformant:
    def __init__(self, dataset):
        self.dataset = dataset
    def judge(self, seq):
        print(" ".join(self.dataset.vocab.decode(seq)), "?")
        while True:
            cmd = input()
            if cmd in ("t", "f"):
                break
        return cmd == "t"


class HWInformant:
    def __init__(self, dataset, scorer):
        self.dataset = dataset
        self.set_dataset = set(dataset.data)
        self.scorer = scorer

    def judge(self, seq):
        # check if in lexicon, return True if so
        if seq in self.set_dataset:
            print(f"seq in lexicon, returning True:\t{seq}")
            return True
        
#        print('seq:', seq)
#        print('cost:',self.scorer.cost(seq))
        # Hacky, but atr_harmony feature weights for bad features are 10.0 (/data/hw/atr_harmony_feature_weights.txt), so will always be False for any sequence that has one bad feature and 0 otherwise
        return self.scorer.cost(seq) < 2.53 # this is a hack!
        #return self.scorer.cost(seq) == 0

    def cost(self, seq):
        #return self.scorer.cost(seq) < 3
        return self.scorer.cost(seq)

class DummyInformant:
    def __init__(self, dataset):
        self.dataset = dataset

    def judge(self, seq):
        if seq == self.dataset.data[0]:
            return True
        if seq == self.dataset.data[1]:
            return False
        assert False
