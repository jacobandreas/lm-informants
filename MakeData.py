import os
import re
import numpy as np
import random
random.seed(3)

def make_a_syllable(plus_atr_vowels,minus_atr_vowels, a, consonants, atr_status_of_preceding_syllable):
    onset = np.random.choice(consonants,1,replace=True)
    if atr_status_of_preceding_syllable == None:
        vowel = np.random.choice(plus_atr_vowels + minus_atr_vowels + a, 1, replace=True)
        if vowel in plus_atr_vowels:
            atr_status_of_this_syllable = "plus"
        elif vowel in minus_atr_vowels:
            atr_status_of_this_syllable = "minus"
        else:
            atr_status_of_this_syllable = None # this is where the transparency vs. opacity comes in.

    elif atr_status_of_preceding_syllable == "plus":
        vowel = np.random.choice(plus_atr_vowels + a, 1, replace=True)
        if vowel in plus_atr_vowels:
            atr_status_of_this_syllable = "plus"
        elif vowel in minus_atr_vowels:
            atr_status_of_this_syllable = "minus"
        else:
            atr_status_of_this_syllable = None # this is where the transparency vs. opacity comes in.

    elif atr_status_of_preceding_syllable == "minus":
        vowel = np.random.choice(minus_atr_vowels + a, 1, replace=True)
        if vowel in plus_atr_vowels:
            atr_status_of_this_syllable = "plus"
        elif vowel in minus_atr_vowels:
            atr_status_of_this_syllable = "minus"
        else:
            atr_status_of_this_syllable = None  # this is where the transparency vs. opacity comes in.
    else:
        print("yikes, something is wrong!")
        assert False

    syllable = "".join(onset.tolist()+vowel.tolist())

    print("my incoming atr status was",atr_status_of_preceding_syllable,"and i made a syllable",syllable,"and I'm returing an atr status of",atr_status_of_this_syllable)
    return syllable, atr_status_of_this_syllable

def make_a_word(plus_atr_vowels,minus_atr_vowels,a, consonants, length_in_sylls,follow_atr_rules):
    word = ""
    atr_status_of_preceding_syllable = None
    print("starting a new word!")
    for _ in range(length_in_sylls):
        syll, atr_status_of_current_syllable = make_a_syllable(plus_atr_vowels,minus_atr_vowels, a, consonants, atr_status_of_preceding_syllable)
        if follow_atr_rules == "agree":
            atr_status_of_preceding_syllable = atr_status_of_current_syllable
        elif follow_atr_rules == "disagree":
            raise NotImplementedError("disagree implemented in different function")
        else:
            atr_status_of_preceding_syllable = None
        print(atr_status_of_preceding_syllable,syll,atr_status_of_current_syllable)
        word += " "+syll

    return word.strip()


def main(vocab_size, plus_atr_vowels,minus_atr_vowels, a,consonants, λ):

    # make a training set with ATR agreement
    training_lexicon = []
    for _ in range(int(round(vocab_size/10))):
        length_in_sylls = 0
        while length_in_sylls == 0:
            possible_length = int(np.random.poisson(λ, 1))
            length_in_sylls = possible_length

        print(length_in_sylls, type(length_in_sylls))
        follow_atr_rules = "agree" # "agree" gets harmony except across a, "disagree" gets disagreement except across a, None gets no harmony enforced
        word = make_a_word(plus_atr_vowels,minus_atr_vowels,a,consonants,length_in_sylls,follow_atr_rules)
        if word not in training_lexicon and word != None:
            training_lexicon.append(word)

    out_lexicon = open("./data/hw/atr_harmony_lexicon.txt", "w", encoding='utf8')
    #print(lexicon)
    #print(len(lexicon))

    for item in training_lexicon:
        out_lexicon.write(item + '\n')
    out_lexicon.close()

    # make a test set with no ATR agreement
    test_set = []
    for _ in range(vocab_size):
        length_in_sylls = 0
        while length_in_sylls == 0:
            possible_length = int(np.random.poisson(λ, 1))
            length_in_sylls = possible_length

        print(length_in_sylls, type(length_in_sylls))
        follow_atr_rules = None
        word = make_a_word(plus_atr_vowels, minus_atr_vowels, a, consonants, length_in_sylls, follow_atr_rules)
        if word not in test_set and word not in training_lexicon:
            test_set.append(word)
    out_test = open("test_set.csv", "w", encoding='utf8')
    for item in test_set:
        out_test.write(item + '\n')

   
    out_test.close()



# +ATR vowels are i, e; -ATR are ɪ, ɛ; a is 0ATR.
# ATR harmony is left-to-right progressive,
# a is either opaque to or is transparent to harmony
# we could live in a world where a has a pair too, and either is targeted or not, or triggers or not, or is invisible to or not, harmony

vowels = ["i","I","e","E","a"]
plus_atr_vowels = ["i","e"]
minus_atr_vowels = ["I","E"]
a = ["a"]
consonants = ["p","t","k","q"]
λ =  2

vocab_size = 1000



main(vocab_size,plus_atr_vowels,minus_atr_vowels,a,consonants,λ)

o = open("all_sylls.csv",'w',encoding='utf8')

length_one_sylls = []
length_two_sylls = []
length_three_sylls = []

for v1 in vowels:
    cv = "p"+v1
    print(cv)
    length_one_sylls.append(cv)


for v1 in vowels:
    for v2 in vowels:
        cvcv ="p"+v1+" p"+v2
        print(cvcv)
        length_two_sylls.append(cvcv)

for v1 in vowels:
    for v2 in vowels:
        for v3 in vowels:
            cvcvcv="p"+v1+" p"+v2+" p"+v3
            print(cvcvcv)
            length_three_sylls.append(cvcvcv)

print(len(length_one_sylls))
print(len(length_two_sylls))
print(len(length_three_sylls))

for word in length_one_sylls+length_two_sylls+length_three_sylls:
    o.write(word+'\n')