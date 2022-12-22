# just how rare are syllables that TELL us that the harmony is progressive and blocked by a?

def is_ti(word):
    telling_sequences = ["iaI",
                        "iaE",
                        "eaI",
                        "eaE",
                        "Iai",
                        "Iae",
                        "Eai",
                        "Eae"]
    word = [char for char in word if char in ["a", "e", "E", "i", 'I']]
    word = "".join(word)
    #print(word)
    verdict = None
    for s in telling_sequences:
        if s in word:
            verdict = True
    if verdict == None:
        return False
    else:
        return verdict


def is_licit(word):
    telling_sequences = ["iI",
                        "iE",
                        "Ii",
                        "Ie",
                        "eI",
                        "eE",
                        "Ei",
                        "Ee"
                        ]
    word = [char for char in word if char in ["a", "e", "E", "i", 'I']]
    word = "".join(word)
    #print(word)
    verdict = None
    for s in telling_sequences:
        if s in word:
            verdict = False
    if verdict == None:
        return True
    else:
        return verdict

# evaluate train set
indat = open("./data/hw/atr_harmony_lexicon.txt","r",encoding='utf8').read().split("\n")
print(indat)

#counter = []
out = open("analysis_of_characteristics_of_train_set.csv","w",encoding="utf8")
out.write("Word,IsTI\n")
for word_full in indat:
    #if is_ti(word_full):
    out.write(word_full+","+str(is_ti(word_full))+'\n')


# make narrow test set
atr_harmony_scores = open("./data/Blicks/TI_test.csv","w",encoding='utf8')
atr_harmony_scores.write("Word\n")
vowels = ['i','a','e','E','I']
for v1 in vowels:
    for v2 in vowels:
        for v3 in vowels:
            word = "q"+v1+" q"+v2+" q"+v3
            if is_ti(word):
                atr_harmony_scores.write(word+'\n')


# evaluate broad test set

indat_broad_Test = open("test_set.csv","r",encoding='utf8').read().split("\n")
print(indat_broad_Test)
o = open("broad_test_set_annotated.csv","w",encoding='utf8')
o.write("Word,IsTI,IsLicit\n")
for word in indat_broad_Test:
    o.write(word+","+str(is_ti(word))+","+str(is_licit(word))+'\n')

