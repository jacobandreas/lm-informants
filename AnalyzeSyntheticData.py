# just how rare are syllables that TELL us that the harmony is progressive and blocked by a?

indat = open("./data/hw/atr_harmony_lexicon.txt","r",encoding='utf8').read().split("\n")
print(indat)

telling_sequences = ["iaI",
                     "iaE",
                     "eaI",
                     "eaE",
                     "Iai",
                     "Iae",
                     "Eai",
                     "Eae"]
#counter = []
out = open("analysis_of_characteristics_of_synthetic_data.csv","w",encoding="utf8")
out.write("TellingSubsequence,WordContainingIt,ProperJudgementForWord\n")
for word_full in indat:
    print()
    word = [char for char in word_full if char in ["a","e","E","i",'I']]
    print(word)
    word = "".join(word)
    print(word)
    for s in telling_sequences:
        if s in word:
            out.write(s+','+word_full+","+"bad\n")
            print('yikes')
            #counter.append((word_full,s))
        else:
            out.write("NONE"+','+word_full+","+"okay\n")

