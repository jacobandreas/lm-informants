
segments = ['B',
'CH',
'D',
'DH',
'F',
'G',
'JH',
'K',
'P',
'S',
'SH',
'T',
'TH',
'V',
'Z',
'ZH',
'HH',
'M',
'N',
'NG',
'L',
'R',
'W',
'Y',
'IY',
'IH',
'EH',
'EY',
'AE',
'AA',
'AW',
'AY',
'AO',
'OY',
'OW',
'UH',
'UW',
'ER',
'AH']

o = open("all_words_english.csv",'w',encoding='utf8')

length_one_sylls = []
length_two_sylls = []
length_three_sylls = []

for seg1 in segments:
    length_one_sylls.append(seg1)


for seg1 in segments:
    for seg2 in segments:
        cvcv =seg1+" "+seg2
        length_two_sylls.append(cvcv)

for seg1 in segments:
    for seg2 in segments:
        for seg3 in segments:
            cvcvcv=seg1+" "+seg2+" "+seg3
            length_three_sylls.append(cvcvcv)

print(len(length_one_sylls))
print(len(length_two_sylls))
print(len(length_three_sylls))

for word in length_one_sylls+length_two_sylls+length_three_sylls:
    o.write(word+'\n')