import random
import re

import numpy as np

def replace_random_capitals(input_string):
    result = ""
    for char in input_string:
        if char.isupper() and random.random() < 0.5 and char != "A":
            result += char.lower()
        else:
            result += char
    return result

def generate_symbol_list(random_recombination=False, num_examples=100, list_to_exclude=[]):
    symbols = ['A', 'F', 'G', 'H', 'I']
    symbol_lists = []

    while len(symbol_lists) < num_examples:
        symbol_list = []
        length = np.random.poisson(2)
        for _ in range(length) if length > 0 else range(1):
            if symbol_list and not random_recombination:
                last_symbol = symbol_list[-1]
                if last_symbol in ['F', 'G', 'H', 'I']:
                    next_symbol = random.choice([last_symbol, 'A'])
                else:
                    next_symbol = random.choice(symbols)
            else:
                next_symbol = random.choice(symbols)

            symbol_list.append(next_symbol)
        word = " ".join(symbol_list)
        word_modified = replace_random_capitals(word)
        if word_modified not in symbol_lists and word_modified not in list_to_exclude:
            symbol_lists.append(word_modified)


    return symbol_lists


def write_symbol_lists_to_files():

    lists_1 = generate_symbol_list(num_examples=500)

    lists_2 = generate_symbol_list(random_recombination=True, num_examples=10000, list_to_exclude=lists_1)

    # Write the generated symbol lists to separate files
    with open("symbol_lists_following_rules.txt", "w") as file1:
        for item in lists_1:
            file1.write(f"{item}\n")

    with open("symbol_lists_randomly_generated.txt", "w") as file2:
        ok_words = []
        bad_words = []
        bad_structures = [
            "FG",
            "GF",
            "FH",
            "HF",
            "FI",
            "IF",
            "GH",
            "HG",
            "GI",
            "IG",
            "HI",
            "IH"
        ]
        for item in lists_2:
            print("item", item)
            reduced_item = item.replace(" ", "").upper()
            bad_yet = False
            for bad_structure in bad_structures:
                if bad_structure in reduced_item:
                    bad_yet = True
            if bad_yet == True:
                bad_words.append(item)
                print("bad_word:", item)
            else:
                ok_words.append(item)
                print("good_word:", item)

        num_items_in_test = min(len(ok_words), len(bad_words))
        print("num_items_in_test", num_items_in_test)
        ok_words = random.sample(ok_words, num_items_in_test)
        bad_words = random.sample(bad_words, num_items_in_test)
        for item in ok_words:
           # print("item", item)
            file2.write(f"{item}\n")
        for item in bad_words:
            file2.write(f"{item}\n")
        # for item in lists_2:
        #     file2.write(f"{item}\n")


    print("Symbol lists have been written to files.")


# Call the function to write symbol lists to files
write_symbol_lists_to_files()
