if __name__ != "__main__":
    raise ImportError("This standalone script should not be imported.")

import itertools as it
import more_itertools as m_it
import functools as ft
import random
import os

@ft.lru_cache(maxsize=32*5)
def _base_msg(category, hol_cat):
    """
    Given category, produce message.
    """
    if hol_cat <= 1:
        return  [2 * i + int(c) for i,c in enumerate(category)]
    else:
        compositional = category[:-hol_cat]
        holistic = category[-hol_cat:]
        compositional_base = [2 * i + int(c) for i,c in enumerate(compositional)]
        holistic_symbol = [int(''.join(map(str, holistic)), base=2) + (len(compositional_base) * 2)]

        return compositional_base + holistic_symbol

def _pretty(message, synonym_table, end_symbol, intersperse):
    """
    Clean up message.
    """
    try:
        sequence = [
            random.choice(synonym_table[sub])
            for sub in message
        ]
    except Exception:
        print(message)
        import pdb; pdb.set_trace()
    for _ in range(intersperse):
        ungrounded_symbol = [random.choice(range(end_symbol + 1, end_symbol + intersperse + 1))]
        random_position = random.randint(0, len(sequence))
        sequence.insert(random_position, ungrounded_symbol)
    return [symbol for item in sequence for symbol in item]

def generate_synonym_table(num_synonyms, upper_mwe_len, hol_cat):
    """
    Input:
        - `num_synonyms` number of synonyms per category
        - `upper_mwe_len` upper bound for the length of MWE
        - `hol_cat` number of fused features
    Output:
        - `synonym_table`  synonym lookup table
        - `base_alphabet_size` number of symbols used
    """
    if hol_cat <= 1:
        num_concepts = 10
    else:
        num_concepts = (((5 - hol_cat) * 2) + int('1' * hol_cat, base=2)) + 1
    if upper_mwe_len == 1:
        base_alphabet_size = num_concepts * num_synonyms
        items = [(i,) for i in range(base_alphabet_size)]
        random.shuffle(items)
        synonym_table = list(m_it.chunked(items, num_synonyms))
    else:
        base_alphabet_size = num_concepts
        items = (
            it.product(range(base_alphabet_size), repeat=i)
            for i in range(1, upper_mwe_len + 1)
        )
        items = list(it.chain.from_iterable(items))
        random.shuffle(items)
        synonym_table = list(m_it.chunked(items, num_synonyms))[:num_concepts]
    return synonym_table, base_alphabet_size

NUMBER_RUNS = 50
for RUN_NUMBER in range(1, NUMBER_RUNS+1):
    OUTPUT_DIR = 'data/exp1/run-%i' % RUN_NUMBER
    MAX_SWAP = 2
    MAX_SYNONYMS = 3
    MAX_UPPER_MWE_LEN = 3
    MAX_INTERSPERSE = 3
    MAX_HOL_CAT = 5
    MAX_NUM_PARAPHRASES = 3

    #0. Create result dir
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for num_paraphrases in range(1, MAX_NUM_PARAPHRASES + 1):
        #1. Generate random languages
        with open(os.path.join(OUTPUT_DIR, 'par-%i_irregular.tsv' % num_paraphrases), 'w') as ostr:
            idx = 0
            for _ in range(num_paraphrases):
                for category in it.product([0, 1], repeat=5):
                        message = random.choices(range(10), k=random.choice(range(1, 11)))
                        print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                        idx += 1

        with open(os.path.join(OUTPUT_DIR, 'par-%i_irregular-constlen.tsv' % num_paraphrases), 'w') as ostr:
            idx = 0
            for _ in range(num_paraphrases):
                for category in it.product([0, 1], repeat=5):
                        message = random.choices(range(10), k=5)
                        print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                        idx +=1

    for synonyms in range(1, MAX_SYNONYMS + 1):
        max_p = 1 if synonyms == 1 else MAX_NUM_PARAPHRASES

        for hol_cat in range(1, MAX_HOL_CAT + 1):

            for intersperse in range(0, MAX_INTERSPERSE + 1):

                for num_paraphrases in range(1, max_p + 1):

                    for upper_mwe_len in range(1, MAX_UPPER_MWE_LEN + 1):

                        base_name = 'par-%i_syn-%i_mwe-%i_isp-%i_hol-%i_' % (num_paraphrases, synonyms, upper_mwe_len, intersperse, hol_cat)

                        for swaps in range(MAX_SWAP + 1):

                            synonym_table, base_alphabet_size = generate_synonym_table(synonyms, upper_mwe_len, hol_cat)
                            if swaps == 0:
                                # no scramble
                                fname = os.path.join(OUTPUT_DIR, base_name + 'strict-order.tsv')
                                print('writing', fname)
                                distinct_messages = set()
                                with open(fname, 'w') as ostr:
                                    idx = 0
                                    for i in range(num_paraphrases):
                                        for category in it.product([0, 1], repeat=5):
                                            message = _base_msg(category, hol_cat)
                                            message = _pretty(message, synonym_table, base_alphabet_size, intersperse)
                                            val = (tuple(category), tuple(message))
                                            if not val in distinct_messages:
                                                distinct_messages.add(val)
                                                print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                                                idx += 1
                                if hol_cat < 5:
                                    synonym_table, base_alphabet_size = generate_synonym_table(synonyms, upper_mwe_len, hol_cat)
                                    # total random scramble
                                    fname = os.path.join(OUTPUT_DIR, base_name + 'random-order.tsv')
                                    print('writing', fname)
                                    distinct_messages = set()
                                    with open(fname, 'w') as ostr:
                                        idx = 0
                                        for i in range(num_paraphrases):
                                            for category in it.product([0, 1], repeat=5):
                                                message = _base_msg(category, hol_cat)
                                                random.shuffle(message)
                                                message = _pretty(message, synonym_table, base_alphabet_size, intersperse)
                                                val = tuple(category), tuple(message)
                                                if not val in distinct_messages:
                                                    distinct_messages.add(val)
                                                    print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                                                    idx += 1

                            # random inversion of `swap` pairs
                            elif hol_cat < MAX_HOL_CAT:
                                fname = os.path.join(OUTPUT_DIR, base_name + 'swap-%i.tsv' % swaps)
                                print('writing', fname)
                                distinct_messages = set()
                                with open(fname, 'w') as ostr:
                                    idx = 0
                                    for i in range(num_paraphrases):
                                        for category in it.product([0, 1], repeat=5):
                                            message = _base_msg(category, hol_cat)
                                            for _ in range(swaps):
                                                i1, i2 = random.sample(range(len(message)), 2)
                                                message[i1], message[i2] = message[i2], message[i1]
                                            message = _pretty(message, synonym_table, base_alphabet_size, intersperse)
                                            val = tuple(category), tuple(message)
                                            if not val in distinct_messages:
                                                distinct_messages.add(val)
                                                print(idx, ' '.join(map(str, category)), ' '.join(map(str, message)), sep="\t", file=ostr)
                                                idx += 1
