import itertools as it
import functools as ft
import csv
import collections
import random
import pathlib

import Levenshtein
from scipy.stats import spearmanr
import scipy
import numpy as np
import tqdm
import multiprocessing as mp
from shared.MantelTest.Mantel import test as mantel_test


def read_csv(tsv_filename, string_msgs=False):
    """
    Open a message TSV file, and return messages paired with categories
    """
    with open(tsv_filename) as istr:
        data = list(csv.reader(istr, delimiter="\t"))

    _, categories, messages = zip(*data)

    if string_msgs:
        c2i = collections.defaultdict(it.count().__next__)
        messages = map(str.strip, messages)
        messages = [tuple(map(c2i.__getitem__, msg)) for msg in messages]
    else:
        messages = map(str.strip, messages)
        messages = map(str.split, messages)
        messages = [tuple(map(int, msg)) for msg in messages]

    categories = map(str.strip, categories)
    categories = map(str.split, categories)
    categories = [tuple(map(int, ctg)) for ctg in categories]

    return messages, categories


@ft.lru_cache(maxsize=32768)
def hamming(str1, str2):
    return Levenshtein.hamming(str1, str2)


@ft.lru_cache(maxsize=32768)
def levenshtein(str1, str2, normalise=False):
    tmp = Levenshtein.distance(str1, str2)
    if(normalise): tmp /= (len(str1) + len(str2))

    return tmp


@ft.lru_cache(maxsize=32768)
def levenshtein_normalised(str1, str2):
    return levenshtein(str1, str2, normalise=True)


@ft.lru_cache(maxsize=32768)
def jaccard(seq1, seq2):
    union = len(seq1)
    intersection = 0
    d = collections.defaultdict(int)
    for i in seq1:
        d[i] += 1
    for i in seq2:
        x = d[i]
        if(x > 0):
            d[i] -= 1
            intersection += 1
        else:
            union += 1
    return 1 - (intersection / union)

"""
@ft.lru_cache(maxsize=32768)
def jaccard2(seq1, seq2):
    proto_union = len(seq1) + len(seq2)
    intersection = 0
    d = collections.defaultdict(int)
    for i in seq1:
        d[i] += 1
    for i in seq2:
        x = d[i]
        if(x > 0):
            d[i] -= 1
            intersection += 1
    return 1 - (intersection / (proto_union - intersection))
"""

def compute_correlation(messages, categories, message_distance=levenshtein, meaning_distance=hamming, map_msg_to_str=True, map_ctg_to_str=True):
    """
    Compute correlation of message distance and meaning distance.
    """

    # Some distance functions are defined over strings only
    if map_msg_to_str:
        messages = [''.join(map(chr, msg)) for msg in messages]
    if map_ctg_to_str:
        categories = [''.join(map(chr, ctg)) for ctg in categories]

    # Compute pairwise distances
    messages = list(it.starmap(message_distance, it.combinations(messages, 2)))
    categories = list(it.starmap(meaning_distance, it.combinations(categories, 2)))

    return spearman(categories, messages)

def mantel(messages, categories, message_distance=levenshtein, meaning_distance=hamming, perms=1000, method='pearson', map_msg_to_str=True, map_ctg_to_str=True):

    if map_msg_to_str:
        messages = [''.join(map(chr, msg)) for msg in messages]
    if map_ctg_to_str:
        categories = [''.join(map(chr, ctg)) for ctg in categories]

    assert len(messages) == len(categories)
    tM = np.array(list(it.starmap(message_distance, it.combinations(messages, 2))))
    sM = np.array(list(it.starmap(meaning_distance, it.combinations(categories, 2))))
    return mantel_test(tM, sM, method=method, perms=perms)

def process_file(input_file):
    messages, categories = read_csv(input_file)

    m_l = mantel(messages, categories)
    m_ln = mantel(messages, categories, message_distance=levenshtein_normalised)
    m_j = mantel(messages, categories, message_distance=jaccard, map_msg_to_str=False)

    return input_file, m_l, m_ln, m_j


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("compute distances and mantels for artifical languages")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    files = list(pathlib.Path(args.input_dir).glob("**/*.tsv"))
    output_file = args.output_file
    with open(output_file, "w") as ostr, mp.Pool(mp.cpu_count()) as pool:
        calls = pool.imap_unordered(process_file, files)
        for input_file, m_l, m_ln, m_j in tqdm.tqdm(calls):
            print(input_file.name, 'levenshtein', *m_l, 'levenshtein normalized', *m_ln, 'jaccard', *m_j, file=ostr)
