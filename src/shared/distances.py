import numpy as np
import Levenshtein
import subprocess
import collections

# TODO: modularize distances

#### Meaning distances
def cdist(v1, v2):
	return 1.0 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def l2(v1, v2) :
	return np.linalg.norm(v1 - v2)


#### Text distances
@functools.lru_cache(maxsize=524288)
def apted(tree1, tree2):
	"""
		Call to apted JAR.
		This shouldn't be called, as it's grossly inefficient.
	"""
	cmd = 'java -jar apted.jar -t %s %s' % (tree1, tree2)
	apted_output = subprocess.check_output(cmd.split())
	score = float(apted_output.decode("utf-8"))
	return score

@functools.lru_cache(maxsize=524288)
def jaccard(seq1, seq2):
	"""
		Compute jaccard index
	"""
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
    #if not union: sys.stderr.write('errored on sequences: %s %s\n' % (str(seq1), str(seq2)) )
    return (1 - (intersection / union)) if union else 0.

@functools.lru_cache(maxsize=524288)
def levenshtein(str1, str2, normalise=False):
	"""
		Compute Levenshtein distance.
		Optionally normalize.
	"""
	tmp = Levenshtein.distance(str1, str2)
	if(normalise) and (len(str1) + len(str2)): tmp /= max(len(str1), len(str2))
	return tmp

@functools.lru_cache(maxsize=524288)
def levenshtein_normalised(str1, str2):
	"""
		Shorthand to compute Levenshtein distance + length normalization
	"""
    return levenshtein(str1, str2, normalise=True)




def word_embedding_levenshtein(seq1, seq2, embeddings, average_distance, r=0.9, normalise=False):
	"""
		Compute Levenshtein w/ embedding similarity-based weigthing
	"""
    x1 = 1 + len(seq1)
    x2 = 1 + len(seq2)

    alpha = r / ((1 - r) * average_distance)

    # Initialisation of the matrix
    d = [] # Using Numpy structures for this is probably not more efficient
    d.append(list(range(x2)))
    for i in range(1, x1):
        d.append([i] * x2)

    # Core of the algorithm
    for i in range(1, x1):
        for j in range(1, x2):
            e1 = seq1[i-1]
            e2 = seq2[j-1]

            if(e1 == e2): c = 0
            else:
                v1 = embeddings[e1]
                v2 = embeddings[e2]

                if((v1 is None) or (v2 is None)): c = 1
                else:
                    dst = np.linalg.norm(v1 - v2) # Distance 2 (or L2 norm of the difference)

                    # Now, we need a function increasing function mapping 0 to 0 and +inf to 1
                    c = 1 - (1 / (1 + (alpha * dst)))

                    #c /= r # If you uncomment this line, the cost of a substitution at distance `average_distance` will be 1 and substitutions might have higher cost, up to 1/r. This might be justified as long as `r` is above 0.5 (otherwise, some substitutions might be more expensive than an insertion followed by a deletion).

            d[i][j] = min(
                (d[(i-1)][j] + 1), # Deletion of seq1[i]
                (d[i][(j-1)] + 1), # Insertion of seq2[j]
                (d[(i-1)][(j-1)] + c) # Substitution from seq1[i] to seq2[j]
            )

    raw = d[-1][-1]

    if(normalise): return (raw / (len(seq1) + len(seq2)))
    return raw



# `weights` is a dictionary of word to weight (between 0 and 1)
# One possibility for computing these weights: for a given token, let f be its document frequency (the proportion of documents in which it appears), then w = 1 - f
def weighted_levenshtein(seq1, seq2, weights, r=0.9, normalise=False):
	"""
		Compute Levenshtein w/ weighting
	"""
    x1 = 1 + len(seq1)
    x2 = 1 + len(seq2)

    alpha = r / ((1 - r) * average_distance)

    # Initialisation of the matrix
    d = [] # Using Numpy structures for this is probably not more efficient
    tmp = 0.0
    first_line = [tmp]
    for e in seq2:
        tmp += weights.get(e, 1)
        first_line.append(tmp)
    d.append(first_line)
    tmp = 0
    for e in seq1:
        tmp += weights.get(e, 1)
        d.append([tmp] * x2)

    # Core of the algorithm
    for i in range(1, x1):
        for j in range(1, x2):
            e1 = seq1[i-1]
            e2 = seq2[j-1]

            w1 = weights.get(e1, 1)
            w2 = weights.get(e2, 1)

            d[i][j] = min(
                (d[(i-1)][j] + w1), # Deletion of seq1[i]
                (d[i][(j-1)] + w2), # Insertion of seq2[j]
                (d[(i-1)][(j-1)] + (int(e1 != e2) * max(w1, w2))) # Substitution from seq1[i] to seq2[j]
			)

    raw = d[-1][-1]

    if(normalise): return (raw / (d[0][-1] + d[-1][0]))
    return raw
