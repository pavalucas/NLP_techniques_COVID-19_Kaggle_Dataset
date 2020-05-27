# Notes for extension of script:
# 	- User readline() to interactively search for word groups
# 	- On a word miss, use L2 or cosine distance to select the nearest word vector
# 		- This would require all 6B tokens to loaded in ram (but not clustered)
#		- Or use levenshtein distance assuming the word is spelled the same.
#   - Provide an interface to perform basic arithmetic on words (king - man + woman = queen)
# Look at this result from 2014 English Wikipedia:
# 'islamic', 'militant', 'islam', 'radical', 'extremists', 'islamist', 'extremist', 'outlawed'
# 'war' - 'violence' + 'peace' = 'treaty' | 300d

from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import numpy as np
import os, sys, codecs, argparse, pprint, time
from utils import *


def find_word_clusters(labels_array, cluster_labels):
    cluster_to_words = {}
    for index, cluster_num in enumerate(cluster_labels):
        cluster_to_words.setdefault(cluster_num, []).append(labels_array[index])
    return cluster_to_words


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_dim', '-d',
                        type=int,
                        choices=[50, 100, 200, 300],
                        default=50,
                        help='What vector GloVe vector dimension to use '
                             '(default: 50).')
    parser.add_argument('--num_words', '-n',
                        type=int,
                        default=10000,
                        help='The number of lines to read from the GloVe '
                             'vector file (default: 10000).')
    parser.add_argument('--num_clusters', '-k',
                        default=1000,
                        type=int,
                        help='Number of resulting word clusters. '
                             'The number of K in K-Means (default: 1000).')
    parser.add_argument('--n_jobs', '-j',
                        type=int,
                        default=-1,
                        help='Number of cores to use when fitting K-Means. '
                             '-1 = all cores. '
                             'More cores = less time, more memory (default: -1).')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    filename = path = 'data/{}'.format(get_cache_filename_from_args(args))
    cluster_to_words = None
    start_time = time.time()

    vector_file = "vectors.txt"
    df, labels_array = build_word_vector_matrix(vector_file, args.num_words)

    kmeans_model = KMeans(init='k-means++', n_clusters=args.num_clusters, n_jobs=args.n_jobs, n_init=10)
    kmeans_model.fit(df)

    cluster_labels = kmeans_model.labels_
    # cluster_inertia = kmeans_model.inertia_
    cluster_words_dict = find_word_clusters(labels_array, cluster_labels)
    cluster_to_words = list(cluster_words_dict.values())

    # cache these clustering results
    save_json(path, cluster_to_words)
    print('Saved {} clusters to {}. Cached for later use.'.format(len(cluster_to_words), path))

    for i, words in enumerate(cluster_to_words):
        print('CLUSTER {}: {}'.format(i + 1, ', '.join(words)))

    if start_time is not None:
        print("--- {:.2f} seconds ---".format((time.time() - start_time)))
