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
from nlp_covid.utils import *

VECTOR_FILE = 'vectors.txt'
VOCAB_FILE = 'vocab.txt'
TOP_N = 20

SYMPTOMS = ['weight loss', 'chills', 'shivering', 'convulsions', 'deformity', 'discharge', 'dizziness', 'vertigo',
            'fatigue','malaise','asthenia','hypothermia','jaundice','muscle weakness','pyrexia','sweats',
            'swelling','swollen','painful lymph node','weight gain','arrhythmia','bradycardia','chest pain',
            'claudication','palpitations','tachycardia','dry mouth','epistaxis','halitosis','hearing loss',
            'nasal discharge','otalgia','otorrhea','sore throat','toothache','tinnitus', 'trismus', 'abdominal pain',
            'fever', 'bloating', 'belching', 'bleeding', 'blood in stool', 'melena', 'hematochezia', 'constipation',
            'diarrhea', 'dysphagia', 'dyspepsia', 'fecal incontinence', 'flatulence', 'heartburn', 'nausea', 'odynophagia',
            'proctalgia fugax', 'pyrosis', 'steatorrhea', 'vomiting', 'alopecia', 'hirsutism', 'hypertrichosis', 'abrasion',
            'anasarca', 'bleeding into the skin', 'petechia', 'purpura', 'ecchymosis and bruising', 'blister', 'edema',
            'itching', 'laceration', 'rash', 'urticaria', 'abnormal posturing', 'acalculia', 'agnosia', 'alexia', 'amnesia',
            'anomia', 'anosognosia', 'aphasia and apraxia', 'apraxia', 'ataxia', 'cataplexy', 'confusion', 'dysarthria',
            'dysdiadochokinesia', 'dysgraphia', 'hallucination', 'headache', 'akinesia', 'bradykinesia', 'akathisia',
            'athetosis', 'ballismus', 'blepharospasm', 'chorea', 'dystonia', 'fasciculation', 'muscle cramps', 'myoclonus',
            'opsoclonus', 'tic', 'tremor', 'flapping tremor', 'insomnia', 'loss of consciousness', 'syncope',
            'neck stiffness', 'opisthotonus', 'paralysis and paresis', 'paresthesia', 'prosopagnosia', 'somnolence',
            'abnormal vaginal bleeding', 'vaginal bleeding in early pregnancy', 'miscarriage',
            'vaginal bleeding in late pregnancy', 'amenorrhea', 'infertility', 'painful intercourse', 'pelvic pain',
            'vaginal discharge', 'amaurosis fugax', 'amaurosis', 'blurred vision', 'double vision', 'exophthalmos',
            'mydriasis', 'miosis', 'nystagmus', 'amusia', 'anhedonia', 'anxiety', 'apathy', 'confabulation', 'depression',
            'delusion', 'euphoria', 'homicidal ideation', 'irritability', 'mania', 'paranoid ideation', 'suicidal ideation',
            'apnea', 'hypopnea', 'cough', 'dyspnea', 'bradypnea', 'tachypnea', 'orthopnea', 'platypnea', 'trepopnea',
            'hemoptysis', 'pleuritic chest pain', 'sputum production', 'arthralgia', 'back pain', 'sciatica', 'Urologic',
            'dysuria', 'hematospermia', 'hematuria', 'impotence', 'polyuria', 'retrograde ejaculation', 'strangury',
            'urethral discharge', 'urinary frequency', 'urinary incontinence','urinary retention']


def check_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_term_count():
    result = {}
    with open(VOCAB_FILE, 'r') as f:
        for line in f:
            line_list = line.split()
            result[line_list[0]] = int(line_list[1])
    return result


def find_word_clusters(labels_array, cluster_labels):
    cluster_to_words = {}
    for index, cluster_num in enumerate(cluster_labels):
        if labels_array[index] in SYMPTOMS:
            print('Word %s -> Cluster %d ' % (labels_array[index], cluster_num))
        cluster_to_words.setdefault(cluster_num, []).append(labels_array[index])
    return cluster_to_words


def get_vectors_from_symptom_cluster(label_vector_dict, cluster_to_words):
    # get cluster for each symptom
    symptom_to_cluster = {}
    for index, cluster in enumerate(cluster_to_words):
        for word in cluster:
            if word in SYMPTOMS:
                return



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

def main():
    args = parse_args()

    filename = path = 'data/{}'.format(get_cache_filename_from_args(args))
    start_time = time.time()

    # get count for each term
    term_count = get_term_count()
    term_count_filtered = dict(filter(lambda elem: elem[1] >= 50 and not check_int(elem[0]), term_count.items()))

    df, labels_array, label_vector_dict = build_word_vector_matrix(VECTOR_FILE, term_count_filtered, args.num_words)

    kmeans_model = KMeans(init='k-means++', n_clusters=args.num_clusters, n_jobs=args.n_jobs, n_init=10)
    kmeans_model.fit(df)

    cluster_labels = kmeans_model.labels_
    # cluster_inertia = kmeans_model.inertia_
    cluster_words_dict = find_word_clusters(labels_array, cluster_labels)
    cluster_to_words = list(cluster_words_dict.values())

    # for each cluster sort its terms descending using term_count
    cluster_to_words_count = []
    for cluster in cluster_to_words:
        words = [(term_count[word], word) for word in cluster]
        cluster_to_words_count.append(words)
    cluster_to_words_count = [sorted(cluster, reverse=True) for cluster in cluster_to_words_count]

    # get only TOP_N for each cluster
    cluster_to_words_updated = []
    for cluster in cluster_to_words_count:
        words = [term_tuple[1] for term_tuple in cluster[:TOP_N]]
        cluster_to_words_updated.append(words)

    # cluster_to_words_updated = cluster_to_words

    # cache these clustering results
    save_json(path, cluster_to_words_updated)
    print('Saved {} clusters to {}. Cached for later use.'.format(len(cluster_to_words_updated), path))

    for i, words in enumerate(cluster_to_words_updated):
        print('CLUSTER {}: {}'.format(i + 1, ', '.join(words)))

    if start_time is not None:
        print("--- {:.2f} seconds ---".format((time.time() - start_time)))


if __name__ == '__main__':
    main()
