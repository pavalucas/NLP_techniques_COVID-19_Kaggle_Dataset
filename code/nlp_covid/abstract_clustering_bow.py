from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import glob
import os, sys
from textblob import TextBlob
from code.nlp_covid.utils import *
from sklearn.cluster import KMeans

ROOT_DIR = sys.path[1]
PROJECT_DIR = os.getcwd()


def append_abstract_body_text_to_file(all_json):
    abstract_list = []
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print('Processing index: {} of {}'.format(idx, len(all_json)))
        content = FileReader(entry)
        abstract_list.append(merge_and_clean_words(content.abstract))
    return abstract_list


def get_bow(text_list):
    count_vec = TfidfVectorizer(tokenizer=textblob_tokenizer,
                                stop_words='english',
                                norm='l1',
                                use_idf=True)
    bow_matrix = count_vec.fit_transform(text_list)
    print('Length of vocabulary: {}'.format(len(count_vec.vocabulary_)))
    return bow_matrix, count_vec


def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def show_clusters(bow_matrix, count_vect):
    num_clusters = 4
    km = KMeans(n_clusters=num_clusters)
    km.fit(bow_matrix)
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = count_vect.get_feature_names()
    for i in range(num_clusters):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cord_dataset_path', type=str)
    args = parser.parse_args()
    # root_path = args.cord_dataset_path
    root_path = ROOT_DIR + '\\glove\\CORD-19-research-challenge_Dataset\\'
    print("Path to read ---> " + root_path)
    all_json = glob.glob('{}/**/*.json'.format(root_path), recursive=True)
    abstract_list = append_abstract_body_text_to_file(all_json[:100])
    bow_matrix, count_vec = get_bow(abstract_list)
    show_clusters(bow_matrix, count_vec)


if __name__ == "__main__":
    main()
