from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import glob
import os, sys
from textblob import TextBlob
from nlp_covid.utils import *
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from yellowbrick.text import TSNEVisualizer, UMAPVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas as pd
import matplotlib.cm as cm


# from matplotlib.axes._axes import _log as matplotlib_axes_logger
# matplotlib_axes_logger.setLevel('ERROR')


# pd.set_option("display.max_columns", 100)

# ROOT_DIR = sys.path[1]
# PROJECT_DIR = os.getcwd()


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


def show_clusters(bow_matrix, count_vect, num_clusters):
    cluster_list = []

    km = KMeans(n_clusters=num_clusters)
    km.fit(bow_matrix)
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = count_vect.get_feature_names()
    for i in range(num_clusters):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        str1 = ' '.join(top_ten_words)
        cluster_list.append(str1)

    return cluster_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cord_dataset_path', type=str)
    args = parser.parse_args()
    # root_path = args.cord_dataset_path
    # root_path = '/media/sf_SharedVM/dataset'
    root_path = 'C:\\Users\\amondejar\\Desktop\\SharedVM\\dataset'
    all_json = glob.glob('{}/**/*.json'.format(root_path), recursive=True)
    # Total papers: 139694
    abstract_list = append_abstract_body_text_to_file(all_json[:1000])
    print(abstract_list)

    bow_matrix, count_vec = get_bow(abstract_list)

    num_clusters = 100

    cluster_lst = show_clusters(bow_matrix, count_vec, num_clusters)

    x_value = 'cancer'
    y_value = 'pneumonia'
    # visualize_kmeans_scikit(cluster_lst, x_value, y_value)

    visualize_pca_tsne(cluster_lst)

    corpus_target = ['fever', 'back_pain', 'tachycardia', 'diarrhea']

    visualize_yellowbrick('t-sne', 'count', cluster_lst, corpus_target)

    visualize_yellowbrick('t-sne', 'tfidf', cluster_lst, corpus_target)

    visualize_yellowbrick('umap', 'count', cluster_lst, corpus_target)

    visualize_yellowbrick('umap', 'tfidf', cluster_lst, corpus_target)


def visualize_kmeans_scikit(texts_list, x_value, y_value):
    # http://jonathansoma.com/lede/algorithms-2017/classes/clustering/k-means-clustering-with-scikit-learn/

    texts = [
        'Penny bought bright blue fishes.',
        'Penny bought bright blue and orange bowl.',
        'The cat ate a fish at the store.',
        'Penny went to the store. Penny ate a bug. Penny saw a fish.',
        'It meowed once at the bug, it is still meowing at the bug and the fish',
        'The cat is at the fish store. The cat is orange. The cat is meowing at the fish.',
        'Penny is a fish.',
        'Penny Penny she loves fishes Penny Penny is no cat.',
        'The store is closed now.',
        'How old is that tree?',
        'I do not eat fish I do not eat cats I only eat bugs.'
    ]

    vec = TfidfVectorizer(tokenizer=textblob_tokenizer,
                          stop_words='english',
                          use_idf=True,
                          max_features=2)
    matrix = vec.fit_transform(texts_list)
    df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())
    print(df)

    colors = cm.rainbow(np.linspace(0, 1, 4))
    ax = df.plot(kind='scatter', x=x_value, y=y_value, alpha=0.2, s=200, color=[colors[1]])

    ax.set_xlabel(x_value.upper())
    ax.set_ylabel(y_value.upper())


# ********************************************************************************************************************

def visualize_pca_tsne(texts_list):
    # Use PCA and t-SNE from scikit-learn.
    # PCA is one approach. For TF-IDF I have also used Scikit Learn's manifold package for non-linear dimension
    # reduction. One thing that I find helpful is to label my points based on the TF-IDF scores.
    num_clusters = 10
    num_seeds = 10
    max_iterations = 300
    labels_color_map = {
        0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
        5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
    }
    pca_num_components = 2
    tsne_num_components = 2

    # texts_list = some array of strings for which TF-IDF is being computed

    # calculate tf-idf of texts
    tf_idf_vectorizer = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, ngram_range=(2, 3))
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(texts_list)

    # create k-means model with custom config
    clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
        precompute_distances="auto",
        n_jobs=-1
    )

    labels = clustering_model.fit_predict(tf_idf_matrix)
    # print labels

    X = tf_idf_matrix.todense()

    # ----------------------------------------------------------------------------------------------------------------------

    reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
    print(reduced_data)

    fig, ax = plt.subplots()
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)
    plt.show()

    # t-SNE plot
    embeddings = TSNE(n_components=tsne_num_components)
    Y = embeddings.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
    plt.show()


# ********************************************************************************************************************


def visualize_yellowbrick(dim_reduction, encoding, corpus_data, corpus_target, labels=True, alpha=0.7, metric=None):
    # https://pypi.org/project/yellowbrick/
    # https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples
    # https://medium.com/@sangarshananveera/rapid-text-visualization-with-yellowbrick-51d3499c9333

    if 'tfidf' in encoding.lower():
        encode = TfidfVectorizer()
    if 'count' in encoding.lower():
        encode = CountVectorizer()
    docs = encode.fit_transform(corpus_data)
    if labels is True:
        labels = corpus_target
    else:
        labels = None
    if 'umap' in dim_reduction.lower():
        if metric is None:
            viz = UMAPVisualizer()
        else:
            viz = UMAPVisualizer(metric=metric)
    if 't-sne' in dim_reduction.lower():
        viz = TSNEVisualizer(alpha=alpha)

    viz.fit(docs, labels)

    return viz.poof()


if __name__ == "__main__":
    main()
