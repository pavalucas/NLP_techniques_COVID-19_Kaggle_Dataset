from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import glob
import os, sys
from textblob import TextBlob
from nlp_covid.utils import *
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE


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


def plot_clusters_distortion(bow_matrix):
    # run kmeans with many different k
    pca = PCA(n_components=0.95, random_state=42)
    X_reduced = pca.fit_transform(bow_matrix.toarray())
    print(bow_matrix)
    print()
    print()
    print()
    print(X_reduced)
    distortions = []
    range_num_clusters = range(2, 30)
    for k in range_num_clusters:
        k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
        k_means.fit(X_reduced)
        distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / bow_matrix.shape[0])
    X_line = [range_num_clusters[0], range_num_clusters[-1]]
    Y_line = [distortions[0], distortions[-1]]

    # Plot the elbow
    plt.plot(range_num_clusters, distortions, 'b-')
    plt.plot(X_line, Y_line, 'r')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def show_clusters(bow_matrix, count_vect, num_clusters):
    km = KMeans(n_clusters=num_clusters)
    km.fit(bow_matrix)
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = count_vect.get_feature_names()
    for i in range(num_clusters):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))


def plot_clusters_using_tsne(bow_matrix, num_clusters):
    # use PCA to remove some noise/outliers from the data and make the clustering problem easier for k-means
    pca = PCA(n_components=0.95, random_state=42)
    X_reduced = pca.fit_transform(bow_matrix.todense())

    # cluster the papers abstracts
    km = KMeans(n_clusters=num_clusters)
    y_pred = km.fit_predict(X_reduced)

    # use TSNE to reduce to 2 dimensions
    tsne = TSNE(verbose=1, perplexity=100, random_state=42)
    X_embedded = tsne.fit_transform(bow_matrix.todense())

    # sns settings
    sns.set(rc={'figure.figsize': (15, 15)})

    # colors
    palette = sns.hls_palette(num_clusters, l=.4, s=.9)

    # plot
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y_pred, legend='full', palette=palette)
    plt.title('t-SNE with Kmeans Labels')
    plt.savefig("improved_cluster_tsne.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cord_dataset_path', type=str)
    args = parser.parse_args()
    #root_path = args.cord_dataset_path
    #root_path = '/media/sf_SharedVM/dataset'
    root_path = 'C:\\Users\\amondejar\\Desktop\\SharedVM\\dataset'
    all_json = glob.glob('{}/**/*.json'.format(root_path), recursive=True)
    # Total papers: 139694
    abstract_list = append_abstract_body_text_to_file(all_json[:100])
    print(abstract_list)
    bow_matrix, count_vec = get_bow(abstract_list)
    # plot_clusters_distortion(bow_matrix)
    # show_clusters(bow_matrix, count_vec, 15)
    plot_clusters_using_tsne(bow_matrix, 15)

if __name__ == "__main__":
    main()
