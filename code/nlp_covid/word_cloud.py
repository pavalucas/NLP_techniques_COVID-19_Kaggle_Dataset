import numpy as np
import nltk
from sklearn.manifold import TSNE
import plotly.express as px
import enchant
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def get_data_from_vectors():
    # Loading Vector file
    file = open("vectors_2020_06_01.txt", "r")
    data = file.read()
    rows = data.split("\n")
    data = []
    file.close()

    print("Number of vectors before cleaning: %d" % len(rows))

    # Splitting the rows
    for row in rows:
        split_row = row.split(" ")
        data.append(split_row)
    return data

def main():
    pass

if __name__ == '__main__':
    main()