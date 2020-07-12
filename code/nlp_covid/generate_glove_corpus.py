import glob
import json
import argparse
from src.nlp_covid.utils import *

ROOT_DIR = sys.path[1]


def append_abstract_body_text_to_file(all_json):
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print('Processing index: {} of {}'.format(idx, len(all_json)))
        content = FileReader(entry)
        with open('covid_corpus.txt', 'a', encoding='utf-8') as f:
            a = merge_and_clean_words(content.abstract)
            b = merge_and_clean_words(content.body_text)
            f.write('{}\n\n{}\n\n\n\n\n'.format(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cord_dataset_path', type=str)
    args = parser.parse_args()
    # root_path = args.cord_dataset_path
    root_path = ROOT_DIR + '\\glove\\CORD-19-research-challenge_Dataset\\'

    all_json = glob.glob('{}/**/*.json'.format(root_path), recursive=True)
    append_abstract_body_text_to_file(all_json)


if __name__ == "__main__":
    main()
