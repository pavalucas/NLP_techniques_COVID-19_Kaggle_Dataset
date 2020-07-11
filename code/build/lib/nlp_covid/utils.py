import codecs
import json
import numpy as np


class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            if 'abstract' in content:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            # Body text
            try:
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
            except:
                print("Paper id: {} -> Body text not found".format(self.paper_id))
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

    def __repr__(self):
        return '{}: {}... {}...'.format(self.paper_id, self.abstract[:200], self.body_text[:200])


def merge_and_clean_words(text):
    symptoms = ['weight loss', 'muscle weakness', 'painful lymph node', 'weight gain', 'chest pain', 'dry mouth',
                'hearing loss', 'nasal discharge', 'sore throat', 'abdominal pain', 'blood in stool',
                'fecal incontinence', 'proctalgia fugax', 'bleeding into the skin', 'ecchymosis and bruising',
                'abnormal posturing', 'aphasia and apraxia', 'muscle cramps', 'flapping tremor',
                'loss of consciousness', 'neck stiffness', 'paralysis and paresis', 'abnormal vaginal bleeding',
                'vaginal bleeding in early pregnancy', 'vaginal bleeding in late pregnancy', 'painful intercourse',
                'pelvic pain', 'vaginal discharge', 'amaurosis fugax', 'blurred vision', 'double vision',
                'homicidal ideation', 'paranoid ideation', 'suicidal ideation', 'pleuritic chest pain',
                'sputum production', 'back pain', 'retrograde ejaculation', 'urethral discharge',
                'urinary frequency', 'urinary incontinence', 'urinary retention']
    merged_symptoms = [x.replace(" ", "_") for x in symptoms]
    merged_text = text.lower()
    # for punctuation in string.punctuation:
    #     merged_text = merged_text.replace(punctuation, " ")
    for i in range(len(symptoms)):
        if symptoms[i] in merged_text:
            merged_text = merged_text.replace(symptoms[i], merged_symptoms[i])
    return merged_text


def build_word_vector_matrix(vector_file, term_count, n_words):
    """Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays"""
    label_vector_dict = {}
    np_arrays = []
    labels_array = []

    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            sr = line.split()
            label = sr[0]
            vector = np.array([float(j) for j in sr[1:]])

            if label in term_count:
                label_vector_dict[label] = vector
                labels_array.append(label)
                np_arrays.append(vector)
            if len(labels_array) == n_words:
                return np.array(np_arrays), labels_array, label_vector_dict
        return np.array(np_arrays), labels_array, label_vector_dict


def get_cache_filename_from_args(args):
    a = (args.vector_dim, args.num_words, args.num_clusters)
    return '{}D_{}-words_{}-clusters.json'.format(*a)


def get_label_dictionaries(labels_array):
    id_to_word = dict(zip(range(len(labels_array)), labels_array))
    word_to_id = dict((v, k) for k, v in id_to_word.items())
    return word_to_id, id_to_word


def save_json(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
