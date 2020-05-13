import glob
import json
import argparse
import os

def merge_words(f):
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
    merged_symptoms = [x.replace(" ","_") for x in symptoms]
    merged_f = f
    for i in range(len(symptoms)):
        if symptoms[i] in f:
            merged_f = merged_f.replace(symptoms[i], merged_symptoms[i])
    merged_f = merged_f.replace(","," ")
    merged_f = merged_f.replace("."," ")
    merged_f = merged_f.replace("'"," ")
    merged_f = merged_f.replace("("," ")
    merged_f = merged_f.replace(")"," ")
    merged_f = merged_f.lower()
    return merged_f

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


def append_abstract_body_text_to_file(all_json):
    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print('Processing index: {} of {}'.format(idx, len(all_json)))
        content = FileReader(entry)
        with open('covid_corpus.txt', 'a', encoding='utf-8') as f:
            a = merge_words(content.abstract)
            b = merge_words(content.body_text)
            f.write('{}\n\n{}\n\n\n\n\n'.format(a,b))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cord_dataset_path', type=str)
    args = parser.parse_args()
    root_path = args.cord_dataset_path
    #root_path = 'C:\\dataset'
    all_json = glob.glob('{}/**/*.json'.format(root_path), recursive=True)
    append_abstract_body_text_to_file(all_json)


if __name__ == "__main__":
    main()
