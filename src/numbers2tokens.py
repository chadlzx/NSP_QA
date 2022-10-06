
#use coreNLP to generate drop_dataset_{split}_annotation_corenlp.json

import argparse
import json
from utils import loadjson,savejson
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm 
import os
import sys

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }
    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))
def annotate_text(snlp, text: str):
    "Annotate a piece of text."
    text = text.replace('%', '%25')
    try: 
        annotations = json.loads(snlp.nlp.annotate(text, properties=snlp.props))
    except:
        print(repr(snlp.nlp.annotate(text, properties=snlp.props)))
        print(sys.exc_info())
    return annotations

def main():
    sNLP = StanfordNLP()
    for split in ['train','dev','test']:
        file_path = f"drop_dataset_{split}.json"
        out_path = f"drop_dataset_{split}_annotation_corenlp.json"
        reader = loadjson(file_path)
        for passageId, text in tqdm(reader.items()):
            passage = text['passage']
            reader[passageId]['passage'] = annotate_text(sNLP, passage)
            qa_pairs = text['qa_pairs']
            for idx, qa_pair in enumerate(qa_pairs):
                question = qa_pair['question']
                reader[passageId]['qa_pairs'][idx]['question']= annotate_text(sNLP, question)
        savejson(out_path, reader)

if __name__=='__main__':
    main()
