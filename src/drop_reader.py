import os
import numpy as np 
import json
import pickle
import re
import string
from typing import List,Dict,Any,Tuple,OrderedDict
from collections import defaultdict
from utils import savejson, loadjson, checkAnswer,getNumberFromString
from data_prepare import new_number2token
from drop_eval import (get_metrics as drop_em_and_f1, answer_json_to_strings)
from qdgat_utils import DropEmAndF1,metric_max_over_ground_truths
from number_type import getNumber
import itertools
from word2number.w2n import word_to_num
from executor import executor
from tqdm import tqdm
IGNORED_TOKENS = {'a', 'an', 'the'}
from transformers import logging
logger = logging.get_logger(__name__)
import html
from retrieve_dates import retrieve_dates

class drop_reader(object):

    def __init__(self):
        self.passageId2corenlp_passage_dict = {}
        self.passageId2corenlp_text_dict = {}
        self.passageId2text_dict = {}
        self.passageId2passage_dict = {}
        self.passageId2replace_number_token_passage_dict = {}
        self.passageId2add_number_token_passage_dict = {}
        self.passageId2clear_passage_dict = {}
        self.passageId2number2token_dict = {}

        self.questionId2corenlp_qa_pair_dict = {}
        self.questionId2passageId_dict = {}
        self.questionId2answer_dict = {}
        self.questionId2qa_pair_dict = {}
        self.questionId2number2token_dict = {}
        self.questionId2program_dict = {}
        self.additional_questionId2program0 = {}
        self.additional_questionId2program1 = {}
        self.additional_questionId2program2 = {}
        self.additional_questionId2program3 = {}

        self.questionId2add_number_token_question_dict = {}

        self.questionIds_train_version2 = []
        self.questionIds_dev_version2 = []

        self.questionId2old2new_dict = {}
        self.pre()

        self.questionId2answers = {}
        self.questionId2scores = {}
        self.questionId2f1 = {}
    def get_questionId2f1(self):
        
        for questionId, answers in self.questionId2answers.items():
            f1s = []
            for answer,_ in answers:
                answer_annotations = self.questionId2answer_dict[questionId]
                ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in answer_annotations] 
                ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in answer_annotations]
                (exact_match, f1_score), max_type = metric_max_over_ground_truths(
                    drop_em_and_f1,
                    answer,
                    ground_truth_answer_strings,
                    ground_truth_answer_types,
                )
                f1s.append(f1_score)
            self.questionId2f1[questionId] = f1s
    def clear_token(self, passage):
        annotated_text = ""
        for sentence in passage['sentences']:
            for idx, token in enumerate(sentence['tokens']):
                annotated_text += token['word']
                annotated_text += " " if token['index'] != len(sentence['tokens']) else ""
            annotated_text +=  " " if sentence['index'] != len(passage['sentences']) else ""
        return annotated_text
    def replace_number_token(self, passage, NorQ='N'):
        useful_ner = ['ORDINAL','DATE', 'NUMBER', 'MONEY','DURATION','PERCENT','TIME']
        annotated_text = ""
        num_id = 0
        for sentence in passage['sentences']:
            for idx, token in enumerate(sentence['tokens']):
                if token['ner'] in useful_ner:
                    if idx + 1 == len(sentence['tokens']):
                        annotated_text += NorQ+str(num_id)
                        num_id += 1
                        annotated_text += " " if token['index'] != len(sentence['tokens']) else ""
                    elif sentence['tokens'][idx+1]['ner'] != token['ner']:
                        annotated_text += NorQ+str(num_id)
                        num_id += 1
                        annotated_text += " " if token['index'] != len(sentence['tokens']) else ""
                    continue
                else :
                    annotated_text += token['word']
                    annotated_text += " " if token['index'] != len(sentence['tokens']) else ""
            annotated_text +=  " " if sentence['index'] != len(passage['sentences']) else ""
        return annotated_text

    def new_add_number_token(self, passage, NorQ='N'):
        number2token_ = {}
        useful_ner = ['ORDINAL','DATE', 'NUMBER', 'MONEY','DURATION','PERCENT','TIME']
        FLAG_NER = '@'

        new_num = 0
        old_num = 0
        old2new_dict = {}
        idx2token = {}
        invalid_token = {}

        for sentence in passage['sentences']:
            pre = []
            idx2token[sentence['index']] = {}
            invalid_token[sentence['index']] = {}
            for id_entity, entity in enumerate(sentence['entitymentions']):
                if entity['ner'] in useful_ner:
                    if id_entity+1 < len(sentence['entitymentions']):
                        next_entity = sentence['entitymentions'][id_entity+1]
                        if next_entity['tokenBegin'] == entity['tokenEnd'] and next_entity['ner'] == entity['ner']:
                            pre.append(entity)
                            continue
                    
                    def getNumType(Entity):
                        
                        number = getNumber(Entity)
                        return number

                    if pre !=[] :
                        pre.append(entity)
                        pre = [entity for entity in pre if entity['text'].lower()!='and']
                        for idx, e in enumerate(pre):
                            number2token_[NorQ+str(new_num)] = getNumType(e)
                            old2new_dict[NorQ+str(old_num)+ '_' + str(idx)] = NorQ +str(new_num)
                            idx2token[sentence['index']][e['tokenEnd'] - 1] = NorQ + str(new_num) 
                            new_num += 1
                        pre = []
                    else :
                        if len(getNumberFromString(entity['text'])) <= 1 or (not ('-' in entity['text'] or ':' in entity['text'] or '/' in entity['text'])): 
                            number2token_[NorQ+str(new_num)] = getNumType(entity)
                            idx2token[sentence['index']][entity['tokenEnd'] - 1] = NorQ + str(new_num) 
                            old2new_dict[NorQ + str(old_num)] = NorQ + str(new_num)
                            new_num += 1

                        else : 
                            iter = re.finditer("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", entity['text'])
                            num_indexs = [(i.group(), i.span()) for i in iter]
                            text = entity['text']
                            len_ins = 0
                            for i in range(entity['tokenBegin'],entity['tokenEnd']):
                                invalid_token[sentence['index']][i] = ""
                            for idx,( num,num_idx) in enumerate(num_indexs):
                                try:
                                    num = abs(float(re.sub(',','',num)))
                                except:
                                    num = 0.0
                                number2token_[NorQ+str(new_num)] = getNumber(num)
                                old2new_dict[NorQ+str(old_num)+ '_' + str(idx)] = NorQ +str(new_num)
                                inserted_string = FLAG_NER + NorQ + str(new_num) + " "
                                new_num += 1

                                text = text[: len_ins + num_idx[1]] + inserted_string + text[ len_ins + num_idx[1]:]
                                len_ins += len(inserted_string)
                            text = ' ' + text
                            invalid_token[sentence['index']][entity['tokenBegin']] = text 
                            old2new_dict[NorQ+ str(old_num)] = NorQ + str(new_num-1)
                    old_num += 1

        for i in range(11):
            number2token_['X'+str(i)] = getNumber(i)
            old2new_dict['X'+str(i)] = 'X' + str(i)
        old2new_dict['1'] = '1'

        num_id = 0
        annotated_text = ""
        for index_sentence,sentence in enumerate(passage['sentences']):
            for idx, token in enumerate(sentence['tokens']):
                if idx in invalid_token[index_sentence] :
                    if invalid_token[index_sentence].get(idx) != "" :
                        annotated_text += invalid_token[index_sentence][idx] + (" " if token['index'] != len(sentence['tokens']) else "")
                    continue
                    
                annotated_text += token['word']
                if idx in idx2token[index_sentence]:
                    annotated_text += FLAG_NER + idx2token[index_sentence][idx]
                annotated_text += " " if token['index'] != len(sentence['tokens']) else ""
            annotated_text +=  " " if sentence['index'] != len(passage['sentences']) else ""
        return annotated_text, old2new_dict

    def pre(self):
        #The annotations_v3,v2,v1 are the original three batches of annotation and fix_program, fix_program_v2 are two later fix files.
        #Use priority: fix_program_v2 > fix_program > annotations_v3 = annotations_v2 = annotations
        #They are all questionId2program dict format 
        #The combination is the final training program data

        #if os.path.exists("data/questionId2program.json"):
        original_annotation = loadjson("data/questionId2program.json")

        #original_annotation = loadjson("data/new_drop_dataset_annotation.json")
        #annotations_v2 = loadjson("data/new_drop_dataset_annotation_v2.json")
        #annotations_v3 = loadjson("data/new_drop_dataset_annotation_v3.json")
        #original_annotation = dict(original_annotation,**annotations_v2)
        #original_annotation = dict(original_annotation,**annotations_v3)
        
        #self.fix_program = {}
        #self.fix_program_v2 = {}

        #if os.path.exists("data/fix_program.json"):
        #    self.fix_program = loadjson("data/fix_program.json")
        #if os.path.exists("data/fix_program_v2.json"):
        #    self.fix_program_v2 = loadjson("data/fix_program_v2.json")

        

        #not used in final version
        """
        self.additional_questionId2program0 = loadjson("data/additional_questionId2program_level0.json")
        self.additional_questionId2program1 = loadjson("data/additional_questionId2program_level1.json")
        self.additional_questionId2program2 = loadjson("data/additional_questionId2program_level2.json")
        self.additional_questionId2program3 = loadjson("data/additional_questionId2program_level3.json")
        """
        if os.path.exists("data/additional_questionId2program.json"):
            self.additional_questionId2program = loadjson("data/additional_questionId2program.json")
            logger.info(f"add additional questionId2program: {len(self.additional_questionId2program)}")


        for split in ['train','dev','test']:
            
            # original DROP dataset files
            original_passage_path = f"data/drop_dataset_{split}.json"
            original_passage = loadjson(original_passage_path)
            
            # DROP dataset file after corenlp tokenize, ner and others for texts
            corenlp_passage_path = f"data/corenlp_result/drop_dataset_{split}_annotation_corenlp.json"

            corenlp_passage = loadjson(corenlp_passage_path)

            for passageId, text in original_passage.items():
                
                #text['passage'] =  html.unescape(text['passage'])
                self.passageId2passage_dict[passageId] =text['passage']
                self.passageId2text_dict[passageId] = text
                self.passageId2corenlp_text_dict[passageId] = corenlp_passage[passageId]
                self.passageId2clear_passage_dict[passageId] = self.clear_token(corenlp_passage[passageId]['passage'])
                self.passageId2corenlp_passage_dict[passageId] = corenlp_passage[passageId]['passage']
                self.passageId2replace_number_token_passage_dict[passageId] = self.replace_number_token(corenlp_passage[passageId]['passage'])
                
                add_number_token_passage, old2new_dict = self.new_add_number_token(corenlp_passage[passageId]['passage'])
                self.passageId2add_number_token_passage_dict[passageId] = add_number_token_passage
                self.passageId2number2token_dict[passageId] =  new_number2token(corenlp_passage[passageId]['passage'],'N')


                for qa_pair in text['qa_pairs']:
                    question = qa_pair['question']
                    questionId = qa_pair['query_id']
                    answer = []
                    if 'answer' in qa_pair:
                        answer = [qa_pair['answer'],] 
                    if 'validated_answers' in qa_pair:
                        for ans in qa_pair['validated_answers']:
                            answer.append(ans)
                    self.questionId2passageId_dict[questionId] = passageId
                    self.questionId2answer_dict[questionId] = answer
                    self.questionId2qa_pair_dict[questionId] = qa_pair
                    if questionId in original_annotation:
                        self.questionId2program_dict[questionId] = original_annotation[questionId]
                    else:
                        self.questionId2program_dict[questionId] = None
                    
                    
                for qa_pair in corenlp_passage[passageId]['qa_pairs']:
                    question = qa_pair['question']
                    questionId = qa_pair['query_id']
                    self.questionId2corenlp_qa_pair_dict[questionId] = qa_pair
                    self.questionId2add_number_token_question_dict[questionId], old2new_dict_q = self.new_add_number_token(question, 'Q')

                    self.questionId2number2token_dict[questionId] = dict(new_number2token(question, 'Q'), **self.passageId2number2token_dict[passageId] )
                    self.questionId2old2new_dict[questionId] = dict(old2new_dict, **old2new_dict_q)

    def questionId2passageId(self, questionId):
        if questionId not in self.questionId2passageId_dict:
            return ""
        return self.questionId2passageId_dict[questionId]
    def questionId2qa_pair(self, questionId):
        return self.questionId2qa_pair_dict[questionId]
    def questionId2answer(self, questionId):
        return self.questionId2answer_dict[questionId]
    def questionId2corenlp_qa_pair(self, questionId):
        return self.questionId2corenlp_qa_pair_dict[questionId]
    def questionId2number2token(self, questionId):
        return self.questionId2number2token_dict[questionId]

    def passageId2add_number_token_passage(self, passageId):
        return self.passageId2add_number_token_passage_dict[passageId]
    def passageId2replace_number_token_passage(self, passageId):
        return self.passageId2replace_number_token_passage_dict[passageId]
    def passageId2corenlp_passage(self, passageId):
        return self.passageId2corenlp_passage_dict[passageId]
    def passageId2passage(self, passageId):
        return self.passageId2passage_dict[passageId]
    def passageId2text(self, passageId):
        return self.passageId2text_dict[passageId]
    def passageId2corenlp_text(self, passageId):
        return self.passageId2corenlp_text_dict[passageId]
    def passageId2clear_passage(self, passageId):
        return self.passageId2clear_passage_dict[passageId]
    def passageId2number2token(self, passageId):
        return self.passageId2number2token_dict[passageId]

def clean(text):
    return re.sub("[.]?:\d+\s+([A-Z][a-z])",'. \g<1>',text)
FLAG_SENTENCE = '##'
FLAG_NER = '@'
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_"])
USTRIPPED_CHARACTERS = ''.join([u"Ġ"])

def get_number_from_word(word, improve_number_extraction=True):
    punctuation = string.punctuation.replace('-', '')
    word = word.lower().strip(punctuation)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                if improve_number_extraction:
                    if re.match('^\d*1st$', word):  # ending in '1st'
                        number = int(word[:-2])
                    elif re.match('^\d*2nd$', word):  # ending in '2nd'
                        number = int(word[:-2])
                    elif re.match('^\d*3rd$', word):  # ending in '3rd'
                        number = int(word[:-2])
                    elif re.match('^\d+th$', word):  # ending in <digits>th
                        # Many occurrences are when referring to centuries (e.g "the *19th* century")
                        number = int(word[:-2])
                    elif len(word) > 1 and word[-2] == '0' and re.match('^\d+s$', word):
                        # Decades, e.g. "1960s".
                        # Other sequences of digits ending with s (there are 39 of these in the training
                        # set), do not seem to be arithmetically related, as they are usually proper
                        # names, like model numbers.
                        number = int(word[:-1])
                    elif len(word) > 4 and re.match('^\d+(\.?\d+)?/km[²2]$', word):
                        # per square kilometer, e.g "73/km²" or "3057.4/km2"
                        if '.' in word:
                            number = float(word[:-4])
                        else:
                            number = int(word[:-4])
                    elif len(word) > 6 and re.match('^\d+(\.?\d+)?/month$', word):
                        # per month, e.g "1050.95/month"
                        if '.' in word:
                            number = float(word[:-6])
                        else:
                            number = int(word[:-6])
                    else:
                        return None
                else:
                    return None
    return number


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    res = []
    start = False
    for token in tokens:
        if token.endswith(FLAG_SENTENCE) or token.endswith(FLAG_NER):
            res.append(token)
            continue
            
        if len(token) == 1:
            res.append(token)
            continue
        
        start = 0
        while start < len(token):
            if token[start] not in STRIPPED_CHARACTERS:
                break
            start += 1

        end = len(token) - 1
        while end >= 0:
            if token[end] not in STRIPPED_CHARACTERS:
                break
           
            end-=1
        
        if start > 0:
            res.extend(list(token[0:start]))
        
        res.append(token[start:end+1])

        if end < len(token)-1:
            res.extend(list(token[end+1:]))
    
    return [item for item in res if item]

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def drop_tokenize(text, tokenizer, is_answer=False):
    split_tokens = []
    sub_token_offsets = []
    ingored_tokens = []
    new_text = ''
    number_indices = []

    text =  re.sub('\s\'\ss','\'s',text)
    tokens = text.split()
    
    for i in range(len(tokens)):
        
        while True:
            pos = tokens[i].find(FLAG_NER)
            if pos < 0:
                break
            if pos > 0:
                pos2 = tokens[i].find(FLAG_NER, pos+1)            
                tokens[i] = tokens[i][:pos]+tokens[i][pos2+len(FLAG_NER):]
            elif pos == 0:
                tokens[i] = ""

        if tokens[i].endswith(FLAG_SENTENCE) or i == len(tokens)-1:
            if tokens[i].endswith(FLAG_SENTENCE):
                tokens[i] = tokens[i][:-len(FLAG_SENTENCE)]
            assert len(tokens[i]) >= 0
    new_text = ' '.join(tokens)

    if FLAG_NER in new_text:
        print(new_text)
        print(tokens)

        exit(0)


    word_piece_mask = []
    word_to_char_offset = []
    prev_is_whitespace = True
    tokens = []

    for i, c in enumerate(new_text):

        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            word_to_char_offset.append(i)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
                word_to_char_offset.append(i)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  

    def checkNumber(num):
        for a in num:
            if a > '9' or a < '0':
                return False
        return True
    

    for i, token in enumerate(tokens):
        
        index = word_to_char_offset[i]
        if i != 0 or is_answer:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        token_number = get_number_from_word(token)
        if token_number is not None:
            number_indices.append(len(split_tokens))

        ingored_tokens_flag = 0
        if (token.startswith('Q') or token.startswith('N')) and checkNumber(token[1:]):
            ingored_tokens_flag = 1

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            sub_token_offsets.append((index, index + len(token)))
            ingored_tokens.append(ingored_tokens_flag)
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)


    return split_tokens, sub_token_offsets,number_indices, word_piece_mask, new_text, ingored_tokens


def clipped_passage_num(number_indices, plen):
    number_indices = [indice for indice in number_indices if indice < plen]
    return number_indices

def get_text_tokens(text, tokenizer):
    text = clean(text)
    text = " ".join(whitespace_tokenize(text))
    tokens, offset, indices, wordpiece_mask, texts, ingored_tokens =  drop_tokenize(text, tokenizer)
    return tokens

class reader(object):
    def __init__(self, tokenizer, mode, drop_reader):
        self.tokenizer = tokenizer
        self.mode = mode
        self.question_length_limit = 46
        self.max_pieces = 512
        self.flexibility_threshold = 1000
        self.drop_reader = drop_reader

    def read(self):
        symbolic_corenlp_path = f"data/corenlp_result/drop_dataset_{self.mode}_annotation.json"
        with open(symbolic_corenlp_path) as dataset_file:
            dataset_symbolic = json.load(dataset_file)

        neural_corenlp_path = f"data/corenlp_result/drop_dataset_{self.mode}_annotation.json"
        with open(neural_corenlp_path) as dataset_file:
            dataset_neural = json.load(dataset_file)


        add_number_token_instances = []
        instances = []
        for passage_id, passage_info in tqdm(dataset_neural.items()):

            for question_answer in passage_info["qa_pairs"]:
                passage_text = passage_info["passage"]
                question_id = question_answer["query_id"]
                question_text = question_answer["question"].strip()
                answer_annotations = []
                if "answer" in question_answer:
                    answer_annotations.append(question_answer["answer"])
                if "validated_answers" in question_answer:
                    answer_annotations += question_answer["validated_answers"]

                passage_text = re.sub("tp@ckl","@",passage_text)
                question_text = re.sub("tp@ckl","@",question_text)
                passage_text = re.sub("tp#ckl","##",passage_text)
                question_text = re.sub("tp#ckl","##",question_text)

                instance = self.text_to_instance(question_text, passage_text, question_id, passage_id,
                                                 answer_annotations, True)                    
                #logger.info("{}".format(instance['question_passage_tokens']))

                passage_text = self.drop_reader.passageId2add_number_token_passage(passage_id)
                passage_text = re.sub("@"," ",passage_text)

                corenlp_question = self.drop_reader.questionId2corenlp_qa_pair(question_id)['question']
                question_text,_ = self.drop_reader.new_add_number_token(corenlp_question,'Q')
                question_text = re.sub("@"," ",question_text)
                
                instance_ = self.text_to_instance(question_text, passage_text, question_id, passage_id,
                                                 answer_annotations, False)
                #logger.info("{}".format(instance_['question_passage_tokens']))

                
                if instance is not None and instance_ is not None:
                    instances.append(instance)
                    add_number_token_instances.append(instance_)
        print(f"Now add_number_token_instances: {len(add_number_token_instances)} questions.")
        print(f"Now instances: {len(instances)} questions.")

        return {"add_number_token_instances":add_number_token_instances, "instances":instances}

    def text_to_instance(self, question_text, passage_text, question_id, passage_id, answer_annotations, is_neural):
        passage_text = clean(passage_text)
        question_text = clean(question_text)
        passage_text = " ".join(whitespace_tokenize(passage_text))
        question_text = " ".join(whitespace_tokenize(question_text))

        if is_neural:
            passage_text = retrieve_dates(passage_text)

        passage_tokens, passage_offset, number_indices, passage_wordpiece_mask, passage_text, passage_ingored_tokens =  \
            drop_tokenize(passage_text, self.tokenizer)
        question_tokens, question_offset,question_number_indices,  question_wordpiece_mask, question_text, question_ingored_tokens = \
            drop_tokenize(question_text, self.tokenizer)
        
        
        question_tokens = question_tokens[:self.question_length_limit]
        question_number_indices = clipped_passage_num(
            question_number_indices, len(question_tokens)
        )
        qp_tokens = ["<s>"] + question_tokens + ["</s>"] + passage_tokens
        qp_ingored_tokens = [1] + question_ingored_tokens + [1] + passage_ingored_tokens
        qp_wordpiece_mask = [1] + question_wordpiece_mask + [1] + passage_wordpiece_mask
        q_len = len(question_tokens)
        
        if len(qp_tokens) > self.max_pieces - 1:
            qp_tokens = qp_tokens[:self.max_pieces - 1]
            qp_ingored_tokens = qp_ingored_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - q_len - 3]
            passage_offset = passage_offset[:self.max_pieces - q_len - 3]
            plen = len(passage_tokens)
            number_indices = clipped_passage_num(number_indices, plen)
            qp_wordpiece_mask = qp_wordpiece_mask[:self.max_pieces - 1]

        qp_ingored_tokens += [1]
        qp_tokens += ["</s>"]
        qp_wordpiece_mask += [1]

        answer_type: str = None
        answer_texts: List[str] = []
        if answer_annotations:
            answer_type, answer_texts = self.extract_answer_info_from_annotation(answer_annotations[0])
            answer_texts = [" ".join(whitespace_tokenize(answer_text)) for answer_text in answer_texts]


        tokenized_answer_texts = []
        ignored_answer_tokens = []
        specific_answer_type = "single_span"
        for answer_text in answer_texts:
            answer_tokens, _,_,  _, _, answer_ignored_tokens = drop_tokenize(answer_text, self.tokenizer, True)
            if answer_type in ["span", "spans"]:
                answer_texts = list(OrderedDict.fromkeys(answer_texts))
            
            if answer_type == "spans" and len(answer_texts) > 1:
                specific_answer_type = "multi_span"
            
            tokenized_answer_text = " ".join(answer_tokens)
            if tokenized_answer_text not in tokenized_answer_texts:
                tokenized_answer_texts.append(tokenized_answer_text)
                ignored_answer_tokens.append(answer_ignored_tokens)
        
        number_indices = [indice + 1 for indice in number_indices]
        number_indices.append(0)
        number_indices.append(-1)#
        
        valid_passage_spans = self.find_valid_spans(passage_tokens, tokenized_answer_texts, passage_ingored_tokens) if tokenized_answer_texts else []



        if len(valid_passage_spans) > 0:
            valid_question_spans = []
        else:
            valid_question_spans = self.find_valid_spans(question_tokens,
                                                        tokenized_answer_texts, question_ingored_tokens) if tokenized_answer_texts else []



        target_numbers = []
        # `answer_texts` is a list of valid answers.
        for answer_text in answer_texts:
            number = get_number_from_word(answer_text, True)
            if number is not None:
                target_numbers.append(number)
        valid_counts: List[int] = []
        if answer_type in ["number"]:
            # Currently we only support count number 0 ~ 9
            numbers_for_count = list(range(10))
            valid_counts = self.find_valid_counts(numbers_for_count, target_numbers)

        


        no_answer_bios = [0] * len(qp_tokens)
        #if specific_answer_type == "multi_span" and (len(valid_passage_spans) > 0 or len(valid_question_spans) > 0):    
        if specific_answer_type == 'multi_span' and  (len(valid_passage_spans) > 0 or len(valid_question_spans) > 0):
            spans_dict = {}
            text_to_disjoint_bios = []
            flexibility_count = 1

            for tokenized_answer_text in tokenized_answer_texts:
                
                #logger.info(f"tokenized_answer_text: { tokenized_answer_text}")
                spans = self.find_valid_spans(qp_tokens, [tokenized_answer_text], qp_ingored_tokens)
                #logger.info(f"qp_tokens: {qp_tokens}")
                #logger.info(f"tokenized_answer_text: {spans}")
                #logger.info(f"qp_ingored_tokens: {qp_ingored_tokens}")
                if len(spans) == 0:
                    continue
                spans_dict[tokenized_answer_text] = spans

                disjoint_bios = []
                for span_ind, span in enumerate(spans):
                    bios = create_bio_labels_simple([span], len(qp_tokens))
                    disjoint_bios.append(bios)

                text_to_disjoint_bios.append(disjoint_bios)
                flexibility_count *= ((2 ** len(spans)) - 1)

            answer_as_text_to_disjoint_bios = text_to_disjoint_bios

            if (flexibility_count < self.flexibility_threshold):
                # generate all non-empty span combinations per each text
                spans_combinations_dict = {}
                for key, spans in spans_dict.items():
                    spans_combinations_dict[key] = all_combinations = []
                    for i in range(1, len(spans) + 1):
                        all_combinations += list(itertools.combinations(spans, i))

                # calculate product between all the combinations per each text
                packed_gold_spans_list = itertools.product(*list(spans_combinations_dict.values()))
                bios_list = []
                for packed_gold_spans in packed_gold_spans_list:
                    gold_spans = [s for sublist in packed_gold_spans for s in sublist]
                    bios = create_bio_labels_simple(gold_spans, len(qp_tokens))
                    bios_list.append(bios)
                answer_as_list_of_bios = bios_list
                answer_as_text_to_disjoint_bios = [[no_answer_bios]]
            else:
                answer_as_list_of_bios = [no_answer_bios]


            bio_labels = create_bio_labels(valid_question_spans, valid_passage_spans, len(qp_tokens), len(question_tokens))
            span_bio_labels = bio_labels

            is_bio_mask = 1
            multi_span = [is_bio_mask, answer_as_text_to_disjoint_bios, answer_as_list_of_bios, span_bio_labels]
        else:
            multi_span = []
        

        valid_passage_spans = valid_passage_spans if specific_answer_type != "multi_span" or len(multi_span) < 1  else []
        valid_question_spans = valid_question_spans if specific_answer_type != "multi_span" or len(multi_span) < 1 else []
        
        program_answer = ""
        
        answer_info = {"answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                           "answer_passage_spans": valid_passage_spans,
                           "answer_question_spans": valid_question_spans,
                           "answer_program":  program_answer,
                           "counts": valid_counts,
                           "answer_count": valid_counts,
                           "multi_span": multi_span}
        return self.make_marginal_drop_instance(question_tokens,
                                                    passage_tokens,
                                                    qp_tokens,
                                                    number_indices,
                                                    qp_wordpiece_mask,
                                                    answer_info,
                                                    additional_metadata={"original_passage": passage_text,
                                                                         "passage_token_offsets": passage_offset,
                                                                         "original_question": question_text,
                                                                         "question_token_offsets": question_offset,
                                                                         "passage_id": passage_id,
                                                                         "question_id": question_id,
                                                                         "answer_info": answer_info,
                                                                         "answer_annotations": answer_annotations})


    @staticmethod
    def make_marginal_drop_instance(question_tokens: List[str],
                                    passage_tokens: List[str],
                                    question_passage_tokens: List[str],
                                    number_indices,
                                    wordpiece_mask: List[int],
                                    answer_info: Dict[str, Any] = None,
                                    additional_metadata: Dict[str, Any] = None):
        metadata = {
                    "question_tokens": [token for token in question_tokens],
                    "passage_tokens": [token for token in passage_tokens],
                    "question_passage_tokens": question_passage_tokens,
                    "number_indices": number_indices,
                    "wordpiece_mask": wordpiece_mask
                    }
        if answer_info:
            metadata["answer_texts"] = answer_info["answer_texts"]
            metadata["answer_passage_spans"] = answer_info["answer_passage_spans"]
            metadata["answer_question_spans"] = answer_info["answer_question_spans"]
            metadata["answer_program"] = answer_info["answer_program"]
            metadata["multi_span"] = answer_info["multi_span"]
            metadata["counts"] = answer_info["counts"]
        metadata.update(additional_metadata)
        return metadata

    @staticmethod
    def find_valid_spans(passage_tokens: List[str], answer_texts: List[str], ingored_tokens) -> List[Tuple[int, int]]:
        
        normalized_tokens = [token.strip(USTRIPPED_CHARACTERS).lower() for token in passage_tokens]
        word_positions: Dict[str, List[int]] = defaultdict(list)
        
        

        for i, token in enumerate(normalized_tokens):
            if ingored_tokens[i] == 1:
                continue
            word_positions[token].append(i)
        
        spans = []
        for answer_text in answer_texts:
            answer_tokens = [token.strip(USTRIPPED_CHARACTERS).lower() for token in answer_text.split()]
            num_answer_tokens = len(answer_tokens)
            
            if answer_tokens[0] not in word_positions:
                continue
            for span_start in word_positions[answer_tokens[0]]:
                span_end = span_start 
                answer_index = 1
                while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                    token = normalized_tokens[span_end + 1]
                    if answer_tokens[answer_index] == token:
                        answer_index += 1
                        span_end += 1
                    elif token in IGNORED_TOKENS or ingored_tokens[span_end + 1] == 1:
                        span_end += 1
                    else:
                        break
                if num_answer_tokens == answer_index:
                    spans.append((span_start, span_end))
        return spans

    @staticmethod
    def extract_answer_info_from_annotation(answer_annotation: Dict[str, Any]) -> Tuple[str, List[str]]:
        answer_type = None
        if answer_annotation["spans"]:
            answer_type = "spans"
        elif answer_annotation["number"]:
            answer_type = "number"
        elif any(answer_annotation["date"].values()):
            answer_type = "date"

        answer_content = answer_annotation[answer_type] if answer_type is not None else None

        answer_texts: List[str] = []
        if answer_type is None:  # No answer
            pass
        elif answer_type == "spans":
            # answer_content is a list of string in this case
            answer_texts = answer_content
        elif answer_type == "date":
            # answer_content is a dict with "month", "day", "year" as the keys
            date_tokens = [answer_content[key] for key in ["month", "day", "year"] if
                           key in answer_content and answer_content[key]]
            answer_texts = date_tokens
        elif answer_type == "number":
            # answer_content is a string of number
            answer_texts = [answer_content]
        return answer_type, answer_texts

    @staticmethod
    def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
        valid_indices = []
        for index, number in enumerate(count_numbers):
            if number in targets:
                valid_indices.append(index)
        return valid_indices


def preprocess(tokenizer, drop_reader):
    for mode in ['train','dev','test']:
        if os.path.exists(f"data/{mode}.pkl"):
            continue
        reader_ = reader(tokenizer, mode, drop_reader)
        data = reader_.read()
        with open(f"data/{mode}.pkl", "wb") as f:
            pickle.dump(data, f)

#question spans and passage spans in labels
def create_bio_labels(question_spans, passage_spans, n_labels, question_len):
    labels = [0] * n_labels

    start_offset = 1
    for span in question_spans:
        start = span[0]
        end = span[1]
        labels[start+ start_offset] = 1 
        labels[start+start_offset+1: end+1+start_offset] = [2] * (end - start )

    start_offset = question_len + 2
    for span in passage_spans:
        start = span[0]
        end = span[1]
        labels[start+ start_offset] = 1 
        labels[start+start_offset+1: end+1+start_offset] = [2] * (end - start )
    return labels

def create_bio_labels_simple(spans, n_labels):
    labels = [0] * n_labels
    for span in spans:
        start = span[0]
        end = span[1]
        labels[start] = 1
        labels[start+1:end+1] = [2] * (end - start)
    return labels