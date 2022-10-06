
from drop_eval import answer_json_to_strings
import os
import pickle
import torch
import random
import numpy as np
from drop_token import Token
import itertools
from executor import executor
import logging
from drop_reader import get_text_tokens
from utils import savejson,loadjson
logger=logging.getLogger()

def do_pad(batch, key, padding):
    lst = [v[key] for v in batch]
    return torch.from_numpy(np.column_stack((itertools.zip_longest(*lst, fillvalue=padding))))

def create_collate_fn(padding_idx=1, use_cuda=False):
    def collate_fn(batch):
        bsz = len(batch)
        #max_seq_len = max([v['input_ids'].shape[0] for v in batch])
        max_seq_len = 512 

        input_ids = torch.LongTensor(bsz, max_seq_len).fill_(padding_idx)
        bart_input_ids = torch.LongTensor(bsz, max_seq_len).fill_(padding_idx)

        input_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
        bart_input_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)

        input_segments = torch.LongTensor(bsz, max_seq_len).fill_(0)
        passage_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
        question_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)

        max_num_len = max([v['number_indices'].shape[0] for v in batch])
        number_indices = torch.LongTensor(bsz, max_num_len).fill_(-1)


        is_program_mask = torch.LongTensor(bsz).fill_(0)
        max_pans_choice = max([v['answer_as_passage_spans'].shape[0] for v in batch])
        answer_as_passage_spans = torch.LongTensor(bsz, max_pans_choice, 2).fill_(-1)

        max_qans_choice = max([v['answer_as_question_spans'].shape[0] for v in batch])
        answer_as_question_spans = torch.LongTensor(bsz, max_qans_choice, 2).fill_(-1)
        
        max_program_length = 128
        answer_as_program = torch.LongTensor(bsz, max_program_length ).fill_(padding_idx)

        answer_as_counts = torch.LongTensor(bsz).fill_(-1)

        max_text_answers = max([v['answer_as_text_to_disjoint_bios'].shape[0] for v in batch])
        max_answer_spans = max([v['answer_as_text_to_disjoint_bios'].shape[1] for v in batch])
        answer_as_text_to_disjoint_bios = torch.LongTensor(bsz, max_text_answers, max_answer_spans, max_seq_len).fill_(0)
        max_correct_sequences = max([v['answer_as_list_of_bios'].shape[0] for v in batch])
        answer_as_list_of_bios = torch.LongTensor(bsz, max_correct_sequences, max_seq_len).fill_(0)
        span_bio_labels = torch.LongTensor(bsz, max_seq_len).fill_(0)
        bio_wordpiece_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
        is_bio_mask = torch.LongTensor(bsz).fill_(0)
        is_ok_for_spans = torch.LongTensor(bsz).fill_(0)


        length_answering_abilities = batch[0]['valid_methods_mask'].shape[0]
        valid_methods_mask = torch.LongTensor(bsz, length_answering_abilities).fill_(0)


        for i in range(bsz):
            item = batch[i]
            try:
                seq_len = min(item['input_ids'].shape[0],512)
                input_ids[i][:seq_len] = torch.LongTensor(item['input_ids'][:seq_len])
                input_mask[i][:seq_len] = torch.LongTensor(item['input_mask'][:seq_len])

                bart_seq_len = min(item['bart_input_ids'].shape[0],512)
                bart_input_ids[i][:bart_seq_len] = torch.LongTensor(item['bart_input_ids'][:bart_seq_len])
                bart_input_mask[i][:bart_seq_len] = torch.LongTensor(item['bart_input_mask'][:bart_seq_len])

                input_segments[i][:seq_len] = torch.LongTensor(item['input_segments'][:seq_len])
                pm_len = min(item["passage_mask"].shape[0],512)
                passage_mask[i][:pm_len] = torch.LongTensor(item["passage_mask"][:pm_len])
                qm_len = min(item["question_mask"].shape[0],512)
                question_mask[i][:qm_len] = torch.LongTensor(item["question_mask"][:qm_len])

                number_indices[i][:item["number_indices"].shape[0]] = torch.LongTensor(item["number_indices"])

                answer_as_passage_spans[i][:item["answer_as_passage_spans"].shape[0],:] = torch.LongTensor(item["answer_as_passage_spans"])
                answer_as_question_spans[i][:item["answer_as_question_spans"].shape[0],:] = torch.LongTensor(item["answer_as_question_spans"])
                
                is_program_mask[i] = torch.LongTensor(item['is_program_mask'])
                program_len = min(item['answer_as_program'].shape[0],128)
                answer_as_program[i][:program_len] = torch.LongTensor(item['answer_as_program'][:program_len])

                answer_as_counts[i] = torch.LongTensor(item['answer_as_counts'])

                s0,s1,s2 = item['answer_as_text_to_disjoint_bios'].shape
                answer_as_text_to_disjoint_bios[i][:s0, :s1, :s2] = torch.LongTensor(item['answer_as_text_to_disjoint_bios'])

                s0, s1 = item['answer_as_list_of_bios'].shape
                answer_as_list_of_bios[i][:s0, :s1] = torch.LongTensor(item['answer_as_list_of_bios'])
                span_bio_labels[i][:item['span_bio_labels'].shape[0]] = torch.LongTensor(item['span_bio_labels'])

                is_bio_mask[i] = torch.LongTensor(item['is_bio_mask'])
                bio_wordpiece_mask[i][:item['bio_wordpiece_mask'].shape[0]] = torch.LongTensor(item['bio_wordpiece_mask'])

                is_ok_for_spans[i] = torch.LongTensor(item['is_ok_for_spans'])

                valid_methods_mask[i] = torch.LongTensor(item['valid_methods_mask'])
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                x=1


        out_batch = {"input_ids": input_ids,"bart_input_ids":bart_input_ids, 
                "input_mask": input_mask, "bart_input_mask":bart_input_mask ,"input_segments": input_segments,
               "passage_mask": passage_mask, "question_mask": question_mask,"number_indices":number_indices,
               "answer_as_passage_spans": answer_as_passage_spans,
               "answer_as_question_spans": answer_as_question_spans,
               "is_program_mask": is_program_mask,
               "answer_as_program": answer_as_program,
               "answer_as_counts": answer_as_counts.unsqueeze(1),
               "answer_as_text_to_disjoint_bios": answer_as_text_to_disjoint_bios,
               "answer_as_list_of_bios": answer_as_list_of_bios,
               "span_bio_labels": span_bio_labels,
               "is_bio_mask": is_bio_mask,
               "bio_wordpiece_mask": bio_wordpiece_mask,
               "is_ok_for_spans": is_ok_for_spans,
               "valid_methods_mask": valid_methods_mask,
               "metadata": [v['metadata'] for v in batch]}
        for k in out_batch.keys():
            if isinstance(out_batch[k], torch.Tensor):
                out_batch[k] = out_batch[k].cuda()

        return out_batch


    return collate_fn

def NumberTokenInPassageCheck(args, program, item, reader):
    if program == "NULL" :
        return 0
    if program == "" or not isinstance(program, str):
        return 1
    if len(program) > 128:
        return 1
    if args.delete_no_number_answer:
        if 'argmax' in program or 'argmin' in program:
            return 1    

    exe = executor(questionId = item['question_id'], program= program,reader = reader)
    dd = exe.tokenize(program)
    token_ = dd['token']
    type_ = dd['type']

    for token,type in zip(token_,type_):
        if args.delete_long_string_in_program:
            if type == "string":
                if len(token.split()) > 3:
                    return 1
            
        if type == "number":
            if token.startswith("N"): #
                offsets = item['passage_token_offsets']
                start_ = offsets[0][0]
                end_ = offsets[-1][1]
                valid_passage = item['original_passage'][start_:end_]
                if not token in valid_passage:
                    #logger.info("find a invalid program")
                    return 1
    return 0


import re
def delete_number_token(string):
    string = re.sub("@"," ",string)
    k = 100
    while k >= 0:
        string = re.sub('N'+str(k),'',string)
        string = re.sub('Q'+str(k),'',string)
        k = k - 1
    string =  re.sub("  ",' ',string)
    string = re.sub(' - ','-',string)
    return string

class DropDataBuilder(object):
    def __init__(self, args, data_mode, tokenizer, padding_idx=1,reader=None):
        self.cls_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.padding_idx = padding_idx
        self.is_train = data_mode == "train"
        
        self.questions = {}
        self.passages = {}

        self.answering_abilities = args.answering_abilities
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        dpath = "{}.pkl".format(data_mode)
        
        with open(f"data/{dpath}", "rb") as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)
        
        data_mode = 'train' if self.is_train else 'dev'

        
        _data = data['instances'] # for roberta
        ant_data = data['add_number_token_instances'] # for bart

        #default :the roberta input is same as the bart input
        #open this option will change the roberta input to _data
        if not args.no_add_number_token_text: 
            _data = ant_data

        raw_data = []
        for item,ant_item in zip(_data,ant_data):
            assert(item["question_id"] == ant_item["question_id"])
            #if not len(item["answer_texts"]) and data_mode == 'train':
            #    continue
            

            questionId = item["question_id"]

            ################## program
            if 'program' in args.answering_abilities and data_mode=='train':
                answer_program = ""
                if questionId in reader.questionId2program_dict and reader.questionId2program_dict[questionId] != None and reader.questionId2program_dict[questionId] != "":
                    answer_program = reader.questionId2program_dict[questionId]
                    if args.delete_null_program and answer_program=='NULL':
                        answer_program = ""
                if answer_program == "" and questionId in reader.additional_questionId2program and args.additional_program_level != -1:
                    answer_program = reader.additional_questionId2program[questionId]

                if  len(args.answering_abilities)==1 or args.valid_program_used:
                    if NumberTokenInPassageCheck(args, answer_program, item, reader):
                        continue
                

                item['answer_program'] = answer_program

            ################### 


            if args.additional_program_level == -1 and data_mode == "train": 
                if not (questionId in reader.questionId2program_dict):
                    continue
                if reader.questionId2program_dict[questionId] == None:
                    continue
            

            if data_mode == "train" and len(args.answering_abilities) == 1:
                if not item['answer_question_spans'] and args.answering_abilities[0] == 'questionspan':
                    continue
                if not item['answer_passage_spans'] and args.answering_abilities[0] == 'passagespan':
                    continue
                if not item['multi_span'] and args.answering_abilities[0] == 'multispans':
                    continue
                if not item['counts'] and args.answering_abilities[0] == 'count':
                    continue
                if not item['answer_program'] and args.answering_abilities[0] == 'program':
                    continue
            
            question_tokens = tokenizer.convert_tokens_to_ids(item["question_tokens"])
            passage_tokens = tokenizer.convert_tokens_to_ids(item["passage_tokens"])
            question_passage_tokens = [ Token(text=iii[0], idx=iii[1][0], edx=iii[1][1] ) for iii in zip(item["question_passage_tokens"],
                    [(0,0)] + item["question_token_offsets"] + [(0,0)]+ item["passage_token_offsets"] + [(0, 0)])]
            item["question_passage_tokens"] = question_passage_tokens


            ant_question_tokens = tokenizer.convert_tokens_to_ids(ant_item["question_tokens"])
            ant_passage_tokens = tokenizer.convert_tokens_to_ids(ant_item["passage_tokens"])

            if 'answer_program' not in item:
                item['answer_program'] = ""
            program_tokens = tokenizer.encode(item["answer_program"])

            raw_data.append((question_tokens,passage_tokens,ant_question_tokens,ant_passage_tokens,program_tokens,item,ant_item))
            
        if data_mode == "dev" and args.eval_dataset_sample!=-1:
            raw_data = random.sample(raw_data, args.eval_dataset_sample)
        if data_mode == "train" and args.train_dataset_sample!=-1:
            raw_data = random.sample(raw_data, args.train_dataset_sample)
        
        self.data = [self.build(item) for item in raw_data]
        del raw_data

        print("Load data size {}.".format(len(self.data)))
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def get(self, offset):
        return self.data[offset]

    def build(self, batch):
        q_tokens, p_tokens, ant_q_tokens,ant_p_tokens, program_tokens, metas, ant_metas = batch

        program_tokens = np.array(program_tokens)
        is_program_mask = np.zeros([1])
        if (metas["answer_program"] != None and metas["answer_program"]!="") or (not self.is_train):
            is_program_mask[0] = 1


        seq_len = len(q_tokens) + len(p_tokens) + 3
        num_len = max(1, len(metas["number_indices"]))
        pans_choice = max(1, len(metas["answer_passage_spans"]))
        qans_choice = max(1, len(metas["answer_question_spans"]))

        
        # qa input.
        input_segments = np.zeros(seq_len)

        # multiple span label
        max_text_answers = max(1, 0 if len(metas["multi_span"]) < 1 else
                                   len(metas["multi_span"][1]))
        max_answer_spans = max(1, 0 if len(metas["multi_span"]) < 1 else
                                   max([len(item) for item in metas["multi_span"][1]])
                                   )
        max_correct_sequences = max(1, 0 if len(metas["multi_span"]) < 1 else
                                   len(metas["multi_span"][2])
                                   )
        bio_wordpiece_mask = np.zeros([seq_len], dtype=np.int)
        answer_as_text_to_disjoint_bios = np.zeros([max_text_answers, max_answer_spans, seq_len])
        answer_as_list_of_bios = np.zeros([max_correct_sequences, seq_len])
        span_bio_labels = np.zeros([seq_len])

        answer_as_passage_spans = np.full([pans_choice, 2], -1)
        answer_as_question_spans = np.full([qans_choice, 2], -1)

        q_len = len(q_tokens)
        p_len = len(p_tokens)
        ant_q_len = len(ant_q_tokens)
        ant_p_len = len(ant_p_tokens)
        # input and their mask
        input_ids = np.array([self.cls_idx] + q_tokens + [self.sep_idx] + p_tokens + [self.sep_idx])
        bart_input_ids = np.array([self.cls_idx] +ant_q_tokens + [self.sep_idx] +ant_p_tokens + [self.sep_idx])
        input_mask = np.ones(3 + q_len + p_len, dtype=np.int)
        bart_input_mask = np.ones(3 + ant_q_len + ant_p_len, dtype=np.int)
        question_mask = np.zeros(seq_len)
        question_mask[1:1 + q_len] = 1
        passage_mask = np.zeros(seq_len)

        passage_start = q_len + 2
        passage_mask[passage_start: passage_start + p_len] = 1

        question_start = 1

        #number
        pn_len = len(metas["number_indices"]) - 1
        number_indices = np.full([num_len], -1)
        passage_number_order = np.full([num_len], -1)
        if pn_len > 0:
            number_indices[:pn_len] = passage_start + np.array(metas["number_indices"][:pn_len])
            number_indices[pn_len - 1] = 0




        # answer info
        pans_len = min(len(metas["answer_passage_spans"]), pans_choice)
        for j in range(pans_len):
            answer_as_passage_spans[j, 0] = np.array(metas["answer_passage_spans"][j][0]) + passage_start
            answer_as_passage_spans[j, 1] = np.array(metas["answer_passage_spans"][j][1]) + passage_start

        qans_len = min(len(metas["answer_question_spans"]), qans_choice)
        for j in range(qans_len):
            answer_as_question_spans[j, 0] = np.array(metas["answer_question_spans"][j][0]) + question_start
            answer_as_question_spans[j, 1] = np.array(metas["answer_question_spans"][j][1]) + question_start
        if len(metas["counts"]) > 0:
            answer_as_counts = np.array(metas["counts"])
        else:
            answer_as_counts = np.array([-1])



        # add multi span prediction
        cur_seq_len = q_len + p_len + 3
        bio_wordpiece_mask[:cur_seq_len] = np.array(metas["wordpiece_mask"][:cur_seq_len])
        is_bio_mask = np.zeros([1])
        
        if len(metas["multi_span"]) > 0:
            is_bio_mask[0] = metas["multi_span"][0]
            span_bio_labels[:cur_seq_len] = np.array(metas["multi_span"][-1][:cur_seq_len])
            for j in range(len(metas["multi_span"][1])):
                for k in range(len(metas["multi_span"][1][j])):
                    answer_as_text_to_disjoint_bios[j, k, :cur_seq_len] = np.array(metas["multi_span"][1][j][k][:cur_seq_len])
            for j in range(len(metas["multi_span"][2])):
                answer_as_list_of_bios[j, :cur_seq_len] = np.array(metas["multi_span"][2][j][:cur_seq_len])
        
        #valid_methods_mask = np.array([0,0,0,0,0])
        valid_methods_mask = np.zeros((len(self.answering_abilities)))
        for idx, answer_ability in enumerate(self.answering_abilities):
            if answer_ability == "passagespan" and len(metas['answer_passage_spans'])>0: 
                valid_methods_mask[idx] = 1
            if answer_ability == "questionspan" and len(metas['answer_question_spans'])>0:
                valid_methods_mask[idx] = 1
            if answer_ability == "multispans" and len(metas["multi_span"]) >0:
                valid_methods_mask[idx] = 1
            if answer_ability == "program" and (metas["answer_program"] != None and metas["answer_program"] != "" and metas['answer_program'] != "NULL"):
                valid_methods_mask[idx] = 1
            if answer_ability == "count" and len(metas["counts"]) > 0:
                valid_methods_mask[idx] = 1
        is_ok_for_spans = np.array([0])
        if len(metas["multi_span"]) >0 or len(metas['answer_passage_spans'])>0 or len(metas['answer_question_spans'])>0:
            is_ok_for_spans = np.array([1])


        out_batch = {"input_ids": input_ids,"bart_input_ids":bart_input_ids ,
                    "input_mask": input_mask,"bart_input_mask":bart_input_mask, "input_segments": input_segments,
                     "passage_mask": passage_mask, "question_mask": question_mask,"number_indices": number_indices,
                     "answer_as_passage_spans": answer_as_passage_spans,
                     "answer_as_question_spans": answer_as_question_spans,
                     "is_program_mask": is_program_mask,
                     "answer_as_program" : program_tokens,
                    "answer_as_counts": np.expand_dims(answer_as_counts, axis=1),
                     "answer_as_text_to_disjoint_bios": answer_as_text_to_disjoint_bios,
                     "answer_as_list_of_bios": answer_as_list_of_bios,
                     "span_bio_labels": span_bio_labels,
                     "is_bio_mask": is_bio_mask,
                     "bio_wordpiece_mask": bio_wordpiece_mask,
                     "valid_methods_mask": valid_methods_mask,
                     "is_ok_for_spans": is_ok_for_spans,
                     "metadata": metas}

        return out_batch

class DropBatchGen(object):
    def __init__(self, args, data_mode, tokenizer, padding_idx=1,reader = None):
        self.gen = DropDataBuilder(args, data_mode, tokenizer, padding_idx,reader)

    def __getitem__(self, index):
        return self.gen.get(index)

    def __len__(self):
        return len(self.gen.data)

    def reset(self):
        return self.gen.reset()



