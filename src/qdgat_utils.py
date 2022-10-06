import sys
import random
import torch
import numpy
import logging
from time import gmtime, strftime
from typing import Union, List, Tuple
from drop_eval import (get_metrics as drop_em_and_f1, answer_json_to_strings)
from utils import loadjson


NUM_NER_TYPES = ['ENT2NUM', 'NUMBER', 'PERCENT','MONEY','TIME','DATE','DURATION','ORDINAL', 'YARD']
def dump_gat_info(gnodes_mask, gedges, meta):
   qp_tokens = meta['question_passage_tokens']
   print(meta['original_passage'])
   print('==========mask================')
   gnodes_mask = gnodes_mask.detach().cpu().numpy()
   gnodes = []
   for j,mask in enumerate(gnodes_mask):
     pos = 0
     for i in range(len(mask)):
       if mask[i]>0: pos=i
     gnodes.append((meta['question_passage_tokens'][(mask[0]-1):(mask[pos])], meta['gnodes_type'][j]))
     print(gnodes[-1])
    
   print('==========edges================')
   edges = gedges.detach().cpu().numpy()
   a = {}
   for edge in (gedges.nonzero().detach().cpu().numpy()):
     etype = edges[edge[0], edge[1]]
     src = gnodes[edge[0]]
     dst = gnodes[edge[1]]
     if edge[0] not in a.keys(): 
       a[edge[0]]=[]
     a[edge[0]].append((dst[0],NUM_NER_TYPES[etype-1]))
   for k in a.keys():
     print(gnodes[k],a[k])



def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log

def format_number(number):
    if isinstance(number, int):
        return str(number)

    # we leave at most 3 decimal places
    num_str = '%.3f' % number

    for i in range(3):
        if num_str[-1] == '0':
            num_str = num_str[:-1]
        else:
            break

    if num_str[-1] == '.':
        num_str = num_str[:-1]

    # if number < 1, them we will omit the zero digit of the integer part
    if num_str[0] == '0' and len(num_str) > 1:
        num_str = num_str[1:]

    return num_str




class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths,ground_types):
    scores_for_ground_truths = []
    max_number = (-1.0,-1.0)
    max_type = "span"
    for idx, ground_truth in enumerate(ground_truths):
        score_tuple = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score_tuple)
        if max_number < score_tuple:
            max_number = score_tuple
            max_type = ground_types[idx]
    
    return max_number, max_type



class DropEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self, predicted_abilities_list=None) -> None:
        if predicted_abilities_list == None:
            self.predicted_abilities_list = ['passagespan','questionspan', 'multispans', 'program','count']
        else :
            self.predicted_abilities_list = predicted_abilities_list
        self.new_predicted_abilities_list = predicted_abilities_list + ['best_answer',]
        #print(self.predicted_abilities_list)
        self.type_list = ['number','spans','span','date',]
        self.examples = []
        self.case_logger = {}
        self.questionId2type = loadjson("data/questionId2type.json")
    def __call__(self, prediction: Union[str, List], ground_truths: List ,predicted_ability_str, questionId ):  # type: ignore
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List``
            All the ground truth answer annotations.
        """
        # If you wanted to split this out by answer type, you could look at [1] here and group by
        # that, instead of only keeping [0].

        ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
        ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in ground_truths]

        example = {}
        example["types"] = ground_truth_answer_types
        example["answer_texts"] = ground_truth_answer_strings
        answers = []
        cases = []

        best_answer = (-1,0,-1.0,)
        best_type = "number"

        for i in range(len(prediction)):
            typpp = self.predicted_abilities_list[i]
            predict_answer = prediction[i][0]
            original_predict_answer = prediction[i][1]
            (exact_match, f1_score), max_type = metric_max_over_ground_truths(
                    drop_em_and_f1,
                    predict_answer,
                    ground_truth_answer_strings,
                    ground_truth_answer_types,
            )
            if questionId in self.questionId2type:
                max_type = self.questionId2type[questionId]
            example[typpp] = {}
            example[typpp]["em"] = exact_match
            example[typpp]["f1"] = f1_score
            example[typpp]["type"] = max_type
            answers.append((exact_match, f1_score, max_type))
            cases.append((str(exact_match),str(f1_score),predict_answer,str(original_predict_answer)))
            
            if best_answer < (exact_match,f1_score,):
                best_answer = (exact_match, f1_score,)
                best_type = max_type

        if questionId in self.questionId2type:
            best_type = self.questionId2type[questionId]

        example['best_answer'] = {}
        example['best_answer']['em'] = best_answer[0]
        example['best_answer']['f1'] = best_answer[1]
        example['best_answer']['type'] = best_type


        example["predict_ability"] = predicted_ability_str
        self.examples.append(example)
        self.case_logger[questionId] = {}
        self.case_logger[questionId]['predict_ability'] = predicted_ability_str
        self.case_logger[questionId]['ground_truth_answer_strings'] = ground_truth_answer_strings
        self.case_logger[questionId]['ground_truth_answer_types'] = ground_truth_answer_types
        self.case_logger[questionId]['cases'] = cases
        return answers

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        out = {}
        def getkey(l, r):
            return "use way:"+l + " actual type:" + r 
        
        for i in range(len(self.new_predicted_abilities_list)):
            for j in range(len(self.type_list)):
                out[getkey(self.new_predicted_abilities_list[i],self.type_list[j])] = [0.0,0.0,0]
            out["real:" + self.new_predicted_abilities_list[i]] = [0.0,0.0,0]
        for i in range(len(self.type_list)):
            out[self.type_list[i]] = [0.0,0.0,0]


        for example in self.examples:            
            for i in range(len(self.new_predicted_abilities_list)):
                ability = self.new_predicted_abilities_list[i]
                predict_ability = example[ability]['type']
                em = example[ability]['em']
                f1 = example[ability]['f1']
                out[getkey(ability, predict_ability)][0] += em
                out[getkey(ability, predict_ability)][1] += f1
                out[getkey(ability, predict_ability)][2] += 1

                if ability == example['predict_ability']:
                    out[predict_ability][0] += em
                    out[predict_ability][1] += f1
                    out[predict_ability][2] += 1

            actual_predict_ability = example["predict_ability"]
            em = example[actual_predict_ability]['em']
            f1 = example[actual_predict_ability]['f1']
            out["real:" + actual_predict_ability][0]+= em
            out["real:" + actual_predict_ability][1]+= f1
            out["real:" + actual_predict_ability][2]+= 1





        out["total_em"] = sum( [out["real:" + str][0] for str in self.new_predicted_abilities_list])
        out["total_f1"] = sum( [out["real:" + str][1] for str in self.new_predicted_abilities_list])
        out["total_count"] = sum( [out["real:" + str][2] for str in self.new_predicted_abilities_list])

        for i in range(len(self.new_predicted_abilities_list)):
            for j in range(len(self.type_list)):
                em = out[getkey(self.new_predicted_abilities_list[i],self.type_list[j])][0]
                f1 = out[getkey(self.new_predicted_abilities_list[i],self.type_list[j])][1]
                count = out[getkey(self.new_predicted_abilities_list[i],self.type_list[j])][2]
                if count == 0:
                    out[getkey(self.new_predicted_abilities_list[i],self.type_list[j])] = [0.0,0.0,0]
                else :
                    out[getkey(self.new_predicted_abilities_list[i],self.type_list[j])] = [em/count,f1/count,count]
        for i in range(len(self.new_predicted_abilities_list)):
            em = out["real:" + self.new_predicted_abilities_list[i]][0]
            f1 = out["real:" + self.new_predicted_abilities_list[i]][1]
            count = out["real:" + self.new_predicted_abilities_list[i]][2]
            if count == 0:
                out["real:" + self.new_predicted_abilities_list[i]] = [0.0,0.0,0]
            else:
                out["real:" + self.new_predicted_abilities_list[i]] = [em/count,f1/count,count]
        if out["span"][2]+out["spans"][2]>0:
            out["span+spans"] = [(out["span"][0]+out["spans"][0])/(out["span"][2]+out["spans"][2]),(out["span"][1]+out["spans"][1])/(out["span"][2]+out["spans"][2]),out["span"][2]+out["spans"][2] ]
        for i in range(len(self.type_list)):
            em = out[self.type_list[i]][0]
            f1 = out[self.type_list[i]][1]
            count = out[self.type_list[i]][2]
            if count == 0:
                out[self.type_list[i]] = [0.0,0.0,0]
            else :
                out[self.type_list[i]] = [em/count, f1/count, count]
        
        

        count = out["total_count"]
        if count != 0:
            out["total_em"] /= count
            out["total_f1"] /= count
        else :
            out["total_em"] = 0
            out["total_f1"] = 0
        if reset:
            self.reset()
        return out

    def reset(self):
        self.examples = []
        self.case_logger = {}
    def __str__(self):
        str = ""
        return str
    