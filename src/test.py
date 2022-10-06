
from data_prepare import new_number2token
import os
from utils import savejson,loadjson,getNumberFromString
import logging
logger = logging.getLogger()
from drop_reader import drop_reader
from drop_eval import answer_json_to_strings
from tqdm import tqdm
from executor import executor
from drop_eval import (get_metrics as drop_em_and_f1, answer_json_to_strings)
from qdgat_utils import DropEmAndF1,metric_max_over_ground_truths
from utils import answer_text,print_eval_result
from transformers import AutoTokenizer,RobertaTokenizer
import math
import argparse
import random
import html

def generate_eval_result():
    ant = [False,True,]
    num_decoder_layers = ['2','4','6','12']
    encoder_type = ['roberta_encoder','bart_encoder',]

    for encoder in encoder_type:
        for if_ant in ant:
            for num_layers in num_decoder_layers:
                ant_name = '_ant' if if_ant else ''
                encoder_name = 'r' if encoder == 'roberta_encoder' else 'b'
                num_name = num_layers

                #name = f"logger/flex_{encoder_name}el{num_name}_all_l5{ant_name}_eval_result.json"
                eval_result = loadjson(name)
                
                

                #print("name: {}".format(name))
                #print("spans: {} span: {}".format(eval_result['spans'],eval_result['span']))
                em,f1,num_example = eval_result['spans'][:3]
                em_,f1_,num_example_ = eval_result['span'][:3]
                em_ = em_ * num_example_ + em * num_example
                f1_ = f1_ * num_example_ + f1 * num_example
                num_example_ += num_example
                em_ /= num_example_
                f1_ /= num_example_
                print("{},{}".format(round(em_*100,2),round(f1_*100,2)))



                #print("{},{},{},{},{},{}".format(round(eval_result['number'][0]*100,2),round(eval_result['number'][1]*100,2),round(eval_result['date'][0]*100,2),round(eval_result['date'][1]*100,2),round(eval_result['span'][0]*100,2),round(eval_result['span'][1]*100,2)))


def get_bad_case1(reader):
    #os.system("hadoop fs -get logger/flex_rel2_all_l5_app_uct/logger/flex_rel2_all_l5_app_uct_case_logger.json logger")
    path = "logger/flex_rel2_all_l5_app_uct_case_logger.json"
    case_logger_1 = loadjson(path)

    badcase_logger1 = {}
    goodcase_logger1 = {}
    method = ["passagespan","questionspan","multispans","program","count"]


    good_nfl_badcase,good_his_badcase = 0,0
    nfl_goodcase_logger, his_goodcase_logger = {}, {}
    good_nfl_passage, good_his_passage = {},{}

    nfl_badcase,his_badcase = 0,0
    nfl_badcase_logger, his_badcase_logger = {}, {}
    nfl_passage, his_passage = {},{}

    total_em,total_f1 = 0.0,0.0
    origin_em,origin_f1 = 0.0,0.0
    levelup_num = 0
    leveldown_num = 0
    total_num = 0
    for k ,v in case_logger_1.items():
        predicted_ability_str = v['predict_ability']
        cases = v['cases']
        predicted_answer = cases[method.index(predicted_ability_str) ]
        em,f1 = float(predicted_answer[0]),float(predicted_answer[1])
        origin_em += em
        origin_f1 += f1
        total_em += em
        total_f1 += f1

        changed = {}
        if  predicted_ability_str == 'program':
            questionId = k
            programs = [line[3] for line in v['answer_program']]
            predicted_answer = ""
            real_program = ""
            for program in programs:
                exe = executor(questionId = questionId, program=program,reader=reader)
                if_run, answer = exe.get_answer()
                if if_run:
                    predicted_answer = answer
                    real_program = program
                    #logger.info(answer)
                    break
            
            if predicted_answer == "":
                predicted_answer = cases[4][3] # count way answer
            
            answer_annotations = reader.questionId2answer_dict[questionId]
            ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in answer_annotations] 
            ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in answer_annotations]
            (exact_match, f1_score), max_type = metric_max_over_ground_truths(
                                drop_em_and_f1,
                                predicted_answer,
                                ground_truth_answer_strings,
                                ground_truth_answer_types,
                        )
            changed = {'em':exact_match,'f1':f1_score,'predicted_answer':predicted_answer,'real_program':real_program}
            if (f1_score >= f1 and exact_match >= em) and (f1_score > f1 or exact_match > em):
                levelup_num += 1
            elif (f1_score <= f1 and exact_match <= em) and (f1_score < f1 or exact_match < em):
                leveldown_num += 1
                output = generate_bad_case_str(questionId, reader)
                logger.info("origin_em:{},origin_f1:{}".format(round(em,4),round(f1,4)))
                logger.info("now_em:{},now_f1:{}".format(round(exact_match,4),round(f1_score,4)))
                logger.info(f"idx: {leveldown_num} output:\n{output['questionId']}\n{output['passage']}\nquestion:{output['question']}\nanswer:{output['answer']}\nanswer:{programs}\norigin_exe_answer:{cases[3][2]}\nnow_exe_answer:{predicted_answer} \nnumber2token:{output['number2token']}\nentity:{output['entity']}")


            total_num += 1
            total_em  = total_em - em + exact_match
            total_f1 = total_f1 - f1 + f1_score
            em,f1 = exact_match,f1_score
        
        max_f1 = max([line[1] for line in v['answer_program']])

        if max_f1 < 0.5 and predicted_ability_str == 'program':
            
            badcase_logger1[k] = generate_bad_case_str(k, reader)
            badcase_logger1[k]["predict_ability"] = predicted_ability_str
            badcase_logger1[k]["cases"] = cases
            badcase_logger1[k]["answer_program"] = v["answer_program"]
            badcase_logger1[k]['changed'] = changed
            passageId = reader.questionId2passageId(k)
            del badcase_logger1[k]["number2token"]
            del badcase_logger1[k]["entity"]


            passageId = reader.questionId2passageId(k)
            if "nfl" in passageId:
                nfl_badcase += 1
                nfl_badcase_logger[k] = badcase_logger1[k]
                nfl_passage[passageId] = True
            else :
                his_badcase += 1
                his_badcase_logger[k] = badcase_logger1[k]
                his_passage[passageId] = True
        elif f1 > 0.8 and predicted_ability_str == 'program':
            goodcase_logger1[k] = generate_bad_case_str(k, reader)
            goodcase_logger1[k]["predict_ability"] = predicted_ability_str
            goodcase_logger1[k]["cases"] = cases
            goodcase_logger1[k]["answer_program"] = v["answer_program"]
            goodcase_logger1[k]['changed'] = changed
            passageId = reader.questionId2passageId(k)
            del goodcase_logger1[k]["number2token"]
            del goodcase_logger1[k]["entity"]

            passageId = reader.questionId2passageId(k)
            if "nfl" in passageId:
                good_nfl_badcase += 1
                nfl_goodcase_logger[k] = goodcase_logger1[k]
                good_nfl_passage[passageId] = True
            else :
                good_his_badcase += 1
                his_goodcase_logger[k] = goodcase_logger1[k]
                good_his_passage[passageId] = True

    len_ = len(case_logger_1)
    print("origin_em:{},origin_f1:{}\ntotal_em:{},total_f1:{}\nlevelup_num:{},leveldown_num:{},total_num:{}".format(round(origin_em/len_,4),round(origin_f1/len_,4),round(total_em/len_,4),round(total_f1/len_,4),levelup_num,leveldown_num,total_num))


    savejson("logger/train_program_realbad_case_new.json",badcase_logger1)
    savejson("logger/train_nfl_program_realbad_case_new.json",nfl_badcase_logger)
    savejson("logger/train_his_program_realbad_case_new.json",his_badcase_logger)
    
    print(f"program realbadcase num: {len(badcase_logger1)}/{len(case_logger_1)}")
    print(f"program realbadcase: nfl:{nfl_badcase} his:{his_badcase}")
    print(f"program realbadcase passage: nfl:{len(nfl_passage)} his:{len(his_passage)}")

    savejson("logger/train_program_good_case_new.json",goodcase_logger1)
    savejson("logger/train_nfl_program_good_case_new.json",nfl_goodcase_logger)
    savejson("logger/train_his_program_good_case_new.json",his_goodcase_logger)
    
    print(f"program goodcase num: {len(goodcase_logger1)}/{len(case_logger_1)}")
    print(f"program goodcase: nfl:{good_nfl_badcase} his:{good_his_badcase}")
    print(f"program goodcase passage: nfl:{len(good_nfl_passage)} his:{len(good_his_passage)}")



useful_ner = ['ORDINAL','DATE', 'NUMBER', 'MONEY','DURATION','PERCENT','TIME']

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
#reader = drop_reader()
def generate_bad_case_str(questionId,reader):
    passageId = reader.questionId2passageId(questionId)
    add_number_token_passage = reader.passageId2add_number_token_passage(passageId)
    add_number_token_question = reader.questionId2add_number_token_question_dict[questionId]
    answer = reader.questionId2answer(questionId)
    answer = [ answer_json_to_strings(ans)[0] for ans in answer]
    number2token = reader.questionId2number2token(questionId)
    #number2token = { k:(str(v),str(type(v))) for k,v in number2token.items()}
    for k,v in number2token.items():
        v.version = 1
    number2token = { k:(str(v),str(type(v))) for k,v in number2token.items()}

    corenlp_passage = reader.passageId2corenlp_passage(passageId)
    entitys = []
    for sentence in corenlp_passage['sentences']:
        for id_entity, entity in enumerate(sentence['entitymentions']):
            if entity['ner'] in useful_ner:
                entitys.append(entity)
    

    add_number_token_passage_ids = tokenizer.encode(add_number_token_passage)
    add_number_token_question_ids = tokenizer.encode(add_number_token_question)
    add_number_token_question_ids[-1] = 0
    add_number_token_passage_ids = add_number_token_passage_ids[1:]
    question_passage_tokens = add_number_token_question_ids + add_number_token_passage_ids
    question_passage_tokens = question_passage_tokens[:512]
    model_question_passage = tokenizer.decode(question_passage_tokens)


    return {'questionId':questionId, 'passage':add_number_token_passage,'model_see':model_question_passage, 'question':add_number_token_question, 'answer': answer,'number2token': number2token, 'entity': entitys , }

def get_bad_case3(reader):
    path = "logger/QDGAT/NARoberta_case_logger.json"
    case_logger_3 = loadjson(path)
    badcase_logger3 = {}
    goodcase_logger3 = {}
    type_acc = {'span':(0.0,0.0,0.0,),'number':(0.0,0.0,0.0,),'spans':(0.0,0.0,0.0,),'date':(0.0,0.0,0.0,),'span+spans':(0.0,0.0,0.0,),'all':(0.0,0.0,0.0,)}
    for di in case_logger_3:
        di['type'] = di['answer_type']
        type_acc[di['type']] = ((type_acc[di['type']][0] + di['em']),(type_acc[di['type']][1] + di['f1']),(type_acc[di['type']][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + di['em']),(type_acc['all'][1] + di['f1']),(type_acc['all'][2] + 1),)
        if di['f1'] < 0.5:
            badcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)
        else :
            goodcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)

    type_acc['span+spans'] = type_acc['span']
    type_acc['span+spans'] = ( type_acc['span+spans'][0]+type_acc['spans'][0], type_acc['span+spans'][1]+type_acc['spans'][1], type_acc['span+spans'][2]+type_acc['spans'][2],)
    for k, v in type_acc.items():
        name = k
        logger.info("{}: em: {}/{} total: {}".format(name,round(v[0]/v[2]*100,2),round(v[1]/v[2]*100,2),round(v[2])))
        

from prettytable import PrettyTable


def eval(questionId, predicted_answer, reader):
    annotations = reader.questionId2answer_dict[questionId]
    answer_strings = [answer_json_to_strings(annotation)[0] for annotation in annotations]
    answer_types = [answer_json_to_strings(annotation)[1] for annotation in annotations]
    (exact_match, f1_score), max_type = metric_max_over_ground_truths(
        drop_em_and_f1,
        predicted_answer,
        answer_strings,
        answer_types,
    )
    return exact_match, f1_score, max_type
def run_program(questionId, program, reader):
    exe = executor(questionId = questionId, program = program, reader=reader)
    if_run, predicted_answer = exe.get_answer()
    return if_run, predicted_answer

def get_bad_case2(reader):


    #_name = "sel12_all_classifier_train_v136_debug_b"
    #_name = "sel12_all_classifier_train_v138_debug_b"
    _name = "sel12_all_classifier_train_v139_a" #正宗
    _name = "eval_checkpoint"

    #path = f"logger/{_name}/{_name}_case_logger_epoch600.json"
    #path = f"logger/{_name}/{_name}_case_logger_epoch400.json"
    path = f"logger/{_name}/{_name}_case_logger_epoch400.json" #正宗
    path = f"logger/{_name}/{_name}_case_logger_epoch4700.json" 

    os.system(f"hadoop fs -get logger/{_name}/ logger/")

    case_logger_1 = loadjson(path)

    path = "logger/new_NARoberta_small_output/QDGAT_case_logger.json"
    case_logger_3 = loadjson(path)

    path = "logger/new_QDGAT_small_output_4step/QDGAT_case_logger.json"
    case_logger_4 = loadjson(path)

    questionId2type =  loadjson("data/dev_questionId2program_type.json")
    questionId2program = loadjson("data/new_drop_dataset_annotation_v4_dev.json")

    #for k, type_ in questionId2type.items():
    #    if type_ =='count_operators':
    #        if "\"" in questionId2program[k]:
    #            questionId2type[k] = "count_hard_operators"

    #case_logger_2 = loadjson(f"logger/{_name}/{_name}_case_logger_epoch200.json")
    #case_logger_2 = loadjson(f"logger/{_name}/{_name}_case_logger_epoch100.json")
    case_logger_2 = loadjson(f"logger/{_name}/{_name}_case_logger_epoch100.json")

    questionId2answer_program = { questionId:cases['answer_program'] for questionId, cases in case_logger_2.items() if 'answer_program' in cases}

    badcase_logger1 = {}
    goodcase_logger1 = {}
    badcase_logger3 = {}
    goodcase_logger3 = {}
    #method2id = ["passagespan","questionspan","multispans",'count']
    method2id = ["passagespan","questionspan","multispans",'program','count']

    #method2id = ['program',]


    program_type = ['add_or_diff_operators','compare_operators','count_operators','complex_operators','other_operators','wa_hard',"EASY"]

    """
    for questionId, type_ in questionId2type.items():
        if type_ == 'add_or_diff_operators' or type_ == 'compare_operators':
            string = type_
            program = questionId2program[questionId]
            num = program.count("add") + program.count("diff") + program.count("max") + program.count("min")
            if num > 1:
                string = string + f"{num}_steps"
                if string not in program_type :
                    program_type.append(string)
                questionId2type[questionId] = string
    """

    #program_type = ['add_or_diff_operators','compare_operators','count_operators','complex_operators','other_operators','wa_hard',"EASY"]

    result = PrettyTable(["method",] + program_type + ["all",])
    
    baseline_dict = {di['questionId']:di for di in case_logger_3}

    num_program_type = {}
    for type_ in program_type:
        num_program_type[type_] = 0
    for di in case_logger_3:
        questionId = di['questionId']
        type_ = questionId2type[questionId]
        num_program_type[type_] += 1
    row = ["比例",]
    total = len(case_logger_3)
    for i in range(len(program_type)):
        type_ = program_type[i]
        row.append("{},{}".format(round(num_program_type[type_]*1.0/total,4), num_program_type[type_]))
    row.append(str(total))
    result.add_row(row)

    output = []

    type_acc = {}
    for key in program_type:
        type_acc[key] = (0.0,0.0,0,)
    type_acc['all'] = (0.0,0.0,0,)

    #nesy
    for k ,v in tqdm(case_logger_1.items()):
        cases = v['cases']
        index = method2id.index(v['predict_ability'])
        #index = method2id.index("count")
        answer_type = v['ground_truth_answer_types']
        
        case = cases[index]
        if v['predict_ability'] == 'program':
            _, case[2] = run_program(k,case[3],reader)
        em,f1 = float(case[0]),float(case[1])
        
        em,f1,_ = eval(questionId=k,reader=reader,predicted_answer=case[2])            

        questionId_type = questionId2type[k]

        type_acc[questionId_type] = ((type_acc[questionId_type][0] + em),(type_acc[questionId_type][1] + f1),(type_acc[questionId_type][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + em),(type_acc['all'][1] + f1),(type_acc['all'][2] + 1),)


    
    logger.info("nesy:")
    row = ["nesy",]
    for k, v in type_acc.items():
        name = k
        method_result = "{}".format(round(v[1]/v[2]*100,2))
        logger.info(method_result)
        row.append(method_result)
    result.add_row(row)


    type_acc = {}
    for key in program_type:
        type_acc[key] = (0.0,0.0,0,)
    type_acc['all'] = (0.0,0.0,0,)



    for k ,v in tqdm(case_logger_1.items()):
        cases = v['cases']
        index = method2id.index("program")
        #index = method2id.index("program")
        case = cases[index]
        _, case[2] = run_program(k,case[3],reader)
        em,f1 = float(case[0]),float(case[1])
        em,f1,_ = eval(questionId=k,reader=reader,predicted_answer=case[2])

        questionId_type = questionId2type[k]

        type_acc[questionId_type] = ((type_acc[questionId_type][0] + em),(type_acc[questionId_type][1] + f1),(type_acc[questionId_type][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + em),(type_acc['all'][1] + f1),(type_acc['all'][2] + 1),)


        if f1 < 0.5:
            badcase_logger1[k] = (float(case[0]),float(case[1]),case[2],v['predict_ability'])
            output_dict = generate_bad_case_str(k, reader)
            output_dict['cases'] = case_logger_1[k]['cases']
            output_dict['predict_ability'] = v['predict_ability']
            del output_dict['entity']
            del output_dict['number2token']
            if k in questionId2answer_program:
                output_dict["answer_program"]  = questionId2answer_program[k]
            output.append(output_dict)
        else :
            goodcase_logger1[k] = (float(case[0]),float(case[1]),case[2],v['predict_ability'])
    
    logger.info("bart:")
    row = ["bart",]
    for k, v in type_acc.items():
        name = k
        method_result = "{}".format(round(v[1]/v[2]*100,2))
        logger.info(method_result)
        row.append(method_result)
    result.add_row(row)

    ##################


    type_acc = {}
    for key in program_type:
        type_acc[key] = (0.0,0.0,0,)
    type_acc['all'] = (0.0,0.0,0,)
    
    case_logger_4_dict = { di['questionId']: di for di in case_logger_4}
    for di in case_logger_4:
        
        #if di['questionId'] not in questionId2type:
        #    continue
        questionId_type = questionId2type[di['questionId']]
        di['em'],di['f1'] = di['cases'][0],di['cases'][1]
        di['type'] = questionId_type
        di['em'],di['f1'] = float(di['em']),float(di['f1'])
        di['em'],di['f1'],_ = eval(questionId=di['questionId'], reader=reader, predicted_answer=di['predicted_answer'])

        type_acc[di['type']] = ((type_acc[di['type']][0] + di['em']),(type_acc[di['type']][1] + di['f1']),(type_acc[di['type']][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + di['em']),(type_acc['all'][1] + di['f1']),(type_acc['all'][2] + 1),)
        if di['f1'] < 0.5:
            badcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)
        else :
            goodcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)
            
    logger.info("QDGAT")
    row = ["QDGAT",]
    for k, v in type_acc.items():
        name = k
        method_result = "{}".format(round(v[1]/v[2]*100,2))
        logger.info(method_result)
        row.append(method_result)
    result.add_row(row)

    type_acc = {}
    for key in program_type:
        type_acc[key] = (0.0,0.0,0,)
    type_acc['all'] = (0.0,0.0,0,)
    
    final_addordiff_badcases = []
    case_logger_3_dict = { di['questionId']: di for di in case_logger_3}
    for di in case_logger_3:
        
        #if di['questionId'] not in questionId2type:
        #    continue
        questionId_type = questionId2type[di['questionId']]
        di['em'],di['f1'] = di['cases'][0],di['cases'][1]
        di['type'] = questionId_type
        di['em'],di['f1'] = float(di['em']),float(di['f1'])
        di['em'],di['f1'],_ = eval(questionId=di['questionId'], reader=reader, predicted_answer=di['predicted_answer'])

        type_acc[di['type']] = ((type_acc[di['type']][0] + di['em']),(type_acc[di['type']][1] + di['f1']),(type_acc[di['type']][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + di['em']),(type_acc['all'][1] + di['f1']),(type_acc['all'][2] + 1),)
        if di['f1'] < 0.5:
            badcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)
        else :
            goodcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)
        

    #type_acc['span+spans'] = type_acc['span']
    #type_acc['span+spans'] = ( type_acc['span+spans'][0]+type_acc['spans'][0], type_acc['span+spans'][1]+type_acc['spans'][1], type_acc['span+spans'][2]+type_acc['spans'][2],)
    
    logger.info("baseline")
    row = ["baseline",]
    for k, v in type_acc.items():
        name = k
        method_result = "{}".format(round(v[1]/v[2]*100,2))
        logger.info(method_result)
        row.append(method_result)
    result.add_row(row)

    logger.info(result)

    exit(0)
    from drop_eval import logger_for_badcase
    for example in logger_for_badcase:
        example[0],example[1] = list(example[0]),list(example[1])
    savejson("logger/eval_badcase.json",logger_for_badcase)
    print(len(logger_for_badcase))


    exit(0)

    output = []
    for questionId in final_addordiff_badcases:
        k = questionId
        di = case_logger_3_dict[questionId]
        
        output_dict = generate_bad_case_str(k, reader)
        output_dict['annotated_program'] = reader.questionId2program_dict[k]
        output_dict['cases'] = case_logger_1[k]['cases']
        output_dict['predict_ability'] = case_logger_1[k]['predict_ability']
        del output_dict['entity']
        del output_dict['number2token']
        if k in questionId2answer_program:
            output_dict["answer_program"]  = questionId2answer_program[k]
        output_dict['qdgat'] = di
        output.append(output_dict)
    output = random.sample(output, 30)
    savejson("logger/addordiff_badcases.json",output)
    print(len(output))
    exit(0)



    num_bad_case = 0
    output = []
    good_output = []
    num_good_case = 0
    

    output_bad_case = {}
    output_bad_case['qdgat answer type'] = {}
    output_bad_case['our answer type'] = {}
    for k in tqdm(case_logger_1.keys()):
        if k in badcase_logger1 :
            if k not in badcase_logger3:
                num_bad_case += 1
                output_dict = generate_bad_case_str(k, reader)
                output_dict['cases'] = case_logger_1[k]['cases']
                output_dict['predicted_ability_str'] = case_logger_1[k]['predict_ability']
                output_dict['qdgat'] = goodcase_logger3[k]
                del output_dict['entity']
                del output_dict['number2token']
                if 'answer_program' in case_logger_1[k]:
                    output_dict["answer_program"]  = case_logger_1[k]['answer_program']
                output.append(output_dict)


        if k in badcase_logger3 :
            if k not in badcase_logger1 and k in goodcase_logger1: 
                num_good_case += 1
                output_dict = generate_bad_case_str(k, reader)
                output_dict['cases'] = case_logger_1[k]['cases']
                output_dict['predicted_ability_str'] = case_logger_1[k]['predict_ability']
                output_dict['qdgat'] = badcase_logger3[k]
                del output_dict['entity']
                del output_dict['number2token']
                if 'answer_program' in case_logger_1[k]:
                    output_dict["answer_program"]  = case_logger_1[k]['answer_program']
                good_output.append(output_dict)
    #savejson(f"logger/bad_case_for_QDGAT_and_{_name}.json",output)
    #savejson(f"logger/good_case_for_QDGAT_and_{_name}.json",good_output)
    
    #logger.info(f"output_bad_case: {output_bad_case}")

    #logger.info("num_real_bad_case: {}".format(num_bad_case))
    #logger.info("num_all_bad_case: {}".format(len(badcase_logger1)))

    #logger.info("num_real_good_case: {}".format(num_good_case))
    #logger.info("num_qdgat_bad_case: {}".format(len(badcase_logger3)))

def get_new_eval_result(reader):
    _name = "sel12_all_classifier_train_v139_a"
    path = f"logger/{_name}/{_name}_case_logger_epoch400.json"

    os.system(f"hadoop fs -get logger/{_name}/ logger/")

    case_logger_1 = loadjson(path)

    path = "logger/new_NARoberta_small_output/QDGAT_case_logger.json"
    case_logger_3 = loadjson(path)

    questionId2type =  loadjson("data/dev_questionId2program_type.json")
    questionId2program = loadjson("data/new_drop_dataset_annotation_v4_dev.json")

    case_logger_2 = loadjson(f"logger/{_name}/{_name}_case_logger_epoch100.json")

    questionId2answer_program = { questionId:cases['answer_program'] for questionId, cases in case_logger_2.items() if 'answer_program' in cases}

    badcase_logger1 = {}
    goodcase_logger1 = {}
    badcase_logger3 = {}
    goodcase_logger3 = {}
    method2id = ["passagespan","questionspan","multispans",'program','count']

    #method2id = ['program',]


    program_type = ['add_or_diff_operators','compare_operators','count_operators','complex_operators','other_operators','wa_hard',"EASY"]

    result = PrettyTable(["method",] + program_type + ["all",])
    
    baseline_dict = {di['questionId']:di for di in case_logger_3}

    num_program_type = {}
    for type_ in program_type:
        num_program_type[type_] = 0
    for di in case_logger_3:
        questionId = di['questionId']
        type_ = questionId2type[questionId]
        num_program_type[type_] += 1
    row = ["比例",]
    total = len(case_logger_3)
    for i in range(len(program_type)):
        type_ = program_type[i]
        row.append("{},{}".format(round(num_program_type[type_]*1.0/total,4), num_program_type[type_]))
    row.append(str(total))
    result.add_row(row)

    output = []

    type_acc = {}
    for key in program_type:
        type_acc[key] = (0.0,0.0,0,)
    type_acc['all'] = (0.0,0.0,0,)

    #nesy
    for k ,v in tqdm(case_logger_1.items()):
        cases = v['cases']
        index = method2id.index(v['predict_ability'])
        #index = method2id.index("count")
        answer_type = v['ground_truth_answer_types']
        
        case = cases[index]
        if v['predict_ability'] == 'program':
            _, case[2] = run_program(k,case[3],reader)
        em,f1 = float(case[0]),float(case[1])
        
        em,f1,_ = eval(questionId=k,reader=reader,predicted_answer=case[2])            

        questionId_type = questionId2type[k]

        type_acc[questionId_type] = ((type_acc[questionId_type][0] + em),(type_acc[questionId_type][1] + f1),(type_acc[questionId_type][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + em),(type_acc['all'][1] + f1),(type_acc['all'][2] + 1),)


    
    logger.info("nesy:")
    row = ["nesy",]
    for k, v in type_acc.items():
        name = k
        method_result = "{}".format(round(v[1]/v[2]*100,2))
        logger.info(method_result)
        row.append(method_result)
    result.add_row(row)


    type_acc = {}
    for key in program_type:
        type_acc[key] = (0.0,0.0,0,)
    type_acc['all'] = (0.0,0.0,0,)



    for k ,v in tqdm(case_logger_1.items()):
        cases = v['cases']
        index = method2id.index("program")
        #index = method2id.index("program")
        case = cases[index]
        _, case[2] = run_program(k,case[3],reader)
        em,f1 = float(case[0]),float(case[1])
        em,f1,_ = eval(questionId=k,reader=reader,predicted_answer=case[2])

        questionId_type = questionId2type[k]

        type_acc[questionId_type] = ((type_acc[questionId_type][0] + em),(type_acc[questionId_type][1] + f1),(type_acc[questionId_type][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + em),(type_acc['all'][1] + f1),(type_acc['all'][2] + 1),)


        if f1 < 0.5:
            badcase_logger1[k] = (float(case[0]),float(case[1]),case[2],v['predict_ability'])
            output_dict = generate_bad_case_str(k, reader)
            output_dict['cases'] = case_logger_1[k]['cases']
            output_dict['predict_ability'] = v['predict_ability']
            del output_dict['entity']
            del output_dict['number2token']
            if k in questionId2answer_program:
                output_dict["answer_program"]  = questionId2answer_program[k]
            output.append(output_dict)
        else :
            goodcase_logger1[k] = (float(case[0]),float(case[1]),case[2],v['predict_ability'])
    
    logger.info("bart:")
    row = ["bart",]
    for k, v in type_acc.items():
        name = k
        method_result = "{}".format(round(v[1]/v[2]*100,2))
        logger.info(method_result)
        row.append(method_result)
    result.add_row(row)

    type_acc = {}
    for key in program_type:
        type_acc[key] = (0.0,0.0,0,)
    type_acc['all'] = (0.0,0.0,0,)
    
    final_addordiff_badcases = []
    case_logger_3_dict = { di['questionId']: di for di in case_logger_3}
    for di in case_logger_3:
        
        #if di['questionId'] not in questionId2type:
        #    continue
        questionId_type = questionId2type[di['questionId']]
        di['em'],di['f1'] = di['cases'][0],di['cases'][1]
        di['type'] = questionId_type
        di['em'],di['f1'] = float(di['em']),float(di['f1'])
        di['em'],di['f1'],_ = eval(questionId=di['questionId'], reader=reader, predicted_answer=di['predicted_answer'])

        type_acc[di['type']] = ((type_acc[di['type']][0] + di['em']),(type_acc[di['type']][1] + di['f1']),(type_acc[di['type']][2] + 1),)
        type_acc['all'] = ((type_acc['all'][0] + di['em']),(type_acc['all'][1] + di['f1']),(type_acc['all'][2] + 1),)
        if di['f1'] < 0.5:
            badcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)
        else :
            goodcase_logger3[di['questionId']] = (di['em'],di['f1'],di['predicted_answer'],di['predicted_ability_str'],)
        

    #type_acc['span+spans'] = type_acc['span']
    #type_acc['span+spans'] = ( type_acc['span+spans'][0]+type_acc['spans'][0], type_acc['span+spans'][1]+type_acc['spans'][1], type_acc['span+spans'][2]+type_acc['spans'][2],)
    
    logger.info("baseline")
    row = ["baseline",]
    for k, v in type_acc.items():
        name = k
        method_result = "{}".format(round(v[1]/v[2]*100,2))
        logger.info(method_result)
        row.append(method_result)
    result.add_row(row)

    logger.info(result)




def get_count_badcase(reader):
    _name = "sel12_all_classifier_train_v139_a"
    path = f"logger/{_name}/{_name}_case_logger_epoch400.json"
    case_logger_1 = loadjson(path)

    path = "logger/new_NARoberta_small_output/QDGAT_case_logger.json"
    case_logger_3 = loadjson(path)



    questionId2type =  loadjson("data/dev_questionId2program_type.json")
    #questionId2type = loadjson("data/questionId2type.json")
    questionId2program = loadjson("data/new_drop_dataset_annotation_v4_dev.json")

    methods = ['passagespan','questionspan','multispans','program','count']

    case_logger_3_dict = { di['questionId']: di for di in case_logger_3}
    for di in case_logger_3:
        questionId_type = questionId2type[di['questionId']]

        di['type'] = questionId_type
        #di['em'],di['f1'] = float(di['em']),float(di['f1'])
        di['em'],di['f1'],_ = eval(questionId=di['questionId'], reader=reader, predicted_answer=di['predicted_answer'])
    output = []


    for k,v in case_logger_1.items():
        questionId = k
        question_type = questionId2type[k]
        
        if v['predict_ability'] != "program":
            continue
        index = methods.index(v['predict_ability'])
        #if v['cases'][3][3] != questionId2program[k]:
        #    continue

        #if questionId2type[k] not in ['number']:
        #    continue
        #if questionId2type[k] != "count_operators":
        #    continue

        if float(v['cases'][index][1]) >= 0.5:
            continue


        output_dict = generate_bad_case_str(k, reader)


        ways_ = {}
        for idx, method in enumerate(methods):
            ways_[method] = case_logger_1[k]['cases'][idx]

        output_dict['cases'] = ways_
        output_dict['predict_ability'] = v['predict_ability']
        output_dict['type'] = question_type
        if k in questionId2program:
            output_dict['manual program'] = questionId2program[k]
        output_dict['NARoberta'] = case_logger_3_dict[k]
        #output_dict['FLAG_for_match'] = "MATCH!" if case[3] == questionId2program[k] else "NOMATCH"
        del output_dict['entity']
        del output_dict['number2token']
        output.append(output_dict)
        
    
    print(len(output))
    random.seed(233377)
    #output = random.sample(output, 50)
    savejson("bad_case_nesy.json",output)
    #print(right_count)
    #print(total_count)
    


    

def little_program(l1,l2):
    a,b,c = l1[0],l1[1],l1[2]
    d,e,f = l2[0],l2[1],l2[2]
    a*=c
    b*=c

    d*=f
    e*=f

    a+=d
    b+=e
    c+=f
    return [a/c,b/c,c,]

def little_program2():
    string = "''number': [0.7782859078590786, 0.7782859078590786, 5904], 'spans': [0.0, 0.20764822134387334, 506], 'span': [0.5709470845972363, 0.647020559487698, 2967], 'date': [0.25157232704402516, 0.39194968553459136, 159], 'total_em': 0.6636954697986577, 'total_f1': 0.7007235738255035, 'total_count': 9536"
    num_list = getNumberFromString(string)
    num_list = [ num for num in num_list if (abs(num-1.0))>1e-5]
    number = num_list[0:3]
    spans = num_list[3:6]
    span = num_list[6:9]
    date = num_list[9:12]
    total = num_list[12:15]
    span_spans = little_program(span,spans)

    print(f"{round(number[0]*100,2)} {round(number[1]*100,2)} {round(date[0]*100,2)} {round(date[1]*100,2)} {round(span_spans[0]*100,2)} {round(span_spans[1]*100,2)}")
    print(f"{round(total[0]*100,2)} {round(total[1]*100,2)}")

def little_program3():
    passageId = "history_1458"
    number2token = reader.passageId2number2token(passageId)
    number2token = {k:str(v) for k,v in number2token.items()}
    logger.info(number2token)

def little_program4():
    nfl = set()
    history = set()
    di = loadjson("logger/bad_case_for_QDGAT_and_rel12_all_rel12_all_app_v2.json")
    for k in di:
        questionId = k['questionId']
        passageId = reader.questionId2passageId(questionId)
        if "nfl" in passageId:
            nfl.add(passageId)
        else :
            history.add(passageId)

    print(len(di))
    print(len(nfl))
    print(len(history))

def test_number2token(reader):
    questionId = "eab026ed-ad3b-4e18-b20a-c0b4897674e0"
    passageId = reader.questionId2passageId(questionId)
    corenlp_passage = reader.passageId2corenlp_passage(passageId)
    new_number2token(corenlp_passage, 'N',debug_mode = True)


import re
def add_string(s):
    if "\"" in s:        
        s = re.sub("\"","\"\"",s)
    s = "\"" + s + "\""
    return s
def csv_line(elements):
    elements = [add_string(str(s)) for s in elements]
    line = ','.join(elements+['',])
    return line

def generate_additional_questionId2program_csv(reader):
    additional_questionId2program_path = "data/additional_questionId2program.json"
    additional_questionId2program = loadjson(additional_questionId2program_path)

    original_train_dataset_path = "data/drop_dataset_train.json"
    original_train_dataset = loadjson(original_train_dataset_path)

    questionId2program_path = "data/new_drop_dataset_annotation.json"
    questionId2program = loadjson(questionId2program_path)

    lines = []
    lines.append(csv_line(['passageId','passage','questionId','question','reference answer','annotation1','annotation2','annotation3']))

    for passageId, text in tqdm(original_train_dataset.items()):
        
        num_valid_question = 0
        qa_pairs = text['qa_pairs']
        for qa_pair in qa_pairs:
            questionId = qa_pair['query_id']
            if questionId in additional_questionId2program and additional_questionId2program[questionId]:
                num_valid_question += 1
        #if num_valid_question >= 1:
        passage = reader.passageId2add_number_token_passage(passageId)
        lines.append(csv_line([passageId,passage,]))
        for qa_pair in qa_pairs:
            questionId = qa_pair['query_id']

            num_valid_question += 1
            question = reader.questionId2add_number_token_question_dict[questionId]
            answer,_ = answer_text(qa_pair['answer'])
            
            FLAG = ""
            if questionId in questionId2program:
                if questionId2program[questionId] and isinstance(questionId2program[questionId],str):
                    FLAG = "ORIGINAL_EXAMPLE"

            if questionId in additional_questionId2program and additional_questionId2program[questionId]:
                annotations = additional_questionId2program[questionId][:3]
                annotations = [annotation[1] for annotation in annotations]
                while len(annotations) < 3:
                    annotations.append('')
            else:
                annotations = ['','','',]
            lines.append(csv_line(["","",questionId,question,answer,annotations[0],annotations[1],annotations[2],FLAG,]))
    with open("logger/additional_questionId2program.csv",mode='w') as f:
        for line in lines:
            f.write(line+ "\n")

def get_question_type(reader):
    num_question = 0
    num = [0,0,0,0,0]


    for questionId, question in reader.questionId2add_number_token_question_dict.items():
        if re.match('^how many[\ a-z0-9\@]*longest[\ a-z0-9\@]*shortest',question.lower()): #longest than shortest question
            num[1] += 1
        elif re.match('^how many yard[\ a-z0-9\@]*',question.lower()) or re.match('^how many longer yard[\ a-z0-9\@]*',question.lower()):
            num[2] += 1
        elif re.match('^how many [\ a-z0-9\@]*',question.lower()) :
            num[3] += 1
        elif re.match('[\ a-z0-9\@\?\,]*if[\ a-z0-9\@]*win[\ a-z0-9\@]*game',question.lower()):
            num[4] += 1
        else :
            pass
        num_question += 1
    
    print(num)
    print(num_question)

def put_in_data(reader,origin_file_name,output_file_name):
    import csv
    questionId_index = 3
    annotation_index = 6
    date_index = 9
    people_index = 7
    reference_answer_index = 5

    with open(origin_file_name,encoding='utf-8-sig') as f:
        csvreader = csv.reader(f)
        header_row = next(csvreader)
        annotations = {line[questionId_index]:line[annotation_index] for line in csvreader if line[questionId_index]!="" and line[people_index]!="" and line[annotation_index]!=""}
    annotations = {k:v for k,v in annotations.items() if v!="" and v.lower().strip() not in ['easy','wa','hard']}
    runok_annotations = {}
    for questionId, annotation in annotations.items():
        exe = executor(annotation, questionId, reader)
        if_run, answer = exe.get_answer()
        if if_run:
            runok_annotations[questionId] = annotation
    
    logger.info("size of new anotations:")
    logger.info(len(runok_annotations))
    #logger.info(annotations)
    savejson(output_file_name,runok_annotations)

def generate_additional_questionId2program(reader):
    path = "logger/data_argument/data_argument_case_logger_epoch0.json"
    case_logger1 = loadjson(path)

    ok_questionId = 0
    error_questionId = 0
    right_level_question = [[],[],[],[],[],[], ]
    right_level = [0.9,0.8,0.6,0.4,0.2,0.0 ]
    #90% 80% 60% 40% 20%
    additional_questionId2program = {}
    


    for questionId , v in tqdm(case_logger1.items()):
        answer_programs = v['answer_program']
        Flag = 0
        for line in answer_programs:
            em,f1 = line[0],line[1]
            program, score = line[3],line[5][0]
            exe = executor(questionId=questionId, reader=reader, program= program)
            if_run, answer = exe.get_answer()
            answer_annotations = reader.questionId2answer_dict[questionId]
            ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in answer_annotations] 
            ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in answer_annotations]
            (exact_match, f1_score), max_type = metric_max_over_ground_truths(
                drop_em_and_f1,
                answer,
                ground_truth_answer_strings,
                ground_truth_answer_types,
            )
            if f1_score >= 0.66:
                Flag = 1
                additional_questionId2program[questionId] = program
                for i in range(6):
                    if math.exp(score) > right_level[i]:
                        right_level_question[i].append(questionId)
                        break
                break
        if Flag:
            ok_questionId += 1
        else :
            error_questionId += 1
    questionId2program = {questionId:program for questionId,program in additional_questionId2program.items() if questionId in right_level_question[0] }
    questionId2program_level1 = {questionId:program for questionId,program in additional_questionId2program.items() if questionId in right_level_question[1] }
    questionId2program_level2 = {questionId:program for questionId,program in additional_questionId2program.items() if questionId in right_level_question[2] }
    questionId2program_level3 = {questionId:program for questionId,program in additional_questionId2program.items() if questionId in right_level_question[3] }

    

    logger.info(f"ok_question:{ok_questionId} error_question:{error_questionId}")
    logger.info(f"whole_question: {len(case_logger1)}")
    logger.info(f"level question: {[len(line) for line in right_level_question]}")





    #savejson("logger/additional_questionId2program_level0.json", questionId2program)
    #savejson("logger/additional_questionId2program_level1.json", questionId2program_level1)
    #savejson("logger/additional_questionId2program_level2.json", questionId2program_level2)
    #savejson("logger/additional_questionId2program_level3.json", questionId2program_level3)

def put_in_data2(reader,origin_file_name,output_file_name):
    import csv
    annotations = {}
    import json
    with open(origin_file_name,encoding='utf-8-sig') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            peo = line[3]
            main_ = json.loads(line[4])
            if 'chadlzx' in peo:
                continue
            #logger.info(main_)
            for dd in main_['mainForm']:
                questionId = dd['name']
                annotation = dd['value']
                if annotation == None:
                    annotation = ""
                if questionId in annotations :
                    logger.info("questionId have logged! original log is : {} : {}".format(questionId, annotations[questionId]))
                annotations[questionId] = annotation
    
    runok_annotations = {}
    notok_annotations = []

    wa_num = 0
    for questionId, annotation in annotations.items():
        exe = executor(annotation, questionId, reader)
        if annotation == 'EASY' or annotation == 'HARD':
            notok_annotations.append(annotation)
            runok_annotations[questionId] = "NULL"
            continue

        if_run, answer = exe.get_answer()
        if if_run:
            runok_annotations[questionId] = annotation
        else :
            #print(annotation)
            if annotation == "WA":
                wa_num += 1

    logger.info("size of new anotations:")
    logger.info(len(runok_annotations))
    logger.info(len(annotations))
    logger.info(wa_num)
    #logger.info(annotations)
    savejson(output_file_name,runok_annotations)

def little1(reader):
    dd1 = loadjson("data/new_drop_dataset_annotation.json")
    dd2 = loadjson("data/drop_dataset_annotation.json")
    for k,v in dd2.items():
        if not isinstance(v, str):
            if k in dd1:
                print("Error!")
            dd1[k] = "NULL"
    savejson("logger/new_drop_dataset_annotation.json",dd1)

def little2(reader):
    import csv
    dd2 = loadjson("data/new_drop_dataset_annotation_v2.json")
    with open("data/drop_pre_annotation_version2.csv") as f:
        csvreader = csv.reader(f)
        header_row = next(csvreader)
        for line in csvreader:
            #print(line)
            k = line[3]
            v = line[6]
            if v == 'HARD' or v == 'EASY':
                if k in dd2:
                    print("ERROR")
                dd2 [k] = 'NULL'
    savejson("logger/new_drop_dataset_annotation_v2.json",dd2)


def little3(reader):
    reader.questionId2program_dict = { k:v for k,v in reader.questionId2program_dict.items() if v!=None and v !="NULL"}
    print(len(reader.questionId2program_dict))

def test_mydataset(reader):
    original_annotation = loadjson("data/new_drop_dataset_annotation.json")
    annotations_v2 = loadjson("data/new_drop_dataset_annotation_v2.json")
    annotations_v3 = loadjson("data/new_drop_dataset_annotation_v3.json")
    original_annotation = dict(original_annotation,**annotations_v2)
    original_annotation = dict(original_annotation,**annotations_v3)
    
    typeNum = {'number':0, 'span':0, 'spans':0, 'date':0}
    for k,v in original_annotation.items():
        questionId = k
        if v == "NULL":
            continue
        answer = reader.questionId2answer_dict[questionId][0]
        result = answer_json_to_strings(answer)
        typeNum[result[1]] += 1
    sum = typeNum['number'] + typeNum['span'] + typeNum['spans'] + typeNum['date'] 
    print({k:round(v*1.0/sum,4) for k,v in typeNum.items()})
    print( typeNum)
    print(sum)

    typeNum = {'number':0, 'span':0, 'spans':0, 'date':0}
    for k,v in reader.questionId2passageId_dict.items():
        questionId = k
        if len(reader.questionId2answer_dict[questionId]) < 1 :
            continue
        answer = reader.questionId2answer_dict[questionId][0]
        result = answer_json_to_strings(answer)
        typeNum[result[1]] += 1
    sum = typeNum['number'] + typeNum['span'] + typeNum['spans'] + typeNum['date'] 
    print({k:round(v*1.0/sum,4) for k,v in typeNum.items()})
    print( typeNum)
    print(sum)






def little4(reader):
    import random
    questionId2program = loadjson("data/additional_questionId2program_level0.json")
    questionId2program_level1 = loadjson("data/additional_questionId2program_level1.json")
    questionId2program_level2 = loadjson("data/additional_questionId2program_level2.json")
    questionId2program_level3 = loadjson("data/additional_questionId2program_level3.json")
    random.seed(1453)

    questionId2program = questionId2program

    questionIds = random.sample(list(questionId2program), 50)
    cases = {}
    for questionId in questionIds:
        output = generate_bad_case_str(reader=reader, questionId=questionId)
        output['program'] = questionId2program[questionId]

        program = questionId2program[questionId]
        exe = executor(questionId=questionId, reader=reader, program= program)
        if_run, answer = exe.get_answer()

        output['program answer'] = answer
        del output['number2token']
        del output['entity']
        
        cases[questionId] = output 

    savejson("logger/test.json", cases)


def little5(reader):
    original_annotation = loadjson("data/new_drop_dataset_annotation.json")
    annotations_v2 = loadjson("data/new_drop_dataset_annotation_v2.json")
    annotations_v3 = loadjson("data/new_drop_dataset_annotation_v3.json")
    original_annotation = dict(original_annotation,**annotations_v2)
    original_annotation = dict(original_annotation,**annotations_v3)

    failed_questionId2program = {}
    error_questionId2program = {}

    for questionId, program in tqdm(original_annotation.items()):
        if program in ["EASY","WA","HARD","NULL"]:
            continue
        exe = executor(program = program, reader = reader, questionId = questionId)
        if_run, answer = exe.get_answer()
        if if_run is False:
            error_questionId2program[questionId] = program
            continue
        answer_annotations = reader.questionId2answer_dict[questionId]
        ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in answer_annotations] 
        ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in answer_annotations]
        (exact_match, f1_score), max_type = metric_max_over_ground_truths(
            drop_em_and_f1,
            answer,
            ground_truth_answer_strings,
            ground_truth_answer_types,
        )
        if f1_score < 0.66:
            failed_questionId2program[questionId] = program
    savejson("logger/failed_questionId2program.json",failed_questionId2program)
    savejson("logger/error_questionId2program.json",error_questionId2program)

def little6(reader):
    failed_questionId2program = loadjson("logger/failed_questionId2program.json")
    failed_questionId2program_badcase = []

    helped_ = 0
    for questionId , program in tqdm(failed_questionId2program.items()):
        output = generate_bad_case_str(reader=reader, questionId=questionId)
        
        exe = executor(questionId=questionId, reader=reader, program= program)
        if_run, answer = exe.get_answer()

        answer_annotations = reader.questionId2answer_dict[questionId]
        ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in answer_annotations] 
        ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in answer_annotations]
        (exact_match, f1_score), max_type = metric_max_over_ground_truths(
            drop_em_and_f1,
            answer,
            ground_truth_answer_strings,
            ground_truth_answer_types,
        )
        if f1_score > 0.66:
            helped_ += 1


        output['program'] = failed_questionId2program[questionId]
        output['program answer'] = answer
        for k, v in output.items():
            logger.info(f"{k}:{v}")
        logger.info("")
        failed_questionId2program_badcase.append(output)
        

    print(f"helped_ questionId2program:{helped_}")
    savejson("logger/failed_questionId2program_badcase.json", failed_questionId2program_badcase)

def little7(reader):
    name = 'bel12_program_debug_v81'
    path = f"logger/{name}/{name}_case_logger_epoch50.json"
    case_logger = loadjson(path)

    for questionId, case in case_logger.items():
        predict_ability = case['predict_ability']
        em,f1,answer,program = case['cases'][0]
        em,f1 = float(em),float(f1)
        if 'number' in case['ground_truth_answer_types'] and f1 < 0.66:
            output = generate_bad_case_str(questionId, reader)
            output['program answer'] = answer
            output['program'] = program
            output['result'] = (round(em,4),round(f1,4),)
            del output['entity']
            del output['number2token']
            for k,v in output.items():
                logger.info(f"{k}:{v}")
            logger.info("")


def test_my_executor(reader):
    name = "bel12_program_v84_debug"
    epoch = "20"
    eval_result = f"logger/{name}/{name}_eval_result_epoch{epoch}.json"
    case_logger = f"logger/{name}/{name}_case_logger_epoch{epoch}.json"

    eval_result = loadjson(eval_result)
    case_logger = loadjson(case_logger)
    type_acc = {'span':(0.0,0.0,0.0,),'number':(0.0,0.0,0.0,),'spans':(0.0,0.0,0.0,),'date':(0.0,0.0,0.0,),'all':(0.0,0.0,0.0,)}

    total_f1,total_em,num =0,0,0

    num_good = 0
    num_bad = 0
    for questionId, case in case_logger.items():
        answer_program = case['answer_program']
        original_em,original_f1 = case['cases'][0][0], case['cases'][0][1]
        original_em,original_f1 = float(original_em),float(original_f1)
        f1,em = 0,0
        type_ = "span"
        program_ = "Empty"
        answer_ = "Empty"
        for program_dict in answer_program:
            program = program_dict[3]
            exe = executor(program = program, questionId = questionId, reader = reader)
            if_run, predicted_answer = exe.get_answer()
            answer_annotations = reader.questionId2answer_dict[questionId]

            ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in answer_annotations] 
            ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in answer_annotations]
            (exact_match, f1_score), max_type = metric_max_over_ground_truths(
                                drop_em_and_f1,
                                predicted_answer,
                                ground_truth_answer_strings,
                                ground_truth_answer_types,
                        )
            type_ = max_type
            if if_run:
                program_ = program
                answer_ = predicted_answer
                f1 = f1_score
                em = exact_match
                break
        a,b,c = type_acc[type_]
        a += em
        b += f1
        c += 1
        type_acc[type_] = (a,b,c,)

        a,b,c = type_acc['all']
        a += em
        b += f1
        c += 1
        type_acc['all'] = (a,b,c,)

        if f1 >= original_f1 and em >= original_em and (f1 > original_f1 or em > original_em):
            logger.info("goodcase!")
            logger.info(f"origin em/f1:{original_em}/{original_f1}")
            logger.info(f"origin :{case['cases'][0][3]} answer:{case['cases'][0][2]}")
            logger.info("")
            logger.info(f"now em/f1:{em}/{f1}")
            logger.info(f"now :{program_} answer:{answer_}")
            logger.info("\n")
            num_good += 1
        if f1 <= original_f1 and em <= original_em and (f1 < original_f1 or em < original_em):
            logger.info("badcase!")
            logger.info(f"origin em/f1:{original_em}/{original_f1}")
            logger.info(f"origin :{case['cases'][0][3]} answer:{case['cases'][0][2]}")
            logger.info("")
            logger.info(f"now em/f1:{em}/{f1}")
            logger.info(f"now :{program_} answer:{answer_}")
            logger.info("\n")
            num_bad += 1
        
        

    def fix(a,b,c):
        return (round(a*100,2),round(b*100,2),round(c))
    kk = {}
    kk['span']  = fix( *eval_result['span'])
    kk['number']  = fix( *eval_result['number'])
    kk['spans']  = fix( *eval_result['spans'])
    kk['date']  = fix( *eval_result['date'])
    kk['total'] = fix( eval_result['total_em'],eval_result['total_f1'],eval_result['total_count'])

    logger.info(f"eval_result_pre:\n{kk}")
    
    for k,v in type_acc.items():
        a,b,c = v
        if abs(c) > 1e-8:
            a = round(a/c*100,2)
            b = round(b/c*100,2)
        type_acc[k] = (a,b,round(c),)

    
    logger.info(f"eval result:\n{type_acc}")

    logger.info(f"badcase num:{num_bad}")
    logger.info(f"goodcase num:{num_good}")


def tiao_dataset(reader):
    case_logger_path = "logger/sel12_all_v99_data_argument_debug/sel12_all_v99_data_argument_debug_case_logger_epoch0.json"
    case_logger = loadjson(case_logger_path)
    methods = ['passagespan','questionspan','multispans','program','count']
    
    metrics = DropEmAndF1(methods)

    eval_result = []
    
    deleted_questionId = loadjson("deleted_questionId.json")
    deleting_questionId = []
    def failed(a):
        if isinstance(a,str):
            a = float(a)
        return 1 if a<0.5 else 0

    def getType(questionId, reader):
        answer = reader.questionId2answer_dict[questionId]
        answer = answer_json_to_strings(answer[0])[1]
        return answer

    for questionId, case in case_logger.items():
        if questionId in deleted_questionId:
            continue
        predict_ability = case['predict_ability']
        answers = case['cases']
        ground_truths = reader.questionId2answer_dict[questionId]
        predicted_answers = []

        ems = []
        f1s = []
        for idx, answer in enumerate(answers):
            em,f1,predicted_answer,_ = answer
            em,f1 = float(em),float(f1)
            predicted_answers.append([predicted_answer,""])
            ems.append(em)
            f1s.append(f1)
        if reader.questionId2program_dict[questionId] == "NULL" and predict_ability == "passagespan":
            deleting_questionId.append(questionId)

    
    print(len(deleting_questionId))
    deleting_questionId = random.sample(deleting_questionId,int(len(deleting_questionId)//2))
    print(len(deleting_questionId))
    deleting_questionId = []
    for questionId, case in case_logger.items():
        if questionId in deleted_questionId or questionId in deleting_questionId: 
            continue
        predict_ability = case['predict_ability']
        answers = case['cases']
        ground_truths = reader.questionId2answer_dict[questionId]
        predicted_answers = []

        ems = []
        f1s = []
        for idx, answer in enumerate(answers):
            em,f1,predicted_answer,_ = answer
            em,f1 = float(em),float(f1)
            predicted_answers.append([predicted_answer,""])
            ems.append(em)
            f1s.append(f1)
        metrics(predicted_answers,ground_truths,predict_ability,questionId)


    eval_result = metrics.get_metric()
    print(print_eval_result(methods,eval_result))
    print(len(deleted_questionId)+len(deleting_questionId))
    
    for questionId in deleting_questionId:
        deleted_questionId[questionId] = True
    #savejson("deleted_questionId.json",deleted_questionId)

def test_print_eval_result():
    eval_result_path = "test.json"
    eval_result = loadjson(eval_result_path)
    answering_abilities = ["passagespan","questionspan","multispans","program","count"]
    #answering_abilities = ["program",]
    print(print_eval_result(answering_abilities,eval_result))

def print_count_info(reader):
    case_logger1 = loadjson("logger/sel12_all_classifier_train_/sel12_all_classifier_train__case_logger_epoch1.json")
    case_logger2 = loadjson("solve_count_problem/sel12_all_apl2_classifier_train_v111_case_logger_epoch0.json")
    answering_abilities = ['passagespan',"questionspan",'multispans','program','count']
    def getType(questionId, reader):
        answer = reader.questionId2answer_dict[questionId]
        answer = answer_json_to_strings(answer[0])[1]
        return answer

    program_right_number = 0
    count_right_number = 0
    program_count_all_failed = 0
    program_count_all_success = 0
    for questionId,case in case_logger1.items():
        cases = case['cases']
        count_index = answering_abilities.index("count")
        program_index = answering_abilities.index("program")
        type_question = case['ground_truth_answer_types'][0]
        if type_question == 'number':
            em_program,f1_program,answer_program,_ = cases[program_index]
            em_count,f1_count,answer_count,_ = cases[count_index]
            f1_program,f1_count = float(f1_program),float(f1_count)
            if f1_program >= 0.5 and f1_count >= 0.5:
                program_count_all_success += 1
            elif f1_program < 0.5 and f1_count > 0.5:
                count_right_number += 1
            elif f1_program >= 0.5 and f1_count < 0.5:
                program_right_number += 1
            elif f1_program < 0.5 and f1_count < 0.5:
                program_count_all_failed += 1
    print(f"all right:{program_count_all_success}\nprogram right:{program_right_number}\ncount_right:{count_right_number}\nall failed:{program_count_all_failed}")

    program_right_number = 0
    count_right_number = 0
    program_count_all_failed = 0
    program_count_all_success = 0
    for questionId,case in case_logger2.items():
        cases = case['cases']
        count_index = answering_abilities.index("count")
        program_index = answering_abilities.index("program")
        type_question = case['ground_truth_answer_types'][0]
        if type_question == 'number':
            em_program,f1_program,answer_program,_ = cases[program_index]
            em_count,f1_count,answer_count,_ = cases[count_index]
            f1_program,f1_count = float(f1_program),float(f1_count)
            if f1_program >= 0.5 and f1_count >= 0.5:
                program_count_all_success += 1
            elif f1_program < 0.5 and f1_count > 0.5:
                count_right_number += 1
            elif f1_program >= 0.5 and f1_count < 0.5:
                program_right_number += 1
            elif f1_program < 0.5 and f1_count < 0.5:
                program_count_all_failed += 1
    print(f"all right:{program_count_all_success}\nprogram right:{program_right_number}\ncount_right:{count_right_number}\nall failed:{program_count_all_failed}")


def divide_small_dataset(reader):
    questionId_list = []
    dev_questionId_list = []
    for questionId, program in reader.questionId2program_dict.items():
        if program == None:
            continue
        questionId_list.append(questionId)
    dev_questionId_list = random.sample(questionId_list, int(len(questionId_list)* 0.2)) #使用0.15的数据量进行dev
    train_questionId_list = [questionId for questionId in questionId_list if questionId not in dev_questionId_list]
    dev_questionId_list = [questionId for questionId in dev_questionId_list if reader.questionId2program_dict[questionId] != "NULL"]
    print(f"dev set:{len(dev_questionId_list)}")
    print(f"train set:{len(train_questionId_list)}")
    print(f"questionId total:{len(questionId_list)}")
    savejson("data/questionIds_train_version2.json",train_questionId_list)
    savejson("data/questionIds_dev_version2.json",dev_questionId_list)

def cases_analysis_good_case(reader):
    path = "logger/good_case_for_QDGAT_and_sel12_all_classifier_train_v120_c.json"
    goodcase = loadjson(path)
    goodcase = { output['questionId']:output for output in goodcase}

    methods = ['passagespan','questionspan','multispans','program','count']
    num_predicted_ability_str = {'passagespan':0,'questionspan':0,'multispans':0,'program':0,'count':0}
    num_program_type = {"addordiff":0,"compare_operator":0,"count":0,'divormul':0,'arg_compare_operator':0,'other':0}
    for questionId, output in goodcase.items():
        predicted_ability_str = output['predicted_ability_str']
        num_predicted_ability_str [predicted_ability_str] += 1

        program_index = methods.index("program")
        if predicted_ability_str == "program":
            program_case = output['cases'][program_index]
            program = program_case[3]
            if 'add' in program or 'diff' in program:
                num_program_type['addordiff'] +=  1
            elif 'argmax' in program or 'argmin' in program:
                num_program_type["arg_compare_operator"]+=1
            elif 'count' in program:
                num_program_type['count'] += 1
            elif 'div' in program or 'mul' in program or 'avg' in program:
                num_program_type['divormul'] += 1
            elif 'max' in program or 'min' in program:
                num_program_type['compare_operator'] += 1
            else :
                num_program_type['other'] += 1
                print(program_case)
    print("good case")
    print(num_predicted_ability_str)
    print(num_program_type)

def cases_analysis_bad_case(reader):
    path = "logger/bad_case_for_QDGAT_and_sel12_all_classifier_train_v120_c.json"
    badcase = loadjson(path)
    badcase = { output['questionId']:output for output in badcase}

    methods = ['passagespan','questionspan','multispans','program','count']
    num_predicted_ability_str = {}
    #num_program_type = {"addordiff":0,"compare_operator":0,"count":0,'divormul':0,'arg_compare_operator':0,'other':0}
    for questionId, output in badcase.items():
        predicted_ability_str = output['qdgat'][3]
        if predicted_ability_str not in num_predicted_ability_str:
            num_predicted_ability_str [predicted_ability_str] = 0
        num_predicted_ability_str[predicted_ability_str] += 1
    print("badcacse:")
    print(num_predicted_ability_str)

def dataset_analysis(reader):
    num_program_type = {"addordiff":0,"compare_operator":0,"count":0,'complex_logit':0,'arg_compare_operator':0,'other':0}
    questionId2type = {}

    for questionId in list(reader.questionIds_dev_version2):
        program = reader.questionId2program_dict[questionId]
        if 'argmax' in program or 'argmin' in program:
            num_program_type["arg_compare_operator"]+=1
            questionId2type[questionId] = 'arg_compare_operator'
        elif 'div' in program or 'mul' in program or 'avg' in program:
            num_program_type['complex_logit'] += 1
            questionId2type[questionId] = 'complex_logit'
        elif 'add' in program or 'diff' in program:
            num_program_type['addordiff'] +=  1
            questionId2type[questionId] = 'addordiff'
        elif 'count' in program:
            num_program_type['count'] += 1
            questionId2type[questionId] = 'count'
        elif 'max' in program or 'min' in program:
            num_program_type['compare_operator'] += 1
            questionId2type[questionId] = 'compare_operator'
        else :
            num_program_type['other'] += 1
            questionId2type[questionId] = 'other'
    
    return questionId2type




def fix_executor(reader):
    questionIds = [ "ec27013a-b39c-4e93-bf4a-69cdd798af32","3c273b67-afd2-430f-ad8b-3c418d279392","62c196fe-850e-4dea-b406-226ac8f46ac4","54a85589-3ffb-4ef2-ad92-0d93d4d21857",]
    for questionId in questionIds:
        program = reader.questionId2program_dict[questionId]
        exe = executor(program = program, questionId = questionId, reader = reader)
        if_run,predicted_answer = exe.get_answer()
        answer_annotations = reader.questionId2answer_dict[questionId]
        answer =[answer_json_to_strings(annotation)[0] for annotation in answer_annotations]
        answer_type =[answer_json_to_strings(annotation)[1] for annotation in answer_annotations]

        output = generate_bad_case_str(questionId = questionId, reader = reader)
        output['program'] = program
        output["predicted_answer"] = (if_run, predicted_answer)
        for k,v in output.items():
            logger.info("{}: {}".format(k,v))
        logger.info("")

import stanza

import string
from drop_reader import drop_tokenize,clean,whitespace_tokenize
STRIPPED_CHARACTERS = string.punctuation + ''.join([u"‘", u"’", u"´", u"`", "_","-"])


def checkNumber(text):
    try:
        float(text)
        return True
    except:
        return False

def checkAnyNumber(text):
    for i in range(10):
        if str(i) in text:
            return 1
    return 0

def checkNumberToken(text):
    if text.startswith('N') or text.startswith('Q'):
        if checkNumber(text[1:]):
            return True
    return False

def normalize_text(text, nlp):#统一各种文本的格式
    text = html.unescape(text)
    text = nlp(text)
    tokens = []
    for sentence in text.sentences:
        for token in sentence.words:
            tokens.append(token.text)
    return tokens

def match_span(tokens, string_tokens):

    if len(string_tokens) == 0:
        return None

    match_index = []
    for i in range(len(tokens)):
        idx, string_idx = i, 0
        while True:
            while True:
                if idx >= len(tokens):
                    break
                if not ( (checkNumberToken(tokens[idx]) ) or tokens[idx]=='-' or any([ str(number) in tokens[idx] for number in range(10)]) ):
                    break
                idx += 1
            
            while True:
                if string_idx >= len(string_tokens):
                    break
                if not ( checkNumberToken(string_tokens[string_idx])   or string_tokens[string_idx]=='-' or any([ str(number) in string_tokens[string_idx] for number in range(10)])) :
                    break
                string_idx += 1

            if idx == len(tokens) or string_idx == len(string_tokens):
                break
            if tokens[idx] != string_tokens[string_idx]:
                break

            idx += 1
            string_idx += 1
        
        if string_idx == len(string_tokens):
            logger.info(tokens[i:idx])
            match_index.append((i,idx))
    return  match_index

def find_right_number_token(tokens):
    proper_number_token = []
    for idx, token in enumerate(tokens):
        if idx + 1 < len(tokens):
            if (token == 'yard' or token == 'yards') and checkNumberToken(tokens[idx+1]):
                proper_number_token.append(tokens[idx+1])
        if idx - 1 >= 0:
            if (token == 'yard' or token == 'yards') and checkNumberToken(tokens[idx-1]):
                proper_number_token.append(tokens[idx-1])
    return proper_number_token


def fix_dataset(reader):


    nlp = stanza.Pipeline('en',processors="tokenize")

    questionId2program = {questionId:program for questionId,program in reader.questionId2program_dict.items() if (program != "NULL") and (program != "") and (program != None) and ('argmin' not in program) and ('argmax' not in program) and ('count' in program)}
    bad_questionId = {}
    fix_questionId = {}
    valid_questionId = {}
    
    
    
    for questionId, program in tqdm(questionId2program.items()):

        passageId = reader.questionId2passageId(questionId)
        if 'nfl' not in passageId :
            continue

        passage = reader.passageId2passage(passageId)
        add_number_token_passage = reader.passageId2add_number_token_passage(passageId)
        add_number_token_passage = re.sub('@',' ',add_number_token_passage)

        text_ant_passage = clean(add_number_token_passage)
        text_passage = clean(passage)
        
        text_ant_passage_tokens = normalize_text(text_ant_passage, nlp)
        text_passage_tokens = normalize_text(text_passage, nlp)

        #首先去除所有的非NFL类型的数据
        exe = executor(questionId = questionId, program= program,reader = reader)
        dd = exe.tokenize(program)
        Flag_program = False
        for token_, type_ in zip(dd['token'],dd['type']):
            if type_ == "string": #发现string类型的token
                valid_questionId[questionId] = True
                string = token_
                string_tokens = normalize_text(clean(string),nlp)
                match_index = match_span(text_ant_passage_tokens, string_tokens)

                
                if not match_index:
                    logger.info("match index less than one")
                    logger.info("string:{}".format(string))
                    logger.info("passage:{}\n".format(text_ant_passage))
                    if questionId not in bad_questionId:
                        bad_questionId[questionId] = []
                    bad_questionId[questionId].append([string,"NO MATCHING!"])
                    continue
                if len(match_index) > 1:
                    logger.info("match index greater than one")
                    logger.info("string:{}".format(string))
                    logger.info("passage:{}\n".format(text_ant_passage))
                    if questionId not in bad_questionId:
                        bad_questionId[questionId] = []
                    bad_questionId[questionId].append([string,"OVER MATCHING!"])
                    continue
                number_list = []
                start, end = match_index[0]
                number_list = find_right_number_token(text_ant_passage_tokens[start:end])
                if len(number_list) == 1:
                    type_ = number_list[0]

                    program = re.sub(re.escape(string), number_list[0], program)
                    
                    logger.info(f"Success! It will replace \"{string}\" with {number_list[0]} in\n {add_number_token_passage}\n")
                    if questionId not in fix_questionId:
                        fix_questionId[questionId] = []
                    fix_questionId[questionId].append([string,"SUCCESS!",number_list[0]])
                else :
                    logger.info("Failed because of number list:{}".format(number_list))
                    logger.info(number_list[start:end])
                    logger.info(match_index)
                    logger.info(f"Failed! String: \"{string}\" and:\n {add_number_token_passage}\n" )
                    if questionId not in bad_questionId:
                        bad_questionId[questionId] = []
                    bad_questionId[questionId].append([string,"NO PROPER NUMBER LIST!",number_list])
    

    savejson("logger/bad_questionId.json",bad_questionId)
    savejson("logger/fix_questionId.json",fix_questionId)
    savejson("logger/valid_questionId.json",valid_questionId)

def generate_fix_program_json(reader):
    bad_questionId = loadjson("logger/bad_questionId.json")
    fix_questionId = loadjson("logger/fix_questionId.json")
    valid_questionId = loadjson("logger/valid_questionId.json")

    fix_program = {}

    for questionId, replace_list in fix_questionId.items():
        program = reader.questionId2program_dict[questionId]
        for v in replace_list:
            token = "\"" + v[0] + "\""
            replaced_number_token = v[2]

            logger.info("pre program:{}".format(program))
            logger.info("{} replace with {}".format(token,replaced_number_token))

            program = re.sub(re.escape(token), replaced_number_token, program)

            logger.info("now program:{}\n".format(program))

        fix_program[questionId] = program
    savejson("data/fix_program.json", fix_program)

def test_dataset_2(reader):
    name_ = "eval_checkpoint"
    path = f"logger/{name_}/{name_}_case_logger_epoch0.json"
    case_logger = loadjson(path)
    valid_questionIds = {}
    num_span = 0
    for questionId, dd in case_logger.items():
        cases = dd['cases'][0]
        em, f1 , predicted_answer, program = cases
        em, f1 =float(em),float(f1)
        annotations = reader.questionId2answer_dict[questionId]
        answer_strings = [answer_json_to_strings(annotation)[0] for annotation in annotations] 
        answer_types = [answer_json_to_strings(annotation)[1] for annotation in annotations]

        if em == 1.0 and 'number' in answer_types and len(program)<=15:
            valid_questionIds[questionId] = program
        if 'number' not in answer_types:
            num_span += 1
    print(len(valid_questionIds))
    print(len(valid_questionIds) + num_span)

    savejson("logger/valid_questionIds.json", valid_questionIds)
    exit(0)

    test_questionIds = {k:v for k,v in valid_questionIds.items()if 'max' in v or 'min' in v}
    test_questionIds = [k for k in test_questionIds.keys()]

    logger.info("test_questionIds 1 number: {}".format(len(test_questionIds)))
    test_questionIds = random.sample(test_questionIds,5)
    output = []
    for questionId in test_questionIds:
        output_dict = generate_bad_case_str(questionId, reader)
        program = valid_questionIds[questionId]
        output_dict['program'] = program
        output_dict['predicted_answer'] = case_logger[questionId]['cases'][0][2]
        del output_dict['entity']
        del output_dict['number2token']
        output.append(output_dict)

    test_questionIds = {k:v for k,v in valid_questionIds.items()if 'add' in v or 'diff' in v}
    test_questionIds = [k for k in test_questionIds.keys()]

    logger.info("test_questionIds 2 number: {}".format(len(test_questionIds)))
    test_questionIds = random.sample(test_questionIds,5)
    for questionId in test_questionIds:
        output_dict = generate_bad_case_str(questionId, reader)
        program = valid_questionIds[questionId]
        output_dict['program'] = program
        output_dict['predicted_answer'] = case_logger[questionId]['cases'][0][2]
        del output_dict['entity']
        del output_dict['number2token']
        output.append(output_dict)

    test_questionIds = {k:v for k,v in valid_questionIds.items()if 'div' in v or 'avg' in v or 'mul' in v}
    test_questionIds = [k for k in test_questionIds.keys()]

    logger.info("test_questionIds 3 number: {}".format(len(test_questionIds)))
    test_questionIds = random.sample(test_questionIds,5)
    for questionId in test_questionIds:
        output_dict = generate_bad_case_str(questionId, reader)
        program = valid_questionIds[questionId]
        output_dict['program'] = program
        output_dict['predicted_answer'] = case_logger[questionId]['cases'][0][2]
        del output_dict['entity']
        del output_dict['number2token']
        output.append(output_dict)
    
    test_questionIds = {k:v for k,v in valid_questionIds.items()if 'count' in v }
    test_questionIds = [k for k in test_questionIds.keys()]

    logger.info("test_questionIds 4 number: {}".format(len(test_questionIds)))
    test_questionIds = random.sample(test_questionIds,5)
    for questionId in test_questionIds:
        output_dict = generate_bad_case_str(questionId, reader)
        program = valid_questionIds[questionId]
        output_dict['program'] = program
        output_dict['predicted_answer'] = case_logger[questionId]['cases'][0][2]
        del output_dict['entity']
        del output_dict['number2token']
        output.append(output_dict)

    test_questionIds = {k:v for k,v in valid_questionIds.items()if not any([ww in v for ww in ['count','add','diff','div','avg','mul','max','min']])}
    test_questionIds = [k for k in test_questionIds.keys()]

    logger.info("test_questionIds 5 number: {}".format(len(test_questionIds)))
    test_questionIds = random.sample(test_questionIds,5)
    for questionId in test_questionIds:
        output_dict = generate_bad_case_str(questionId, reader)
        program = valid_questionIds[questionId]
        output_dict['program'] = program
        output_dict['predicted_answer'] = case_logger[questionId]['cases'][0][2]
        del output_dict['entity']
        del output_dict['number2token']
        output.append(output_dict)
    
    savejson("logger/dev_program_generation.json", output)


def get_questionId_type(reader):
    questionId2type = {}
    for questionId,answers in tqdm(reader.questionId2answer_dict.items()):
        if answers == None or answers == []:
            continue
        types = ['number','span','spans','date']
        num_types = {type:0 for type in types}
        for annotation in answers:
            string, type = answer_json_to_strings(annotation)
            num_types [type] += 1
        max_type = 'number'
        max_number = 0
        for type in types:
            if max_number < num_types[type]:
                max_number = num_types[type]
                max_type = type
        questionId2type[questionId] = max_type
    savejson("data/questionId2type.json",questionId2type)


def get_badcase_classifier(reader):
    name = "sel12_all_classifier_train_v122_debug_d"
    path = "logger/{}/{}_case_logger_epoch2.json".format(name,name)
    case_logger = loadjson(path)
    output = []
    method = ['passagespan','questionspan','multispans','program','count']
    for k, v in case_logger.items():
        predicted_ability_str = v['predict_ability']
        index_predicted_ability = method.index(predicted_ability_str)
        cases = v['cases']
        
        best_answer = (-1.0,-1.0)
        best_predicted_ability = ""
        for i in range(len(cases)):
            em,f1 = cases[i][0],cases[i][1]
            if best_answer < (float(em),float(f1),):
                best_answer = (float(em),float(f1),)
                best_predicted_ability = method[i]
        if best_answer > (float(cases[index_predicted_ability][0]),float(cases[index_predicted_ability][1]),):
            output_dict = generate_bad_case_str(k, reader)
            output_dict['best_predicted_ability'] = best_predicted_ability
            output_dict['best_answer'] = best_answer

            output_dict['predict_ability'] = v['predict_ability']
            output_dict['predicted_answer'] = (cases[index_predicted_ability][0],cases[index_predicted_ability][1])
            output_dict['cases'] = v['cases']
            del output_dict['entity']
            del output_dict['number2token']
            if 'answer_program' in v:
                output_dict["answer_program"]  = v['answer_program']
            output.append(output_dict)
    savejson("logger/badcase_classifier.json",output)


def add_string(string):
    if "\"" in string:
        string = re.sub("\"","\"\"",string)
    string = "\"" + string + "\""
    return string

def generate_csv_line(ll):
    string = ""
    for l in ll:
        l = add_string(l)
        string += l
        string += ','
    return string

def generate_dev_csv_annotation(reader):
    drop_dataset_dev = loadjson("data/drop_dataset_dev.json")
    dev_questionIds = []
    for passageId, text in drop_dataset_dev.items():
        for qa_pair in text['qa_pairs']:
            questionId = qa_pair['query_id']
            dev_questionIds.append(questionId)
    
    valid_questionIds = loadjson("logger/valid_questionIds.json")
    pre_annotated_questionId = []
    for questionId in dev_questionIds:
        if questionId in valid_questionIds:
            continue
        annotations = reader.questionId2answer_dict[questionId]
        strings = [answer_json_to_strings(annotation)[0] for annotation in annotations]
        types = [answer_json_to_strings(annotation)[1] for annotation in annotations]
        if 'number' in types:
            pre_annotated_questionId.append(questionId)
    
    lines = []
    lines.append(generate_csv_line(['passageId','passage','add number token passage','questionId','question','reference answer','annotation','',]))
    for passageId , text in drop_dataset_dev.items():
        passage = reader.passageId2add_number_token_passage(passageId)
        clear_passage = reader.passageId2clear_passage(passageId)
        num_question = 0
        for qa_pair in text['qa_pairs']:
            questionId = qa_pair['query_id']
            if questionId in pre_annotated_questionId:
                num_question += 1
        if num_question:
            lines.append(generate_csv_line([passageId,clear_passage,passage,'',]))
            for qa_pair in text['qa_pairs']:
                questionId = qa_pair['query_id']
                if questionId not in pre_annotated_questionId:
                    continue

                annotations = reader.questionId2answer_dict[questionId]
                strings = [answer_json_to_strings(annotation)[0] for annotation in annotations]
                answer_dict = {}
                for string in strings:
                    if string not in answer_dict:
                        answer_dict[string] = 0
                    answer_dict[string] += 1
                max_string, max_num = "" , 0
                for string, num in answer_dict.items():
                    if answer_dict[string] > max_num:
                        max_num = answer_dict[string]
                        max_string = string
                answer = '  '.join(max_string)
                question = reader.questionId2add_number_token_question_dict[questionId]
                lines.append(generate_csv_line(['','','',questionId,question,answer,'',]))
            
    with open("drop_dataset_pre_annotation_v4.csv",mode='w') as f:
        for line in lines:
            f.write(line+'\n')
    print(len(pre_annotated_questionId))

def date_case_ananysis(reader):
    name = "sel12_all_classifier_train_v135_debug_a"
    path = f"logger/{name}/{name}_case_logger_epoch14.json"
    case_logger1 = loadjson(path)

    name = "NARoberta_small_output"
    path = f"logger/{name}/QDGAT_case_logger.json"
    case_logger2 = loadjson(path)

    questionId2type = loadjson("data/questionId2type.json")

    ems,f1s = [],[]

    
    baseline_ok_date_questionIds = {}
    for output in case_logger2:
        questionId = output["questionId"]
        predicted_answer = output['predicted_answer']
        em,f1,_ = eval(questionId=questionId, predicted_answer=predicted_answer,reader=reader)
        type_ = questionId2type[questionId]
        if type_ == 'span':
            baseline_ok_date_questionIds[questionId] = (em,f1,)
    case_logger2 = {output['questionId']:output for output in case_logger2}
    method = ['passagespan','questionspan','multispans','program','count']
    output_questionIds = []
    for questionId, v in case_logger1.items():
        cases = v['cases']
        index = method.index("passagespan")
        type_ = questionId2type[questionId]
        predict_ability = v['predict_ability']
        em,f1 = cases[index][0],cases[index][1]
        em,f1 = float(em),float(f1)
        if type_ == 'span' and (em,f1,) < baseline_ok_date_questionIds[questionId] and predict_ability=="passagespan":
            output_questionIds.append(questionId)
    
    outputs = []
    for questionId in output_questionIds:
        output = generate_bad_case_str(questionId, reader)
        output["predict_ability"] = case_logger1[questionId]['predict_ability']
        output["cases"] = case_logger1[questionId]['cases']
        if 'answer_program' in case_logger1[questionId]:
            output["answer_program"] = case_logger1[questionId]["answer_program"]
        output['NARoberta'] = case_logger2[questionId]
        del output["number2token"]
        del output["entity"]
        outputs.append(output)
    print(len(outputs))
    #outputs = random.sample(outputs, 20)
    savejson("logger/passagespan_diff.json",outputs)
    

def print_hard_count(reader):
    path = "logger/bad_questionId.json"
    bad_questionId = loadjson(path)
    outputs = []

    drop_dataset_train = loadjson("data/drop_dataset_train.json")
    dev_questionIds = []
    for passageId, text in drop_dataset_train.items():
        for qa_pair in text['qa_pairs']:
            questionId = qa_pair['query_id']
            dev_questionIds.append(questionId)
    
    lines = []
    lines.append(generate_csv_line(['passageId','passage','add number token passage','questionId','question','reference answer','pre annotation','now annotation',]))
    for passageId , text in drop_dataset_train.items():
        passage = reader.passageId2add_number_token_passage(passageId)
        clear_passage = reader.passageId2clear_passage(passageId)
        passage = html.unescape(passage)
        clear_passage = html.unescape(clear_passage)
        num_question = 0
        for qa_pair in text['qa_pairs']:
            questionId = qa_pair['query_id']
            if questionId in bad_questionId:
                num_question += 1
        if num_question:
            lines.append(generate_csv_line([passageId,clear_passage,passage,'',]))
            for qa_pair in text['qa_pairs']:
                questionId = qa_pair['query_id']
                if questionId not in bad_questionId:
                    continue
                program = reader.questionId2program_dict[questionId]
                annotations = reader.questionId2answer_dict[questionId]
                strings = [answer_json_to_strings(annotation)[0] for annotation in annotations]
                answer_dict = {}
                for string in strings:
                    if string not in answer_dict:
                        answer_dict[string] = 0
                    answer_dict[string] += 1
                max_string, max_num = "" , 0
                for string, num in answer_dict.items():
                    if answer_dict[string] > max_num:
                        max_num = answer_dict[string]
                        max_string = string
                answer = '  '.join(max_string)
                question = reader.questionId2add_number_token_question_dict[questionId]
                lines.append(generate_csv_line(['','','',questionId,question,answer,program,]))
    
    with open("drop_dataset_pre_annotation_v5.csv",mode='w') as f:
        for line in lines:
            f.write(line+'\n')
    print(len(lines))
    print(len(bad_questionId))


def print_how_many_count(reader):
    path = "logger/sel12_all_classifier_train_v122_debug_d/train_answers.json"
    logger = loadjson(path)

    drop_dataset_dev = loadjson("data/drop_dataset_dev.json")
    dev_questionIds = []

    for passageId, text in drop_dataset_dev.items():
        for qa_pair in text['qa_pairs']:
            questionId = qa_pair['query_id']
            dev_questionIds.append(questionId)
    train_questionIds = []

    for questionId in tqdm(logger.keys()):
        if questionId not in dev_questionIds:
            train_questionIds.append(questionId)

    case_logger_name = "sel12_all_classifier_train_v122_debug_d"
    case_logger = loadjson("logger/{}/{}_case_logger_epoch18.json".format(case_logger_name,case_logger_name))
    
    ems,f1s = [],[]
    original_ems,original_f1s = [],[]
    method = ["passagespan","questionspan","multispans","program","count"]
    for questionId, case in case_logger.items():
        cases = case['cases']
        predicted_ability_str = case['predict_ability']
        index_predicted_ability = method.index(predicted_ability_str)
        em,f1 = cases[index_predicted_ability][0],cases[index_predicted_ability][1]
        em,f1 = float(em),float(f1)
        original_ems.append(em)
        original_f1s.append(f1)
        question = reader.questionId2add_number_token_question_dict[questionId]
        if "how many year" in question or 'how many years' in question or 'how many month' in question or 'how many monthes' in question or 'how many day' in question or 'how many days' in question:
            program_index = method.index("program")
            now_em,now_f1 = em,f1 = cases[program_index][0],cases[program_index][1]
            ems.append(float(now_em))
            f1s.append(float(now_f1))
        else :
            ems.append(em)
            f1s.append(f1)
    
    import numpy as np
    print(f"{np.mean(original_ems)} {np.mean(original_f1s)}")
    print(f"{np.mean(ems)} {np.mean(f1s)}")

from retrieve_dates import retrieve_dates

def clean(text):
    return re.sub("[.]?:\d+\s+([A-Z][a-z])",'. \g<1>',text)

def dev_retrieve_date(reader):
    passageId2passage = reader.passageId2clear_passage_dict
    dev_questionIds = []
    stanza.install_corenlp()
    from stanza.server import CoreNLPClient
    di = {}
    with CoreNLPClient(
            endpoint="http://localhost:9007",
            output_format="json",
            annotators=['tokenize','ssplit','pos','lemma','ner', ],
            timeout=1000000,) as client:
        step = 0
        for passageId,_ in tqdm(passageId2passage.items()):
            passage = reader.passageId2passage(passageId)
            passage = clean(passage)
            passage = " ".join(whitespace_tokenize(passage))
            pre_passage = passage
            passage = retrieve_dates(passage)

            passage_corenlp = client.annotate(passage)
            di[passageId] = passage_corenlp
            if pre_passage !=passage :
                logger.info("pre passage!:\n{}".format(pre_passage))
                logger.info("passage!:\n{}".format(passage))
                step+= 1

    savejson("drop_dataset_corenlp_v2.json", di)

def get_dev_program_distr(reader):
    drop_dataset_dev = loadjson("data/drop_dataset_dev.json")
    dev_questionIds = []
    for passageId, text in drop_dataset_dev.items():
        for qa_pair in text['qa_pairs']:
            questionId = qa_pair['query_id']
            dev_questionIds.append(questionId)
    dev_questionId2program_type = {}
    logger.info(len(dev_questionIds))


    valid_questionIds = loadjson("logger/valid_questionIds.json")

    logger.info("valid_questionId number:{}".format(len(valid_questionIds)))
    questionId_index = 3
    annotation_index = 6
    people_index = 7
    reference_answer_index = 5

    file_name = "data/drop_dataset_pre_annotation_v4.csv"
    import csv
    with open(file_name,encoding='utf-8-sig') as f:
        csvreader = csv.reader(f)
        header_row = next(csvreader)
        annotation_today = [(line[questionId_index],line[annotation_index],line[people_index],line[reference_answer_index],) for line in csvreader if line[questionId_index]!="" and line[people_index]!="" and line[annotation_index]!=""] 


        #print(annotation_today)
        questionId2program_today = {questionId:annotation for questionId,annotation,_,_ in annotation_today}
        logger.info("annotation number:{}".format(len(questionId2program_today)))
    valid_questionIds = dict(valid_questionIds, **questionId2program_today)

    compare_operators = {}
    add_or_diff_operators = {}
    complex_operators = {}
    count_operators = {}
    other_operators = {}
    wa_hard = {}
    for questionId, program in valid_questionIds.items():
        program = re.sub("year\(","",program)
        program = re.sub("month\(","",program)
        program = re.sub("day\(","",program)
        program = re.sub("million\(","",program)
        program = re.sub("billion\(","",program)
        program = re.sub("decade\(","",program)


        if program.startswith("max") or program.startswith("min"):
            compare_operators[questionId] = program
            dev_questionId2program_type[questionId] = "compare_operators"
        elif program.startswith("add") or program.startswith("diff"):
            add_or_diff_operators[questionId] = program
            dev_questionId2program_type[questionId] = "add_or_diff_operators"
        elif program.startswith("div") or program.startswith("mul") or program.startswith("avg"):
            complex_operators[questionId] = program
            dev_questionId2program_type[questionId] = "complex_operators"
        elif program.startswith("count"):
            count_operators[questionId] = program
            dev_questionId2program_type[questionId] = "count_operators"

        elif program.startswith("HARD") or program.startswith("WA"):
            wa_hard[questionId] = program
            dev_questionId2program_type[questionId] = "wa_hard"
        else :
            if (not program.startswith("argmax")) and (not program.startswith("argmin")) and (not program.startswith("X")) and (not program.startswith("N")) and (not program.startswith("EASY")):
                logger.info(program)
            
            if (not program.startswith("argmax")) and (not program.startswith("argmin")) and (not program.startswith("EASY")):
                other_operators[questionId] = program
                dev_questionId2program_type[questionId] = "other_operators"
            else:
                dev_questionId2program_type[questionId] = "EASY"
    
    logger.info("compare operator: {}".format(len(compare_operators)))
    logger.info("add_or_diff_operators: {}".format(len(add_or_diff_operators)))
    logger.info("complex_operators: {}".format(len(complex_operators)))
    logger.info("count_operators: {}".format(len(count_operators)))
    logger.info("wa_hard: {}".format(len(wa_hard)))
    logger.info("other_operators: {}".format(len(other_operators)))

    EASY_number = 9536 - len(compare_operators) - len(add_or_diff_operators) - len(count_operators) - len(complex_operators) -len(wa_hard) - len(other_operators)
    logger.info("EASY: {}".format(EASY_number))

    for questionId in dev_questionIds:
        if questionId in dev_questionId2program_type :
            continue
        dev_questionId2program_type[questionId] = "EASY"
    logger.info(len(dev_questionId2program_type))

    values = {}
    for questionId,value in dev_questionId2program_type.items():
        if value not in values:
            values[value] = 0
        values[value] += 1
    logger.info(values)
    savejson("data/new_drop_dataset_annotation_v4_dev.json", valid_questionIds)
    savejson("data/dev_questionId2program_type.json",dev_questionId2program_type)

def get_dev_questionId2program_type(reader):
    drop_dataset_dev = loadjson("data/drop_dataset_dev.json")
    dev_questionIds = []
    for passageId, text in drop_dataset_dev.items():
        for qa_pair in text['qa_pairs']:
            questionId = qa_pair['query_id']
            dev_questionIds.append(questionId)
    dev_questionId2program_type = {}
    logger.info(len(dev_questionIds))
    for i in range(len(dev_questionIds)):
        for j in range(len(dev_questionIds)):
            if dev_questionIds[i] == dev_questionIds[j] and i!=j:
                print(dev_questionIds[i])

    valid_questionIds = loadjson("data/new_drop_dataset_annotation_v4_dev.json")

    logger.info("valid_questionId number:{}".format(len(valid_questionIds)))

    compare_operators = {}
    add_or_diff_operators = {}
    complex_operators = {}
    count_operators = {}
    other_operators = {}
    wa_hard = {}
    for questionId, program in valid_questionIds.items():
        program = re.sub("year\(","",program)
        program = re.sub("month\(","",program)
        program = re.sub("day\(","",program)
        program = re.sub("million\(","",program)
        program = re.sub("billion\(","",program)
        program = re.sub("decade\(","",program)
        if questionId in  dev_questionId2program_type:
            print(questionId)

        if program.startswith("max") or program.startswith("min"):
            compare_operators[questionId] = program
            dev_questionId2program_type[questionId] = "compare_operators"
        elif program.startswith("add") or program.startswith("diff"):
            add_or_diff_operators[questionId] = program
            dev_questionId2program_type[questionId] = "add_or_diff_operators"
        elif program.startswith("div") or program.startswith("mul") or program.startswith("avg"):
            complex_operators[questionId] = program
            dev_questionId2program_type[questionId] = "complex_operators"
        elif program.startswith("count"):
            count_operators[questionId] = program
            dev_questionId2program_type[questionId] = "count_operators"

        elif program.startswith("HARD") or program.startswith("WA"):
            wa_hard[questionId] = program
            dev_questionId2program_type[questionId] = "wa_hard"
        else :
            if (not program.startswith("argmax")) and (not program.startswith("argmin")) and (not program.startswith("X")) and (not program.startswith("N")) and (not program.startswith("EASY")):
                logger.info(program)
            
            if (not program.startswith("argmax")) and (not program.startswith("argmin")) and (not program.startswith("EASY")):
                other_operators[questionId] = program
                dev_questionId2program_type[questionId] = "other_operators"
            else:
                dev_questionId2program_type[questionId] = "EASY"
    
    logger.info("compare operator: {}".format(len(compare_operators)))
    logger.info("add_or_diff_operators: {}".format(len(add_or_diff_operators)))
    logger.info("complex_operators: {}".format(len(complex_operators)))
    logger.info("count_operators: {}".format(len(count_operators)))
    logger.info("wa_hard: {}".format(len(wa_hard)))
    logger.info("other_operators: {}".format(len(other_operators)))

    EASY_number = 9536 - len(compare_operators) - len(add_or_diff_operators) - len(count_operators) - len(complex_operators) -len(wa_hard) - len(other_operators)
    logger.info("EASY: {}".format(EASY_number))

    for questionId in dev_questionIds:
        if questionId in dev_questionId2program_type :
            continue
        dev_questionId2program_type[questionId] = "EASY"
    logger.info(len(dev_questionId2program_type))

    values = {}
    for questionId,value in dev_questionId2program_type.items():
        if value not in values:
            values[value] = 0
        values[value] += 1
    logger.info(values)
    savejson("data/dev_questionId2program_type.json",dev_questionId2program_type)




def test_xiangdeng(reader):
    num = 0
    for i in ['passages','questions']:
        filea = f"naroberta_dev_now_{i}.json"
        fileb = f"dev_now_{i}.json"
        a = loadjson(fileb)
        b = loadjson(fileb)
        for questionId in a.keys():
            logger.info(a[questionId])
            logger.info(b[questionId])
            if a[questionId] != b[questionId]:
                logger.info(questionId)
                logger.info(f"a: {a[questionId]}")
                logger.info(f"b: {b[questionId]}")
                num += 1
    logger.info(num)



import pickle
def get_diff_of_naroberta_and_nesy(reader):
    for data_mode in ['train','dev']:
        dpath = "{}.pkl".format(data_mode)
        with open(f"new_pkl_for_NARoberta/{dpath}", "rb") as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)
        data_roberta = data

        dpath = "{}.pkl".format(data_mode)
        with open(f"data/{dpath}", "rb") as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)
        data_nesy = data['instances']

        data_roberta = {item['question_id']:item for item in data_roberta}
        data_nesy = {item['question_id']:item for item in data_nesy}

        logger.info("len:{} {}".format(len(data_roberta), len(data_nesy)))

        for step, questionId in enumerate(data_roberta.keys()):
            item_roberta = data_roberta[questionId]
            item_nesy = data_nesy[questionId]

            del_list = []
            
            for key in item_roberta.keys():
                if key.startswith("gnode") or key.startswith("gedge") or key in ['signs_for_add_sub_expressions',]:
                    del_list.append(key)
            for key in del_list:
                del item_roberta[key]
            
            logger.info("questionId:{}".format(questionId))
            for key in item_nesy.keys():
                if key not in item_roberta:
                    continue
                if key == "answer_info" or key == "multi_span":
                    continue
                
                v1 = item_nesy[key]
                v2 = item_roberta[key]
                if isinstance(v1,dict) and isinstance(v2,dict):
                    if 'answer_program' in v1:
                        del v1['answer_program']
                    if 'answer_count' in v1:
                        del v1['answer_count']
                    if 'signs_for_add_sub_expressions' in v2:
                        del v2['signs_for_add_sub_expressions']
                

                if v1 != v2:
                    logger.info("key: {}".format(key))
                    logger.info("nesy:      {}".format(v1))
                    logger.info("naroberta: {}\n".format(v2))
                    break


            #logger.info("item_roberta:\n{}".format(item_roberta))
            #logger.info("item_nesy   :\n{}".format(item_nesy))
            if step > 10000:
                break


def fix_program_refix(reader):
    fix_program = loadjson("data/fix_program.json")
    file_name = "data/drop_dataset_pre_annotation_0128.csv"
    questionId_index = 3
    annotation_index = 7
    people_index = 8
    reference_answer_index = 5

    with open(file_name,encoding='utf-8-sig') as f:
        import csv
        csvreader = csv.reader(f)
        header_row = next(csvreader)
        annotation_today = [(line[questionId_index],line[annotation_index],line[people_index],line[reference_answer_index],) for line in csvreader if line[questionId_index]!="" and line[people_index]!="" and line[annotation_index]!=""] 
        #print(annotation_today)
        questionId2program_today = {questionId:annotation for questionId,annotation,_,_ in annotation_today}

    fix_program_v2 = {}
    for questionId , program in questionId2program_today.items():
        if program != "OK" and program != "HARD" and program != "WA" and program != "EASY":
            if_run, answer = run_program(questionId = questionId, reader= reader, program = program)
            if if_run:
                fix_program_v2 [questionId] = program
    
    savejson("data/fix_program_v2.json", fix_program_v2)


def getAB(reader):
    name = "sel12_all_classifier_train_v148_debug_a"
    path = f"logger/{name}/{name}_case_logger_epoch200.json"
    case_logger1 = loadjson(path)

    method = ['passagespan','questionspan','multispans','program','count']
    log = {}
    num = 0
    for questionId , item in case_logger1.items():
        num += 1
        predict_ability = item['predict_ability']
        predict_ability_index = method.index(predict_ability)
        predict_F1 = float( item['cases'][predict_ability_index][1])

        for i in range(len(method)):
            if i == predict_ability_index:
                continue
            now_F1 =float(item['cases'][i][1])
            if now_F1 > predict_F1:
                if f"({i}:{predict_ability_index})" not in log:
                    log[f"({i}:{predict_ability_index})"] = 0
                log[f"({i}:{predict_ability_index})"] += 1
    print(log)
    print(num)

def get_additioal_program(reader):
    name = "eval_checkpoint_train"
    case_logger = loadjson(f"logger/{name}/{name}_case_logger_epoch0.json")
    question2program_dict = {}
    
    for k,case in case_logger.items():
        if k in reader.questionId2program_dict:
            if reader.questionId2program_dict[k] != None:
                continue
        program_case = case['cases'][3]
        program = program_case[3]
        questionId = k
        em,f1 = program_case[0],program_case[1]
        em,f1 = float(em),float(f1)
        if em == 1.0:
            question2program_dict[questionId] = program
    
    print(len(question2program_dict))
    savejson("data/additional_questionId2program.json",question2program_dict)

def get_new_dataset(reader):
    pre_path = "new_data"

    for split in ['train','dev','test']:
        original_passage_path = f"data/drop_dataset_{split}.json"
        original_passage = loadjson(original_passage_path)
        savejson(f"{pre_path}/drop_dataset_{split}.json",original_passage) #原始DROP数据

        for passageId, text in original_passage.items():
            original_passage[passageId]['passage'] = reader.passageId2add_number_token_passage_dict[passageId]
            for id, qa_pair in enumerate(text['qa_pairs']):
                questionId = qa_pair['query_id']
                original_passage[passageId]["qa_pairs"][id]['question'] = reader.questionId2add_number_token_question_dict[questionId]
        savejson(f"{pre_path}/drop_dataset_add_number_token_{split}.json",original_passage) #原始DROP数据
    savejson(f"{pre_path}/questionId2program.json",reader.questionId2program_dict)

def get_new_dataset2(reader):
    questionId2program = reader.questionId2program_dict
    for questionId , program in questionId2program.items():
        if program == None:
            continue
        if 'argmin' in program or 'argmax' in program:
            questionId2program[questionId] = "NULL"
    savejson("new_data/new_questionId2program.json", questionId2program)


def data_subset_Anlysis(reader):
    questionId2program = loadjson("new_data/questionId2program_dev.json")
    no_NULL_questionId2program = {questionId:program for questionId,program in questionId2program.items() if program!=None}
    print("DROP-subset size:{}".format(len(no_NULL_questionId2program)))
    NULL_num = 0
    no_null_num = 0
    type2num = {}
    questionId2type = loadjson("new_data/questionId2type.json")
    for questionId, program in no_NULL_questionId2program.items():
        if program == "NULL" or 'argmin' in program or 'argmax' in program:
            NULL_num += 1
        else :
            no_null_num += 1
        type_ = questionId2type[questionId]
        if type_ not in type2num:
            type2num[type_] = 0
        type2num[type_] += 1
        
        #if type_ != "number" and (not(program == "NULL" or 'argmin' in program or 'argmax' in program)):
        #    print(f"type_:{type_} program:{program}")

    compare_operators = {}
    add_or_diff_operators = {}
    complex_operators = {}
    count_operators = {}
    other_operators = {}
    wa_hard = {}
    NULL_ = {}

    dev_questionId2program_type = {}
    for questionId, program in no_NULL_questionId2program.items():
        program = re.sub("year\(","",program)
        program = re.sub("month\(","",program)
        program = re.sub("day\(","",program)
        program = re.sub("million\(","",program)
        program = re.sub("billion\(","",program)
        program = re.sub("decade\(","",program)


        if program.startswith("max") or program.startswith("min"):
            compare_operators[questionId] = program
            dev_questionId2program_type[questionId] = "compare_operators"
        elif program.startswith("add") or program.startswith("diff"):
            add_or_diff_operators[questionId] = program
            dev_questionId2program_type[questionId] = "add_or_diff_operators"
        elif program.startswith("div") or program.startswith("mul") or program.startswith("avg"):
            complex_operators[questionId] = program
            dev_questionId2program_type[questionId] = "complex_operators"
        elif program.startswith("count"):
            count_operators[questionId] = program
            dev_questionId2program_type[questionId] = "count_operators"

        elif program.startswith("HARD") or program.startswith("WA"):
            wa_hard[questionId] = program
            dev_questionId2program_type[questionId] = "wa_hard"
        else :
            if (not program.startswith("argmax")) and (not program.startswith("argmin")) and (not program.startswith("NULL")):
                other_operators[questionId] = program
                dev_questionId2program_type[questionId] = "other_operators"
            elif program == "NULL":
                NULL_ [questionId] = program
                dev_questionId2program_type[questionId] = "NULL"
            else :
                print("!!!program:{}".format(program))






    print("DROP NULL num:{} DROP not NULL num:{}".format(NULL_num,no_null_num))
    print("type distr:{}".format(type2num))

    print("addordiff:{}\ncompare:{}\ncount:{}\ncomplex:{}\nother:{}\nwa_hard:{}\nNULL:{}".format(len(add_or_diff_operators),len(compare_operators),len(count_operators),len(complex_operators),len(other_operators),len(wa_hard),len(NULL_)))
    
    num_ = []
    num_.append(len(add_or_diff_operators))
    num_.append(len(compare_operators))
    num_.append(len(count_operators))
    num_.append(len(complex_operators))
    num_.append(len(other_operators))
    num_.append(len(wa_hard))
    num_.append(len(NULL_))
    sum_num_ = sum(num_)
    print(sum_num_)
    sum_num_ = [ round(i*1.0/sum_num_*100,2) for i in num_]
    print(sum_num_)

    depth_distr = {}
    not_run_num = 0
    sum_ = 0
    questionId2depth = {}
    for questionId, program in no_NULL_questionId2program.items():
        if program == "NULL":
            continue
        sum_ += 1
        exe = executor(program, questionId, reader)
        if_run, _ = exe.get_answer()
        depth = exe.max_depth
        if if_run :
            if depth not in depth_distr:
                depth_distr [depth] = []
            depth_distr [depth].append(program)
            questionId2depth[questionId] = depth
        else :
            print(program)
            depth_distr [2].append(program)
            questionId2depth[questionId] = 2
            not_run_num += 1
            
    #print(depth_distr)
    for depth, program_list in depth_distr.items():
        print("depth:{}".format(depth))
        print("数目:{} 比例:{}".format(len(program_list),round(len(program_list)*100.0/sum_,2)))
        print("example program:{}".format(program_list[:min(2,len(program_list))]))
    print(sum_)
    print(not_run_num)
    savejson("logger/questionId2depth.json",questionId2depth)

def get_list_f1( questionIds, questionId2f1):
    em, f1, num = 0.0,0.0,0
    for questionId in questionIds:
        num += 1
        em += float(questionId2f1[questionId][0])
        f1 += float(questionId2f1[questionId][1])
    if num > 0:
        em /= num
        f1 /= num
    em,f1 = round(em*100,2),round(f1*100,2)
    return em,f1,num

def get_logit_num_result(reader):
    questionId2depth = loadjson("logger/questionId2depth.json")

    case_logger_NA_Roberta = loadjson("output/NARoberta_small_output/QDGAT_case_logger.json")
    case_logger_NA_Roberta_fix = loadjson("output/NARoberta_small_output_fix/QDGAT_case_logger.json")
    case_logger_NSP = loadjson("logger/eval_checkpoint/eval_checkpoint_case_logger_epoch0.json")
    
    dev_questionIds = []

    

    answering_abilities = ["passage_span_extraction", "question_span_extraction","addition_subtraction", "counting", "multiple_spans"]
    questionId2f1_NARoberta = {}
    for case in case_logger_NA_Roberta:
        questionId2f1_NARoberta[case['questionId']] = case['cases'][answering_abilities.index(case["predicted_ability_str"])][:2]
        dev_questionIds.append(case['questionId'])

    print(len(dev_questionIds))

    questionId2f1_NARoberta_fix = {}
    for case in case_logger_NA_Roberta_fix:
        questionId2f1_NARoberta_fix[case['questionId']] = case['cases'][answering_abilities.index(case["predicted_ability_str"])][:2]

    answering_abilities = ["passagespan","questionspan","multispans","program","count"]
    questionId2f1_NSP = {}
    for questionId, case in case_logger_NSP.items():
        questionId2f1_NSP[questionId] = case['cases'][answering_abilities.index(case['predict_ability'])][:2]
        
    depth2questionIds = {}
    for questionId, depth in questionId2depth.items():
        if questionId not in dev_questionIds:
            continue
        if depth not in depth2questionIds:
            depth2questionIds[depth] = []
        depth2questionIds[depth].append(questionId)

    #print(len(dev_questionIds))
    #print(len(questionId2f1_NARoberta))
    #print(questionId2f1_NARoberta[dev_questionIds[0]])

    print("NARoberta:")

    print("total:")
    results = get_list_f1(dev_questionIds, questionId2f1_NARoberta)
    print(results)

    for depth , questionIds in depth2questionIds.items():
        print("depth:{}".format(depth))
        results = get_list_f1(questionIds, questionId2f1_NARoberta)
        print(results)
    
    print("NARoberta_fix:")

    print("total:")
    results = get_list_f1(dev_questionIds, questionId2f1_NARoberta_fix)
    print(results)

    for depth , questionIds in depth2questionIds.items():
        print("depth:{}".format(depth))
        results = get_list_f1(questionIds, questionId2f1_NARoberta_fix)
        print(results)

    print("NSP:")

    print("total:")
    results = get_list_f1(dev_questionIds, questionId2f1_NSP)
    print(results)

    for depth , questionIds in depth2questionIds.items():
        print("depth:{}".format(depth))
        results = get_list_f1(questionIds, questionId2f1_NSP)
        print(results)



def clear_dev_questionId2program(reader):
    case_logger_NA_Roberta = loadjson("output/NARoberta_small_output/QDGAT_case_logger.json")
    dev_questionIds = []
    for case in case_logger_NA_Roberta:
        dev_questionIds.append(case['questionId'])
    print(len(dev_questionIds))
    questionId2program = loadjson("new_data/questionId2program_dev.json")
    questionId2depth = {}

    not_run = 0
    for questionId in dev_questionIds:
        if questionId not in questionId2program:
            questionId2program[questionId] = "NULL"
        if questionId2program[questionId] == "EASY":
            questionId2program[questionId] = "NULL"
        if questionId2program[questionId] in ['WA','HARD']:
            questionId2program[questionId] = "NULL"
        if 'argmin' in questionId2program[questionId] or 'argmax' in questionId2program[questionId]:
            questionId2program[questionId] = "NULL"
        if questionId2program[questionId] == "NULL":
            continue 
        program = questionId2program[questionId]
        exe = executor(questionId = questionId, program = program, reader = reader)
        if_run, answer = exe.get_answer()
        if if_run:
            questionId2depth[questionId] = exe.max_depth
        else :
            not_run += 1
            print(program)
            print(answer)
    savejson("new_data/questionId2program_dev.json", questionId2program)
    print(not_run)


def get_dataset(reader):
    print(len(reader.questionId2program_dict))
    questionId2program_dict = loadjson("data/questionId2program.json")
    questionId2program_dict = {a:b for a,b in questionId2program_dict.items() if b != None}
    print(len(questionId2program_dict))

if __name__ == '__main__':

    reader = None
    
    reader = drop_reader()
    random.seed(1453)

    logging.basicConfig(filename="test.py.log",filemode="w",format="%(message)s",level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_executor", action='store_true', help = "")

    args = parser.parse_args()

    if args.test_executor:
        test_my_executor(reader)
    else :
        #get_bad_case2(reader)
        #cases_analysis_good_case(reader)
        #cases_analysis_bad_case(reader)
        #fix_executor(reader)
        #get_new_dataset2(reader)
        get_dataset(reader)
        # test_print_eval_result()
            #generate_additional_questionId2program(reader)
    #
    #exit(0)
    
    #little4(reader)


    #get_bad_case3(reader)
    #get_question_type(reader)
    #generate_additional_questionId2program(reader)
    #test_mydataset(reader)
    #put_in_data2(reader, origin_file_name="data/drop_dataset_annotation_version3.csv",output_file_name = "logger/new_drop_dataset_annotation_v3.json")
