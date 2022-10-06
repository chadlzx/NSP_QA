import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    RobertaTokenizer,
    RobertaModel,
    BartForConditionalGeneration,
    BartConfig,
)
import logging
from qdgat_utils import AverageMeter
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer_utils import is_main_process
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from itertools import cycle
import copy

import random 
import numpy
from torch import optim,nn
from baseline_model import ModelForDrop,BartForDrop,my_masked_mean
from drop_eval import evaluate_prediction_file
from utils import *
from tqdm import tqdm
from executor import executor
from data_prepare import *
from drop_reader import drop_reader, preprocess
from dataloader import DropBatchGen, create_collate_fn

logger = logging.getLogger(__file__)

os.environ["WANDB_DISABLED"] = "true"
def my_masked_mean(x):
    num_valid = (x<0).long().sum(dim=-1) # batchsize * 1
    y = x.masked_fill_(x>0.0, 0 ).sum(dim=-1) #去除x中所有的大于0的数字，换成0，避免干扰sum
    batch_mask = num_valid < 0.5
    num_valid = num_valid.masked_fill_( batch_mask, 1) #去除 num_valid中所有的0换成1， 避免0/0的惨剧发生
    y = y / num_valid  #除一下
    y = y.masked_fill_(batch_mask,1) #将里面的无效的batch的结果变成1
    return y

def fix_output_spacing(s: str) -> str:
    """Fixing the odd bug that numerical numbers are losing a whitespace in 
    front after adding digits to special tokens.
    """
    match = re.compile(r'([a-z]|,|-)(\d)')
    s = re.sub(match, r'\1 \2', s)
    match = re.compile(r'(\d|[a-z])( )?(-)( )?(\d|[a-z])')
    s = re.sub(match, r'\1\3\5', s)
    return s

def load(network, path, path2=""):
    if path2 != "":#第一个span，第二个decoder
        logger.info(f"prepare to load from {path} and {path2}")
        model_state_dict = network.state_dict()
        path_state_dict = torch.load(path)
        path2_state_dict = torch.load(path2)
        path_state_dict = { k:v for k,v in path_state_dict.items() if not k.startswith("bart") and not k.startswith("_answer_predictor") and not k.startswith("valid_methods_predictor") and not k.startswith("_answer_ability_predictor")}
        path2_state_dict = { k:v for k,v in path2_state_dict.items() if k.startswith("bart") and not k.startswith("_answer_predictor") and not k.startswith("valid_methods_predictor") and not k.startswith("_answer_ability_predictor")}
        model_state_dict.update(path_state_dict)
        model_state_dict.update(path2_state_dict)
        network.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Load network params from {path} and {path2}")
        return network

    else :
        logger.info(f"prepare to load from {path}")
        if not os.path.exists(path):
            logger.info("load checkpoint fail!")
            return network
        path_state_dict = torch.load(path)
        for param_tensor in network.state_dict():
            if param_tensor not in path_state_dict:
                continue
            #if param_tensor.startswith("_answer_ability_predictor"):
            #    del path_state_dict[param_tensor]
            #    logger.info(f"del {param_tensor}")
            #    continue
                
            if path_state_dict[param_tensor].size() != network.state_dict()[param_tensor].size():
                logger.info("Dont load param_tensor:{}".format(param_tensor))
                del path_state_dict[param_tensor]

        network.load_state_dict(path_state_dict, strict=False)

        logger.info('Load network params from '+path)
        return network

def save(args, network, prefix, epoch):

    network_state = dict([(k, v.cpu()) for k, v in network.module.state_dict().items()])
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    state_path = prefix + f"state_{epoch}.pt"
    torch.save(network_state, state_path)
    logger.info('model saved to {}'.format(prefix))

def _monitor_tensorboard(log_dir, port=None):
    if not port:
        port = os.environ.get('ARNOLD_TENSORBOARD_CURRENT_PORT')
    cmd = 'tensorboard --logdir {}'.format(log_dir)
    if port is not None:
        cmd += ' --port {}'.format(port)
    #os.system(cmd + ' --bind_all &')


def upload_model(args):
    return 
    #Useless
    logger.info("Start upload model!")
    os.system(f"hadoop fs -put logger/{args.name_of_this_trial} logger/")

    os.system(f"hadoop fs -put output/{args.name_of_this_trial} output/")

    logger.info("End upload model!")

def lr_scheduler_function(epoch): 
    global total_step
    warmup = 0.06
    x = epoch*1.0/total_step
    if x < warmup:
        return x/warmup
    return 1.0 - x

def lr_scheduler_function_constant(epoch):
    return 1.0


def get_optimizer(args, network):
    if args.loss_type == 'only_classifier' or args.no_train_bart:
        optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, network.parameters()),lr = args.learning_rate,weight_decay=args.weight_decay)
        if not args.lr_scheduler_constant:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_scheduler_function, last_epoch=args.start_epoch)
        else :
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_scheduler_function_constant, last_epoch=args.start_epoch)
        return optimizer, scheduler

    else :
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if 'program' in args.answering_abilities:
            optimizer_grouped_parameters = network.module.parameters()
        else :
            optimizer_grouped_parameters = [
            {'params': [p for n, p in network.module.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in network.module.roberta.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in network.module.named_parameters() if not n.startswith("roberta.")],
                "weight_decay": args.weight_decay, "lr": args.learning_rate}
        ]


        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr = args.learning_rate, weight_decay=args.weight_decay)
        if not args.lr_scheduler_constant:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_scheduler_function,last_epoch=args.start_epoch)
        else :
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_scheduler_function_constant,last_epoch=args.start_epoch)
        return optimizer, scheduler



def check_full_(network, batch):
    for meta in batch['metadata']:
        questionId = meta['question_id']
        if questionId not in network.module.reader.questionId2answers:
            return 0
    return 1
def upload_writer(log_path):
    return
    os.system("hadoop fs -rm $ARNOLD_OUTPUT/*")
    os.system(f"hadoop fs -put {log_path}/* $ARNOLD_OUTPUT")

def train(args, network, train_itr, dev_itr):
    logger.info("Start training.")

    
    log_path = f"tflog/{args.name_of_this_trial}"
    _monitor_tensorboard(log_dir=log_path)
    writer = SummaryWriter(logdir=log_path)

    global total_step
    total_step = len(train_itr) * args.max_epoch / args.gradient_accumulation_steps

    optimizer, scheduler = get_optimizer(args, network)

    start_epoch = 1

    update_cnt, step = 0, 0
    train_start = datetime.now()
    save_prefix = args.save_dir

    loss_metric = AverageMeter()
    whole_loss_metric = AverageMeter()
    loss_list = [AverageMeter() for i in range(len(args.answering_abilities))]
    loss_valid_methods = AverageMeter()
    
    def eval():
        if args.num_eval_epoch != -1:
            eval_loss, eval_result = evaluate(args, network, dev_itr, epoch)
        else :
            eval_loss, eval_result = evaluate(args, network, dev_itr, step)
        eval_result = print_eval_result(args.answering_abilities,eval_result)
        writer.add_scalar('eval_loss', eval_loss, step)
        writer.add_text('eval_result',eval_result, step)

        upload_writer(log_path)
        logger.info("Step {} eval result, loss {}, result:\n {} .".format(step, eval_loss,eval_result))
        print("Step {} eval result, loss {}, result:\n {}.".format(step, eval_loss,eval_result))
        if args.save_model:
            if args.num_eval_epoch != -1:
                save(args, network, save_prefix, epoch)
            else :
                save(args, network, save_prefix, step)

    for epoch in range(start_epoch, args.max_epoch + 1):
        logger.info('Start epoch {}'.format(epoch))

        print("epoch : {}".format(epoch))
        for idx, batch in enumerate(tqdm(train_itr)):
            if batch['input_ids'].shape[0] != args.batch_size:
                continue
            step += 1

            if args.loss_type == 'only_classifier' and check_full_(network,batch):
                network.module.pre_stop = True
            else :
                network.module.pre_stop = False

            network.train()
            output_dict = network(**batch)
            if args.loss_type == 'only_classifier':
                output_dict = network.module.only_classifier_loss(batch, output_dict)
            loss = output_dict["loss"].mean()
            if args.use_original_loss and args.loss_type != "only_classifier":
                loss = output_dict['original_loss'].mean()
            loss_metric.update(loss.item())
            

            if "log_marginal_likelihood_list" in output_dict:
                for i in range(len(args.answering_abilities)):
                    loss_i_answering_abilities = my_masked_mean(output_dict["log_marginal_likelihood_list"][i]).item()
                    if loss_i_answering_abilities < 0:
                        loss_list[i].update(-loss_i_answering_abilities)
            whole_loss_metric.update(loss.item())
            loss_valid_methods.update(output_dict["loss_for_valid_methods"].mean().item())

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()


            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_cnt += 1
                scheduler.step()

                
            if update_cnt % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or update_cnt == 1:

                loss_avg_list = [loss.avg for loss in loss_list]
                for loss in loss_list:
                    loss.reset()
                logger.info("Answering abilities loss:{}".format(loss_avg_list))
                print("Answering abilities loss:{}".format(loss_avg_list))

                logger.info("loss for valid methods:{}".format(loss_valid_methods.avg))
                print("loss for valid methods:{}".format(loss_valid_methods.avg))
                loss_valid_methods.reset()


                loss_avg = loss_metric.avg
                loss_metric.reset()
                current_metrics = network.module.get_metrics(True)
                current_metrics = print_eval_result(args.answering_abilities,current_metrics)
                logger.info(f"QDGAT step:{step} loss: {loss_avg} result: {current_metrics}")

                for i in range(len(loss_list)):
                    writer.add_scalar('train_loss_{}'.format(i), loss_avg_list[i], step)
                writer.add_scalar('train_loss', loss_avg, step)
                writer.add_text('train_result', current_metrics, step)
                upload_writer(log_path)
                
                if args.loss_type == "only_classifier":
                    savejson(f"logger/{args.name_of_this_trial}/train_answers.json", network.module.reader.questionId2answers)
                    savejson(f"logger/{args.name_of_this_trial}/train_scores.json", network.module.reader.questionId2scores)

            if args.num_eval_step != -1 and step % args.num_eval_step == 0:
                eval()


        if args.num_eval_epoch!=-1 and epoch % args.num_eval_epoch == 0 and args.num_eval_step == -1:
            eval()
        
        

    logger.info("Train cost {}s.".format((datetime.now() - train_start).seconds))


def evaluate(args, network, dev_itr, epoch):
    logger.info("Start evaluating.")
    network.module.get_metrics(True) 
    network.eval()
    loss_metric = AverageMeter()
    valid_methods_metric = AverageMeter()

    eval_start = datetime.now()
    example_cnt = 0
    total_log = {}
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dev_itr)):
            example_cnt += batch['input_ids'].shape[0]

            if args.loss_type == 'only_classifier' and check_full_(network,batch):
                network.module.pre_stop = True
            else :
                network.module.pre_stop = False

            output = network(**batch)
            output_dict = network.module.predict(batch,output)
            output_dict['loss'] = output_dict['loss'].mean()
            loss_metric.update(output_dict["loss"].item())
            valid_methods_metric.update(sum(output_dict['valid_methods_result'])/len(output_dict['valid_methods_result']))
            if 'program' in args.answering_abilities and "answer_program" in output_dict:
                log = {questionId:answer_program for questionId,answer_program in zip(output_dict['questionId'],output_dict["answer_program"])}
                total_log = dict(total_log, **log)
            if idx % 100 == 0:
                metrics = network.module.get_metrics(False)
                metrics = print_eval_result(args.answering_abilities,metrics)
                logger.info(f"step {idx}: {metrics}")
                

    case_logger = network.module._drop_metrics.case_logger
    if 'program' in args.answering_abilities :
        for questionId, v in case_logger.items():
            if questionId in total_log:
                v["answer_program"] = total_log[questionId]
    if args.do_test:
        method_list = args.answering_abilities
        index_predicted_ability = {k:method_list.index(v['predict_ability']) for k,v in case_logger.items() }
        questionId2answer = { k:v['cases'][index_predicted_ability[k]][2] for k,v in case_logger.items()}
        savejson(f"logger/{args.name_of_this_trial}/{args.name_of_this_trial}_questionId2answer_{epoch}.json",questionId2answer)

    savejson(f"logger/{args.name_of_this_trial}/{args.name_of_this_trial}_case_logger_epoch{epoch}.json",case_logger)
    eval_metrics = network.module.get_metrics(True)    
    savejson(f"logger/{args.name_of_this_trial}/{args.name_of_this_trial}_eval_result_epoch{epoch}.json",eval_metrics)
    logger.info("Eval {} examples cost {}s.".format(example_cnt, (datetime.now() - eval_start).seconds))
    logger.info("valid_methods result:{}".format(valid_methods_metric.avg))
    print("valid_methods result:{}".format(valid_methods_metric.avg))

    return loss_metric.avg, eval_metrics

def set_environment(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--passage_length_limit", type=int, default=463)
    parser.add_argument("--question_length_limit", type=int, default=46)
    parser.add_argument("--data_dir", default="/opt/tiger/drop_dataset/data/", type=str, help="The data directory.")
    parser.add_argument("--save_dir", default="/opt/tiger/drop_dataset/output/", type=str, help="The directory to save checkpoint.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--log_per_updates", default=100, type=int, help="log pre update size.")
    parser.add_argument("--do_train", action="store_true", help="Whether to do training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to do evaluation.")
    parser.add_argument("--do_test", action="store_true", help="Whether to do test.")
    parser.add_argument("--max_epoch", default=20, type=int, help="max epoch.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="weight decay.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate.")
    parser.add_argument("--grad_clipping", default=1.0, type=float, help="gradient clip.")
    parser.add_argument('--warmup', type=float, default=0.06,help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")
    parser.add_argument("--optimizer", default="adam", type=str, help="train optimizer.")
    parser.add_argument('--seed', type=int, default=1453, help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--pre_path', type=str, default="/opt/tiger/drop_dataset/data/", help="Load from pre trained.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size.")
    parser.add_argument('--eval_batch_size', type=int, default=4, help="eval batch size.")
    parser.add_argument("--eps", default=1e-8, type=float, help="ema gamma.")
    parser.add_argument("--lamda", default=1.0, type=float, help="program loss parameters.")
    parser.add_argument("--bert_learning_rate",default = 1e-5, type=float, help="bert learning rate.")
    parser.add_argument("--bert_weight_decay",default = 0.01,type=float, help="bert weight decay.")
    parser.add_argument("--local_rank", type=int) 

    parser.add_argument("--in_trial", action="store_true", help="Whether in the trial.")
    parser.add_argument("--answering_abilities",nargs='+', default=['passagespan','questionspan','multispans','program','count'] ,help="answering_abilities: a list.")
    parser.add_argument("--lambda_list",nargs='+', default=["1.0","1.0","1.0","1.0","1.0"],help="lambda_list: a list.")

    parser.add_argument("--eval_dataset_sample",type=int , default=-1,help="Whether to sample eval dataset")
    parser.add_argument("--train_dataset_sample",type=int , default=-1,help="Whether to sample train dataset")
    parser.add_argument("--pre_train_program",action="store_true",help="Whether to pre train dataset")
    parser.add_argument("--program_annotation_method",type=str,choices=["only_annotation","add_no_number_examples","add_span_examples","add_all"],default="only_annotation")

    parser.add_argument("--debug_mode",action="store_true", help="Whether to print debug logging")
    parser.add_argument("--save_model",action="store_true", help="Whether to save model to HDFS")
    parser.add_argument("--load_checkpoint",action="store_true", help="Whether to load checkpoint")
    parser.add_argument("--checkpoint_path",type=str,default="", help="Checkpoint path to load model")
    parser.add_argument("--checkpoint_path2",type=str,default="", help="Checkpoint path to load model")
    parser.add_argument("--training_with_generate",action="store_true", help="Whether to generate with training")
    parser.add_argument("--program_model",type=str,default="",help="program model path, only load the bart model")
    parser.add_argument("--span_model",type=str,default="",help="span model path, only load the roberta and other")
    parser.add_argument("--loss_type",type=str,choices=['all','only_classifier'],default="all", help="only train classifier or train all model")
    parser.add_argument("--classifier_method",type=int,choices=[1,2],default=1,help='how to train the classifier of model')
    parser.add_argument("--multispan_head_name",type=str,choices=['simple_bio','simple_io','flexible_loss'],default="flexible_loss", help="use which multispan head")
    parser.add_argument("--encoder_type",type=str,default="split", choices=['split','roberta_encoder','bart_encoder'],help='use which encoder, split encoder, roberta_encoder or bart_encoder ?')
    parser.add_argument("--num_decoder_layers",type=int,default=6,help='if you not use the split way, you should select num of bart decoder layers to use')
    parser.add_argument("--add_number_token",action="store_true",help="if use want to add number token to the roberta tokenizer")
    parser.add_argument("--num_eval_step",type=int, default=-1, help="num of train step then to eval")
    parser.add_argument("--name_of_this_trial",type=str, default="hahaha",help="you should take a good name to your trial, thank you!")
    parser.add_argument("--flexible_train",action='store_true',help='whether you want to use the flexible to save our four way train! Attention! If you choose this training method, the answer abilities will be defaultly training all way!' )
    parser.add_argument("--delete_all_count_program",action='store_true',help="debug function to rm all count type program")
    parser.add_argument("--upload_model",action='store_true',help="debug function to rm all count type program")
    parser.add_argument("--data_argument",action='store_true',help="debug function to rm all count type program")
    parser.add_argument("--additional_program_level",type=int,default=-1,choices=[-1,0,1,2,3],help="additional_program_level, you can choose from -1,0,1,2,3. The number higher, the programs are added more.")
    parser.add_argument("--override_checkpoint",action='store_true',help="if you dont want the old checkpoint")
    parser.add_argument("--roberta_model",type=str,default='roberta-large', choices=["roberta-large", "roberta-large-squad2"],help='if you want to use roberta_squad model')
    parser.add_argument("--start_epoch",type=int,default = -1,help="default start epoch, type = int")
    parser.add_argument("--delete_null_program",action='store_true',help='whether to delete null program')
    parser.add_argument("--num_eval_epoch",type=int,default=-1,help='eval epoch num')
    parser.add_argument("--bart_model", type=str, default = "facebook/bart-large", choices=['bart-large','bart-base'],help="bart-large or bart-base")
    parser.add_argument("--no_train_bart",action='store_true',help='whether to train bart')
    parser.add_argument("--delete_no_number_answer",action='store_true',help="if you want to delete no number annotation in train program")
    parser.add_argument("--valid_program_used",action='store_true',help="if you want to make the train dataset must have valid program")
    parser.add_argument("--program_multiple_times",type=float,default=-1,help="program multiple times")
    parser.add_argument("--count_down_sampling",type=float,default=-1,help="count_down_sampling times")
    parser.add_argument("--count_bad_probility",type=float,default=-1,help="count_down_sampling times")
    parser.add_argument("--lambda_is_ok_for_spans",type=float,default=0.0,help="lambda for is_ok_for_spans loss")
    parser.add_argument("--get_old_train_answers", action='store_true',help="use old train answers")
    parser.add_argument("--no_add_number_token_text", action= 'store_true',help="if you want to no add number token")
    parser.add_argument("--train_dataset_v2", action="store_true",help="if you want to take dataset version2 mode")
    parser.add_argument("--delete_long_string_in_program", action="store_true",help="delete all long string program")
    parser.add_argument("--use_original_loss",action="store_true",help="use origin loss caculator")
    parser.add_argument("--as_label",type=str,choices=['f1','em',],default='f1',help='use what label to train classifier')
    parser.add_argument("--lr_scheduler_constant",action="store_true", help="Whether to use constant scheduler")

    args = parser.parse_args()

    assert(args.count_bad_probility == -1 or (args.count_bad_probility >=0 and args.count_bad_probility <= 1.0))
    assert(args.count_down_sampling == -1 or (args.count_down_sampling >=0 and args.count_down_sampling <= 1.0))

    assert( (args.num_eval_epoch!=-1 or args.num_eval_step!=-1) and (args.num_eval_epoch==-1 or args.num_eval_step==-1) )


    args.lambda_list = [float(i) for i in args.lambda_list]
    args.save_dir = f"output/{args.name_of_this_trial}/"
    
    logging.basicConfig(filename="config.log",filemode="w",format="%(asctime)s-%(name)s-%(levelname)s- %(message)s",level=logging.DEBUG)
    
    logger.info(args)
    return args
def add_number_token(tokenizer):
    logger.info("start add number tokenizer")
    logger.debug("before add special tokens, tokenizer size: {}".format(len(tokenizer)))

    special_tokens_dict = {'additional_special_tokens': [] } 
    FF = 99
    while FF >= 0:
        special_tokens_dict['additional_special_tokens'].append('N'+str(FF))
        FF -= 1 
    FF = 10
    while FF >= 0:
        special_tokens_dict['additional_special_tokens'].append('Q'+str(FF))
        FF -= 1 
    special_tokens_dict['additional_special_tokens'].append('argmax')
    special_tokens_dict['additional_special_tokens'].append('argmin')
    tokenizer.add_special_tokens(special_tokens_dict)

    logger.debug("After add special tokens, tokenizer size: {}".format(len(tokenizer)))

    return tokenizer

def main():
    args = get_args()

    if not os.path.exists(f"logger/{args.name_of_this_trial}"):
        os.system(f"mkdir logger/{args.name_of_this_trial}")
    if not os.path.exists(f"output/{args.name_of_this_trial}"):
        os.system(f"mkdir output/{args.name_of_this_trial}")
    

    logger.info("start preprocess")
    #### preprocess
    set_environment(args.seed)
    reader = drop_reader()
    if args.loss_type == 'only_classifier' and args.get_old_train_answers:
        if os.path.exists(f"logger/{args.name_of_this_trial}/train_answers.json") and os.path.exists(f"logger/{args.name_of_this_trial}/train_scores.json"):
            reader.questionId2answers = loadjson(f"logger/{args.name_of_this_trial}/train_answers.json")
            reader.questionId2scores = loadjson(f"logger/{args.name_of_this_trial}/train_scores.json")

    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
    if args.add_number_token:
        tokenizer = add_number_token(tokenizer)
    preprocess(tokenizer, reader)

    #### 
    logger.info("end preprocess")

    logger.info("start get dataloader")
    #### get dataloader
    train_dataset, eval_dataset = None, None
    collate_fn = create_collate_fn(tokenizer.pad_token_id, use_cuda = True)
    
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.batch_size = args.batch_size * torch.cuda.device_count()
    args.eval_batch_size = args.eval_batch_size * torch.cuda.device_count()

    if args.do_train:
        train_dataset = DropBatchGen(args, data_mode="train", tokenizer=tokenizer,reader= reader)
        train_sampler = RandomSampler(train_dataset)
        train_dataset = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False)
            
    if args.do_eval :
        eval_dataset = DropBatchGen(args, data_mode="dev", tokenizer=tokenizer,reader = reader)
        eval_dataset = DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False, shuffle=False)

    if args.do_test:
        eval_dataset = DropBatchGen(args, data_mode="test", tokenizer=tokenizer,reader = reader)
        eval_dataset = DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False, shuffle=False)



    ####
    logger.info("end get dataloader")
    
    logger.info("start build model")
    #### build model
    if args.encoder_type == 'roberta_encoder': #roberta_encoder
        roberta_model = RobertaModel.from_pretrained(args.roberta_model)
        bart_config = BartConfig.from_pretrained(args.bart_model)
        bart_config.decoder_layers = args.num_decoder_layers
        bart_model = BartForDrop(bart_config)

        robertaWordEmbedding = roberta_model.get_input_embeddings()
        bart_model.model.set_input_embeddings(robertaWordEmbedding)


    elif args.encoder_type == 'bart_encoder': #bart_encoder
        roberta_model = RobertaModel.from_pretrained(args.roberta_model)
        bart_config = BartConfig.from_pretrained(args.bart_model)
        bart_config.decoder_layers = args.num_decoder_layers
        bart_model = BartForDrop.from_pretrained(args.bart_model,config=bart_config)
        #robertaWordEmbedding = roberta_model.get_input_embeddings()
        #bart_model.model.set_input_embeddings(robertaWordEmbedding)

    else: #split
        roberta_model = RobertaModel.from_pretrained(args.roberta_model)
        bart_model = BartForDrop.from_pretrained(args.bart_model)
    

    if args.add_number_token:
        roberta_model.resize_token_embeddings(len(tokenizer))
        bart_model.resize_token_embeddings(len(tokenizer))
        
    model = ModelForDrop(args, roberta_model, bart_model=bart_model, reader = reader, tokenizer = tokenizer)    
    if 'program' not in args.answering_abilities:
        del model.bart

    if args.load_checkpoint:
        if args.checkpoint_path != "" and args.checkpoint_path2 == "":
            if not os.path.exists(args.checkpoint_path):
                checkpoint_path = os.path.normpath(args.checkpoint_path)
                path, file_name = os.path.split(checkpoint_path)
                if not os.path.exists(path):
                    os.makedirs(f"{path}")
                #os.system(f"hadoop fs -get {args.checkpoint_path} {path}")
            model = load(model, args.checkpoint_path)
        if args.checkpoint_path != "" and args.checkpoint_path2 != "":
            if not os.path.exists(args.checkpoint_path):
                checkpoint_path = os.path.normpath(args.checkpoint_path)
                path, file_name = os.path.split(checkpoint_path)
                os.makedirs(f"{path}")
                #os.system(f"hadoop fs -get {args.checkpoint_path} {path}")
            if not os.path.exists(args.checkpoint_path2):
                checkpoint_path = os.path.normpath(args.checkpoint_path2)
                path, file_name = os.path.split(checkpoint_path)
                os.makedirs(f"{path}")
                #os.system(f"hadoop fs -get {args.checkpoint_path2} {path}")
            model = load(model, args.checkpoint_path, args.checkpoint_path2)
    
    if args.loss_type == 'only_classifier':
        model.classifier_roberta = RobertaModel.from_pretrained(args.roberta_model)
        model.classifier_roberta = None
        for n,p in model.named_parameters():
            if not n.startswith('_answer_ability_predictor') and not n.startswith("_answer_predictor"):
                p.requires_grad = False
    ################################
    #for n,p in model.named_parameters():
    #    if not n.startswith('valid_methods_predictor') :
    #        p.requires_grad = False
    ###############################




    if args.no_train_bart :
        for n,p in model.named_parameters():
            if n.startswith('bart.'):
                p.requires_grad = False
        


    logger.info("end build model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model= nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.to(device)




    #### train
    if args.do_train:
        train(args, model, train_dataset, eval_dataset)
    elif args.do_eval or args.do_test:
        eval_loss, eval_result = evaluate(args, model, eval_dataset, 0)
        if args.save_model:
            save(args, model, args.save_dir, 0)
        logger.info("Step {} eval result, loss {}, result:\n{}.".format(0, eval_loss,print_eval_result(args.answering_abilities,eval_result)))
        print("Step {} eval result, loss {}, result:\n{}.".format(0, eval_loss,print_eval_result(args.answering_abilities,eval_result)))
    ####
    return
    
if __name__=='__main__':
    main()
