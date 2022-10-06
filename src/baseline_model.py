
from os import replace
from transformers import BartModel, BartTokenizer, BartConfig,BartForConditionalGeneration,logging,BartPretrainedModel
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput, 
    BaseModelOutputWithPastAndCrossAttentions, 
    CausalLMOutputWithCrossAttentions, 
    Seq2SeqLMOutput, 
    Seq2SeqModelOutput, 
    Seq2SeqQuestionAnsweringModelOutput, 
    Seq2SeqSequenceClassifierOutput, 
)
import math
import random
import numpy as np
import re
import torch 
from typing import Union, List, Tuple, Dict, Any, OrderedDict
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from multispan_heads import multispan_heads_mapping, remove_substring_from_prediction
from qdgat_utils import DropEmAndF1,metric_max_over_ground_truths

def my_masked_mean(x):
    num_valid = (x<0).long().sum(dim=-1) # batchsize * 1
    y = x.masked_fill_(x>0.0, 0 ).sum(dim=-1) #
    batch_mask = num_valid < 0.5
    num_valid = num_valid.masked_fill_( batch_mask, 1) #
    y = y / num_valid  #
    y = y.masked_fill_(batch_mask,1) #
    return y



def clear_summ(summ):
    summ = re.sub("<s>"," ",summ)
    summ = re.sub("<unk>"," ",summ)
    summ = re.sub("</s>"," ",summ)
    summ = re.sub("<ss>"," ",summ)
    summ = re.sub("<pad>"," ",summ)
    summ = re.sub("<sv>"," ",summ)
    summ = re.sub("  "," ",summ)
    summ = summ.strip()
    return summ 

logger = logging.get_logger(__name__)
from utils import savejson, checkAnswer
from executor import executor
from drop_reader import drop_reader
from network_utils import BertFeedForward
from allennlp_util import masked_log_softmax,masked_softmax,weighted_sum,replace_masked_values,get_best_span,logsumexp
from drop_eval import (get_metrics as drop_em_and_f1, answer_json_to_strings)
from tqdm import tqdm


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class BartForDrop(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        cross_attn_head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")

            loss_1 = loss_fct(lm_logits.transpose( 1, 2), labels)
            masked_lm_loss = torch.mean(loss_1, dim = 1)
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class ModelForDrop(nn.Module):
    def __init__(self,
                args,
                roberta,
                hidden_size: int = 1024,
                dropout_prob: float = 0.1,
                unique_on_multispan: bool = True,
                training: bool = True,
                bart_model = None,
                reader = None, 
                tokenizer = None,
                ):
        super(ModelForDrop, self).__init__()
        self.training = training
        self.roberta = roberta
        self.loss_type = args.loss_type
        self.answering_abilities = args.answering_abilities
        modeling_out_dim = hidden_size
        self.F1_total_for_train_dataset = []
        
        if args.encoder_type == 'roberta_encoder':
            hidden_size = self.roberta.config.hidden_size
        elif args.encoder_type == 'bart_encoder':
            hidden_size = bart_model.config.hidden_size
        elif args.encoder_type == 'split':
            hidden_size = self.roberta.config.hidden_size
        else :
            hidden_size = 1024

        self.classifier_train_flag = False
        self.classifier_eval_flag = False
        self.pre_stop = False
        self.count_down_sampling = args.count_down_sampling
        self.count_bad_probility = args.count_bad_probility

        self.lamda = args.lamda
        self.interesting_lamda = []

        self._drop_metrics = DropEmAndF1(self.answering_abilities)

        self.multispan_head_name = args.multispan_head_name
        self._dont_add_substrings_to_ms = True
        self.multispan_use_bio_wordpiece_mask = True
        self.multispan_use_prediction_beam_search = False
        self.multispan_generation_top_k = 0
        self.multispan_prediction_beam_size = 5

        self.encoder_type = args.encoder_type
        self.lambda_list = args.lambda_list
        assert(len(self.lambda_list)==len(self.answering_abilities))
        self.lambda_list = torch.tensor(self.lambda_list)
        
        self.as_label = args.as_label
        self.badcase_num = 0
        self.training_with_generate = args.training_with_generate
        self.classifier_method = args.classifier_method
        if self.classifier_method == 1 and self.loss_type == 'only_classifier':
            self.training_with_generate = True

        self.valid_methods_predictor = BertFeedForward( 3 * hidden_size  , hidden_size, len(self.answering_abilities))
        self.lambda_for_valid_methods_predictor = args.lambda_is_ok_for_spans

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = BertFeedForward( 3 * hidden_size  , int(hidden_size), len(self.answering_abilities))
            self._answer_predictor = torch.nn.Linear(len(self.answering_abilities), len(self.answering_abilities)) # use 

        if "passagespan" in self.answering_abilities or "questionspan" in self.answering_abilities:
            self._span_start_predictor = nn.Linear ( 4 * hidden_size, 1, bias=False)
            self._span_end_predictor = nn.Linear(4 * hidden_size, 1, bias=False)
        
        if "multispans" in self.answering_abilities:
            self.multispan_head = multispan_heads_mapping[args.multispan_head_name](modeling_out_dim,
                                                                                   generation_top_k=self.multispan_generation_top_k,
                                                                                   prediction_beam_size=self.multispan_prediction_beam_size)
            #self._multispan_module = self.multispan_head.module
            #self._multispan_log_likelihood = self.multispan_head.log_likelihood
            #self._multispan_prediction = self.multispan_head.prediction
            self._unique_on_multispan = unique_on_multispan
        
        if "count" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("count")
            self._count_number_predictor = BertFeedForward(5 * hidden_size, hidden_size, 10)


        self._dropout = torch.nn.Dropout(p=dropout_prob)
        self._proj_sequence_g0 = BertFeedForward(hidden_size, hidden_size, 1)
        self._proj_sequence_g1 = BertFeedForward(hidden_size, hidden_size, 1)
        self._proj_sequence_g2 = BertFeedForward(hidden_size, hidden_size, 1)

        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)

        self._proj_number = nn.Linear(hidden_size*2, 1, bias=False)
        
        self.bart = bart_model

        self.tokenizer = tokenizer
        self.reader = reader
    def decode__(self, answer):
        if not isinstance(answer, list):
            answer = answer.detach().cpu().numpy().tolist()
        answer = self.tokenizer.convert_ids_to_tokens(answer)
        answer = self.tokenizer.convert_tokens_to_string(answer)
        answer = clear_summ(answer)
        return answer
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._drop_metrics.get_metric(reset)

    def get_passage_spans(self, start_idx, end_idx, meta):
        
        passage_start = len(meta["question_tokens"]) + 2
        passage_str = meta['original_passage']
        offsets = meta['passage_token_offsets']

        
        start_offset = offsets[start_idx - passage_start][0]
        end_offset = offsets[end_idx - passage_start][1]

        predicted_answer = passage_str[start_offset:end_offset]
        return predicted_answer
    def get_question_spans(self, start_idx, end_idx, meta):
        question_start = 1
        question_str = meta['original_question']
        offsets = meta['question_token_offsets']

        start_offset = offsets[start_idx - question_start][0]
        end_offset = offsets[end_idx - question_start][1]

        predicted_answer = question_str[start_offset:end_offset]
        return predicted_answer

    def forward(self,
                input_ids: torch.LongTensor,
                bart_input_ids: torch.LongTensor,
                input_mask: torch.LongTensor,
                bart_input_mask: torch.LongTensor,
                input_segments: torch.LongTensor,
                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,
                number_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                is_program_mask : torch.LongTensor = None,
                answer_as_program: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                answer_as_text_to_disjoint_bios: torch.LongTensor = None,
                answer_as_list_of_bios: torch.LongTensor = None,
                span_bio_labels: torch.LongTensor = None,
                bio_wordpiece_mask: torch.LongTensor = None,
                is_bio_mask: torch.LongTensor = None,
                is_ok_for_spans = None,
                valid_methods_mask: torch.LongTensor = None, 
                metadata = None,
                ):
        output = {}
        output['input_ids'] = input_ids
        output['bart_input_ids'] = bart_input_ids
        output['input_mask'] = input_mask
        output['bart_input_mask'] = bart_input_mask




        if self.encoder_type == 'split':
            encoder_outputs = self.roberta(input_ids, attention_mask = input_mask , token_type_ids = input_segments,output_hidden_states=True, return_dict = True)
        elif self.encoder_type == 'roberta_encoder':
            encoder_outputs = self.roberta(input_ids, attention_mask = input_mask , token_type_ids = input_segments,output_hidden_states=True, return_dict = True)
        else : # bart encoder
            encoder_outputs = self.bart.model.encoder(input_ids, attention_mask = input_mask, output_hidden_states=True,  return_dict = True)
        
        sequence_output = encoder_outputs[0] #
        sequence_output_list = [ item for item in encoder_outputs['hidden_states'][-4:] ] #M0,M1,M2

        output['encoder_outputs'] = encoder_outputs
        output['sequence_output'] = sequence_output

        sequence_h2_weight = self._proj_sequence_h(sequence_output_list[2]).squeeze(-1) 
        passage_h2_weight = masked_softmax(sequence_h2_weight, passage_mask) # 
        passage_h2 = weighted_sum(sequence_output_list[2], passage_h2_weight) # passage embedding

        question_h2_weight = masked_softmax(sequence_h2_weight, question_mask)
        question_h2 = weighted_sum(sequence_output_list[2], question_h2_weight) # question embedding
        
        # passage g0, g1, g2
        question_g0_weight = self._proj_sequence_g0(sequence_output_list[0]).squeeze(-1)
        question_g0_weight = masked_softmax(question_g0_weight, question_mask)
        question_g0 = weighted_sum(sequence_output_list[0], question_g0_weight)

        question_g1_weight = self._proj_sequence_g1(sequence_output_list[1]).squeeze(-1)
        question_g1_weight = masked_softmax(question_g1_weight, question_mask)
        question_g1 = weighted_sum(sequence_output_list[1], question_g1_weight)

        question_g2_weight = self._proj_sequence_g2(sequence_output_list[2]).squeeze(-1)
        question_g2_weight = masked_softmax(question_g2_weight, question_mask)
        question_g2 = weighted_sum(sequence_output_list[2], question_g2_weight)



        real_number_indices = number_indices.squeeze(-1) - 1
        number_mask = (real_number_indices > -1).long()
        clamped_number_indices = replace_masked_values(real_number_indices, number_mask, 0)
        encoded_passage_for_numbers = torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=-1)

        encoded_numbers = torch.gather(encoded_passage_for_numbers, 1,
            clamped_number_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_numbers.size(-1)))
        number_weight = self._proj_number(encoded_numbers).squeeze(-1)
        number_mask = (number_indices > -1).long()
        number_weight = masked_softmax(number_weight, number_mask)
        number_vector = weighted_sum(encoded_numbers, number_weight)
        
        output['logits'] = []

        valid_methods_logits = self.valid_methods_predictor(torch.cat([passage_h2, question_h2, sequence_output[:, 0] ], 1))
        
        loss_for_valid_methods = F.binary_cross_entropy(torch.sigmoid(valid_methods_logits), valid_methods_mask.float())
        #loss_for_valid_methods = F.binary_cross_entropy(torch.sigmoid(valid_methods_logits), is_ok_for_spans.float().unsqueeze(1))

        output['loss_for_valid_methods'] = loss_for_valid_methods
        output['valid_methods_mask'] = valid_methods_mask
        output['valid_methods_logits'] = valid_methods_logits

        if self.classifier_method == 1 and self.pre_stop == True and self.loss_type == 'only_classifier':
            return output



        if "count" in self.answering_abilities:
            # Shape: (batch_size, 10)
            count_number_logits = self._count_number_predictor(torch.cat([number_vector, passage_h2, question_h2, sequence_output[:, 0]], dim=1))
            count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
            best_count_number = torch.argmax(count_number_log_probs, -1)
            #output['logits'].append(count_number_logits)
            output['count_score'] = count_number_log_probs.max(-1).values
            output['best_count_number'] = best_count_number

        if "passagespan" in self.answering_abilities or "questionspan" in self.answering_abilities:
            # start 0, 2
            sequence_for_span_start = torch.cat([sequence_output_list[2],
                                                 sequence_output_list[0],
                                                 sequence_output_list[2]* question_g2.unsqueeze(1),
                                                 sequence_output_list[0]* question_g0.unsqueeze(1)],
                                                dim=2) #
            sequence_span_start_logits = self._span_start_predictor(sequence_for_span_start).squeeze(-1)
            
            sequence_for_span_end = torch.cat([sequence_output_list[2],
                                               sequence_output_list[1],
                                               sequence_output_list[2]*question_g2.unsqueeze(1),
                                               sequence_output_list[1]*question_g1.unsqueeze(1)],
                                            dim=2)
            sequence_span_end_logits = self._span_end_predictor(sequence_for_span_end).squeeze(-1)

            if "passagespan" in self.answering_abilities:
                passage_span_start_log_probs = masked_log_softmax(sequence_span_start_logits, passage_mask)
                passage_span_end_log_probs = masked_log_softmax(sequence_span_end_logits, passage_mask)

                # Info about the best passage span prediction
                #passage_span_start_logits = replace_masked_values(sequence_span_start_logits, passage_mask, -1e7)
                #passage_span_end_logits = replace_masked_values(sequence_span_end_logits, passage_mask, -1e7)
                # Shage: (batch_size, topk, 2)
                best_passage_span, best_passage_scores = get_best_span(passage_span_start_log_probs, passage_span_end_log_probs)
                #logger.info("best_passage_span:{}".format(best_passage_span))
                output['best_passage_span'] = best_passage_span
                output['passagespan_score'] = best_passage_scores
                output['logits'].append(passage_span_start_log_probs + passage_span_end_log_probs)

            if "questionspan" in self.answering_abilities:
                question_span_start_log_probs = masked_log_softmax(sequence_span_start_logits, question_mask)
                question_span_end_log_probs = masked_log_softmax(sequence_span_end_logits, question_mask)

                # Info about the best question span prediction
                #question_span_start_logits = replace_masked_values(sequence_span_start_logits, question_mask, -1e7)
                #question_span_end_logits = replace_masked_values(sequence_span_end_logits, question_mask, -1e7)
                # Shape: (batch_size, topk, 2)
                best_question_span,best_question_scores = get_best_span(question_span_start_log_probs, question_span_end_log_probs)
                output['logits'].append(question_span_start_log_probs + question_span_end_log_probs)
                output['best_question_span'] = best_question_span
                output['questionspan_score'] = best_question_scores
        
        if bio_wordpiece_mask is None or not self.multispan_use_bio_wordpiece_mask:
            multispan_mask = input_mask
        else:
            multispan_mask = input_mask * bio_wordpiece_mask
        
        output['multispan_mask'] = multispan_mask
        output['bio_wordpiece_mask'] = bio_wordpiece_mask
        
        if "multispans" in self.answering_abilities:
            if self.multispan_head_name == "flexible_loss":
                multispan_log_probs, multispan_logits = self.multispan_head.module(sequence_output, seq_mask=multispan_mask)
            else:
                multispan_log_probs, multispan_logits = self.multispan_head.module(sequence_output)
            
            
            output['multispan_log_probs'] = multispan_log_probs
            output['multispan_logits'] = multispan_logits

        if "program" in self.answering_abilities:
            if self.encoder_type == 'split':
                bart_outputs = self.bart(input_ids = bart_input_ids,labels = answer_as_program, encoder_outputs= None )
            else :
                bart_outputs = self.bart(input_ids = bart_input_ids,labels = answer_as_program, encoder_outputs= (sequence_output,) )
            loss_program = - bart_outputs[0]
            loss_program = replace_masked_values(loss_program, is_program_mask, 1.0)
            loss_program = my_masked_mean(loss_program)
            original_loss_program = - bart_outputs[0] * self.lamda
            original_loss_program = replace_masked_values(original_loss_program, is_program_mask, -1e7)

            loss_program_in =  original_loss_program.clone().unsqueeze(1)
            output['logits'].append(loss_program_in)

            if (self.training==False or self.training_with_generate) :
                if not (self.loss_type == 'only_classifier' and self.pre_stop == True):
                    if self.encoder_type == 'split':
                        program_tokens = self.bart.generate(input_ids = bart_input_ids, 
                            attention_mask = bart_input_mask,
                            max_length=256,
                            top_k=50,
                            max_time=None,
                            num_beams = 5,
                            num_return_sequences=5,
                            return_dict_in_generate = True,
                            output_scores = True,
                        )
                    else :
                        program_tokens = self.bart.generate(input_ids = bart_input_ids, 
                            attention_mask = bart_input_mask,
                            max_length=256,
                            top_k=50,
                            max_time=None,
                            num_beams = 5,
                            num_return_sequences=5,
                            encoder_outputs=Seq2SeqModelOutput(last_hidden_state= sequence_output),
                            return_dict_in_generate = True,
                            output_scores = True,
                        )
                    output['sequences'] = program_tokens['sequences']
                    m = nn.ZeroPad2d((0,256 - output['sequences'].shape[1],0,0))
                    output['sequences'] = m(output['sequences'])
                    output['sequences_scores'] = program_tokens['sequences_scores']
        
        if len(self.answering_abilities) > 1: 
            
            answer_ability_logits = self._answer_ability_predictor(torch.cat([passage_h2, question_h2, sequence_output[:, 0] ], 1))
            answer_ability_log_probs = F.log_softmax(answer_ability_logits)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)
            output['best_answer_ability'] = best_answer_ability
            output['answer_ability_logits'] = answer_ability_logits
            if self.classifier_method == 1 and self.pre_stop == True and self.loss_type == 'only_classifier':
                return output
        


        loss = None
        original_loss = None
        if answer_as_passage_spans is not None or answer_as_question_spans is not None or answer_as_list_of_bios or answer_as_text_to_disjoint_bios or answer_as_program is not None or answer_as_counts:
            log_marginal_likelihood_list = []
            original_marginal_likelihood_list = []
            for answering_ability in self.answering_abilities:
                if answering_ability == "passagespan":
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_as_passage_spans[:, :, 0]  
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = (gold_passage_span_starts != -1).long()

                    clamped_gold_passage_span_starts = replace_masked_values( gold_passage_span_starts, gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends =   replace_masked_values( gold_passage_span_ends,   gold_passage_span_mask, 0)

                    log_likelihood_for_passage_span_starts = torch.gather(passage_span_start_log_probs, 1,
                                                                            clamped_gold_passage_span_starts)
                    log_likelihood_for_passage_span_ends = torch.gather(passage_span_end_log_probs, 1,
                                                                        clamped_gold_passage_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = replace_masked_values(log_likelihood_for_passage_spans,gold_passage_span_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = logsumexp(log_likelihood_for_passage_spans)

                    original_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span.clone())

                    log_marginal_likelihood_for_passage_span = log_marginal_likelihood_for_passage_span.masked_fill_(log_marginal_likelihood_for_passage_span < -1e5, 1.0)

                    log_marginal_likelihood_for_passage_span = my_masked_mean(log_marginal_likelihood_for_passage_span)

                    #logger.info(log_marginal_likelihood_for_passage_span)
                    #temp_mask = (log_marginal_likelihood_for_passage_span > -1e5).long()
                    #log_marginal_likelihood_for_passage_span = replace_masked_values(log_marginal_likelihood_for_passage_span,temp_mask, 1.0)
                    #logger.info(log_marginal_likelihood_for_passage_span)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)                                
                elif answering_ability == "questionspan":
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).long()
                    clamped_gold_question_span_starts = replace_masked_values(gold_question_span_starts,gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = replace_masked_values(gold_question_span_ends,gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = torch.gather(question_span_start_log_probs, 1, clamped_gold_question_span_starts)
                    log_likelihood_for_question_span_ends = torch.gather(question_span_end_log_probs, 1, clamped_gold_question_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_question_spans = replace_masked_values(log_likelihood_for_question_spans,gold_question_span_mask, -1e7)

                    
                    # Shape: (batch_size, )
                    # pylint: disable=invalid-name
                    log_marginal_likelihood_for_question_span = logsumexp(log_likelihood_for_question_spans)

                    original_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span.clone())
                    # question multi span prediction
                    #temp_mask = (log_marginal_likelihood_for_question_span > -1e5).long()
                    #log_marginal_likelihood_for_question_span = replace_masked_values(log_marginal_likelihood_for_question_span,temp_mask, 1.0)

                    log_marginal_likelihood_for_question_span = log_marginal_likelihood_for_question_span.masked_fill_(log_marginal_likelihood_for_question_span < -1e5, 1.0)
                    log_marginal_likelihood_for_question_span = my_masked_mean(log_marginal_likelihood_for_question_span)



                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)
                    # print('log_marginal_likelihood_for_question_span: ', log_marginal_likelihood_for_question_span)
                   
                elif answering_ability == "multispans":
                    if self.multispan_head_name == "flexible_loss":
                        log_marginal_likelihood_for_multispan = \
                            self.multispan_head.log_likelihood(answer_as_text_to_disjoint_bios,
                                                           answer_as_list_of_bios,
                                                           span_bio_labels,
                                                           multispan_log_probs,
                                                           multispan_logits,
                                                           multispan_mask,
                                                           bio_wordpiece_mask,
                                                           is_bio_mask)
                    else:
                        log_marginal_likelihood_for_multispan = \
                            self.multispan_head.log_likelihood(span_bio_labels,
                                                           multispan_log_probs,
                                                           multispan_mask,
                                                           is_bio_mask,
                                                           logits=multispan_logits)
                    
                    original_marginal_likelihood_list.append(log_marginal_likelihood_for_multispan.clone())
                    #temp_mask = (log_marginal_likelihood_for_multispan > -1e5).long()
                    #log_marginal_likelihood_for_multispan = replace_masked_values(log_marginal_likelihood_for_multispan,temp_mask, 1.0)
                    log_marginal_likelihood_for_multispan = log_marginal_likelihood_for_multispan.masked_fill_(log_marginal_likelihood_for_multispan < -1e5, 1.0)
                    log_marginal_likelihood_for_multispan = my_masked_mean(log_marginal_likelihood_for_multispan)



                    
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_multispan)
                elif answering_ability == "program":
                    loss_for_bartModel = loss_program
                    original_marginal_likelihood_list.append(original_loss_program)
                    log_marginal_likelihood_list.append(loss_for_bartModel.mean())
                    
                elif answering_ability == "count":
                    gold_count_mask = (answer_as_counts != -1).long()
                    # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = replace_masked_values(answer_as_counts, gold_count_mask, 0)
                    log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts)
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_counts = replace_masked_values(log_likelihood_for_counts, gold_count_mask,
                                                                           -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_count = logsumexp(log_likelihood_for_counts)
                    
                    original_marginal_likelihood_list.append(log_marginal_likelihood_for_count.clone())
                    #temp_mask = (log_marginal_likelihood_for_count > -1e5).long()
                    #log_marginal_likelihood_for_count = replace_masked_values(log_marginal_likelihood_for_count,temp_mask, 1.0)
                    log_marginal_likelihood_for_count = log_marginal_likelihood_for_count.masked_fill_(log_marginal_likelihood_for_count < -1e5, 1.0)
                    log_marginal_likelihood_for_count = my_masked_mean(log_marginal_likelihood_for_count)

                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)
                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")
            
            output['log_marginal_likelihood_list'] = log_marginal_likelihood_list
            if len(log_marginal_likelihood_list) > 1:
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                #lambda_list = torch.tensor(self.lambda_list, device = all_log_marginal_likelihoods.device)
                #log_marginal_likelihood = all_log_marginal_likelihoods * lambda_list
                log_marginal_likelihood = all_log_marginal_likelihoods 
                log_marginal_likelihood = log_marginal_likelihood.masked_fill_(all_log_marginal_likelihoods > 0, -1e7)
                log_marginal_likelihood = logsumexp(log_marginal_likelihood)

                #logger.info(original_marginal_likelihood_list)
                original_all_log_marginal_likelihoods = torch.stack(original_marginal_likelihood_list, dim=-1)
                output['loss_list'] = torch.stack(original_marginal_likelihood_list, dim=-1)
                original_all_log_marginal_likelihoods = original_all_log_marginal_likelihoods + answer_ability_log_probs
                original_marginal_log_likelihood = logsumexp(original_all_log_marginal_likelihoods)


            else :
                log_marginal_likelihood = log_marginal_likelihood_list[0]
                original_marginal_log_likelihood = original_marginal_likelihood_list[0]
            loss = - log_marginal_likelihood
            original_loss = - original_marginal_log_likelihood.mean()
            #loss = loss + loss_for_valid_methods.mean() * self.lambda_for_valid_methods_predictor 
            #original_loss = original_loss + loss_for_valid_methods.mean() * self.lambda_for_valid_methods_predictor
        output['loss'] = loss
        output['original_loss'] =  original_loss
        #output['best_count_number'] = best_count_number
        

        return output
    
    def only_classifier_loss(self, batch, encoder_output):
        
        #if self.classifier_method == 2:
        #    encoder_output['loss'] = encoder_output['original_loss']
        #    return encoder_output

        if self.classifier_method == 1:
            log = self.predict(batch, encoder_output)
            eval_result = log['eval_result']
            if self.as_label == 'f1':
                eval_result =  [      [ll[1] for ll in result]    for result in eval_result]
            else:
                eval_result =  [      [ll[0] for ll in result]    for result in eval_result]

        else : # classifier_method=2
            eval_result =  encoder_output['loss_list']
            eval_result = torch.exp(eval_result)


        answer_ability_logits = encoder_output['answer_ability_logits']
        
        
        ###################method 2
        #answer_ability_input = torch.tensor(log['scores'], device = encoder_output['input_ids'].device, dtype= torch.float, requires_grad=True)
        #answer_ability_input = answer_ability_input.exp()


        ####################method 1
        #answer_ability_input = encoder_output['valid_methods_logits']

        ####################
        #answer_ability_logits = self._answer_predictor(answer_ability_input)

        #logger.info(eval_result)
        answer_ability_label = torch.tensor(eval_result, device = encoder_output['input_ids'].device, dtype= torch.float, requires_grad=True)
        if 'loss' in encoder_output:
            encoder_output['pre_loss'] = encoder_output['loss']
        encoder_output["loss"] = F.binary_cross_entropy(torch.sigmoid(answer_ability_logits), answer_ability_label)
        return encoder_output

    def predict(self, batch, encoder_output = None):
        log = {}

        log['answer_program'] = []
        log['questionId'] = []
        
        
        with torch.no_grad():
            if encoder_output == None:
                output = self(**batch)
            else :
                output = encoder_output
            metadata = batch['metadata']
            batch_size = output['input_ids'].shape[0]
            answers = []
            scores = []
            if 'program' in self.answering_abilities:
                if 'sequences' in output:
                    output['sequences'] = output['sequences'].view(batch_size,5,-1)
                if 'sequences_scores' in output:
                    output['sequences_scores'] = output['sequences_scores'].view(batch_size,5,-1)
            for i in range(batch_size):

                log['questionId'].append(metadata[i]['question_id'])
                questionId = metadata[i]['question_id']
                answers.append([])
                scores.append([])
                if questionId in self.reader.questionId2answers and self.loss_type == 'only_classifier':
                    #logger.info(f"try to load:{questionId}")
                    scores[-1] = self.reader.questionId2scores[questionId]
                    answers[-1] = self.reader.questionId2answers[questionId]
                    continue
                #logger.info(f"now questionId:{questionId}")
                for answering_ability in self.answering_abilities :
                    if answering_ability == "passagespan":
                        best_passage_span = output['best_passage_span']
                        
                        passage_start = len(metadata[i]["question_tokens"]) + 2
                        passage_str = metadata[i]['original_passage']
                        offsets = metadata[i]['passage_token_offsets']
                        predicted_span = tuple(best_passage_span[i].detach().cpu().numpy())
                        #logger.info("best_passage_span:{}\npredicted_span:{}\n".format(best_passage_span[i],predicted_span))
                        start_offset = offsets[predicted_span[0] - passage_start][0]
                        end_offset = offsets[predicted_span[1] - passage_start][1]

                        predicted_answer = passage_str[start_offset:end_offset]

                        #logger.info(output['passagespan_score'])
                        scores[-1].append(output['passagespan_score'][i].item())

                        answers[-1].append((predicted_answer,str(predicted_span),))
                    if answering_ability == "questionspan":
                        best_question_span = output['best_question_span']

                        question_start = 1
                        question_str = metadata[i]['original_question']
                        offsets = metadata[i]['question_token_offsets']
                        predicted_span = tuple(best_question_span[i].detach().cpu().numpy())
                        start_offset = offsets[predicted_span[0] - question_start][0]
                        end_offset = offsets[predicted_span[1] - question_start][1]
                        predicted_answer = question_str[start_offset:end_offset]
                        
                        scores[-1].append(output['questionspan_score'][i].item())
                        answers[-1].append((predicted_answer,str(predicted_span),))
                    if answering_ability == "multispans":
                        multispan_log_probs = output['multispan_log_probs']
                        multispan_logits = output['multispan_logits']
                        multispan_mask = output['multispan_mask']
                        bio_wordpiece_mask = output['bio_wordpiece_mask']


                        passage_str = metadata[i]["original_passage"]
                        question_str = metadata[i]['original_question']
                        qp_tokens = metadata[i]["question_passage_tokens"]
                        if self.multispan_head_name == "flexible_loss":
                            (predicted_answer, spans, invalid_spans), multispans_score = \
                                self.multispan_head.prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens,
                                                        passage_str,
                                                        question_str,
                                                        multispan_mask[i], bio_wordpiece_mask[i],
                                                        self.multispan_use_prediction_beam_search and not self.training)
                        else:
                            predicted_answer, spans, invalid_spans = \
                                self.multispan_head.prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens,
                                                        passage_str,
                                                        question_str,
                                                        multispan_mask[i])
                        #logger.info("multispans answer v1:"+str(predicted_answer))
                        if self._unique_on_multispan:
                            predicted_answer = list(OrderedDict.fromkeys(predicted_answer))
                            if self._dont_add_substrings_to_ms:
                                predicted_answer = remove_substring_from_prediction(predicted_answer)
                        #logger.info("multispans answer v2:"+str(predicted_answer))
                        answers[-1].append((predicted_answer,""))
                        #logger.info(multispans_score)
                        scores[-1].append(multispans_score.item())
                    if answering_ability == "program":
                        questionId = metadata[i]['question_id']
                        
                        #print(program_tokens.items())
                        #logger.info(program_tokens['sequences'].shape)
                        programs = output['sequences'][i]
                        sequence_scores = output['sequences_scores'][i].detach().cpu().numpy().tolist()
                        programs = [self.decode__(program_) for program_ in programs]
                        #logger.info(programs,sequence_scores)
                        #print(programs)
                        flag = False
                        passage_str = metadata[i]["original_passage"]
                        question_str = metadata[i]['original_question']
                        answer_annotations = metadata[i].get('answer_annotations', [])
                        ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in answer_annotations] 
                        ground_truth_answer_types = [answer_json_to_strings(annotation)[1] for annotation in answer_annotations]

                        wrong_answer = []
                        answer_program = []
                        for score,program in zip(sequence_scores, programs):
                            exe = executor( program ,questionId, self.reader)
                            if_program_run, answer = exe.get_answer()
                            (exact_match, f1_score), max_type = metric_max_over_ground_truths(
                                    drop_em_and_f1,
                                    answer,
                                    ground_truth_answer_strings,
                                    ground_truth_answer_types,
                            )
                            if if_program_run:
                                answer_program.append((exact_match,f1_score,max_type,program,answer,score,))
                            else :
                                answer_program.append((exact_match,f1_score,max_type,program,"Error Program",score,))
                        
                        log["answer_program"].append(answer_program)

                        for score,program in zip(sequence_scores, programs):
                            exe = executor( program ,questionId, self.reader)
                            if_program_run, answer = exe.get_answer()
                            wrong_answer.append(answer)
                                    
                            (exact_match, f1_score), max_type = metric_max_over_ground_truths(
                                    drop_em_and_f1,
                                    answer,
                                    ground_truth_answer_strings,
                                    ground_truth_answer_types,
                            )
                            
                            if if_program_run :
                                predicted_answer = answer
                                flag = True
                                answers[-1].append((predicted_answer,program,))
                                #logger.info(f"program score:{score[0]}")
                                scores[-1].append(score[0])
                                break
                            else :
                                predicted_answer = "Error Program"
                            
                        if not flag:
                            scores[-1].append(-1e7)
                            answers[-1].append(("Error Program","Error Program",))
                    if answering_ability == "count":
                        best_count_number = output['best_count_number']
                        predicted_count = best_count_number[i].detach().cpu().numpy()
                        predicted_answer = str(predicted_count)
                        answers[-1].append((predicted_answer,predicted_answer,))
                        scores[-1].append(output['count_score'][i].item())


                self.reader.questionId2answers[questionId] = answers[-1]
                self.reader.questionId2scores[questionId] = scores[-1]

                #logger.info("save answers:{}".format(questionId))
            log['eval_result'] = []
            log['valid_methods_result'] = []

            if 'loss' in output:
                log['loss'] = output['loss']
            else :
                log['loss'] = torch.tensor([0.0])
            log['answers'] = answers
            log['scores'] = scores
            for i in range(batch_size):
                questionId = metadata[i]['question_id']
                if len(self.answering_abilities) > 1 :
                    if 'answer_ability_logits' not in output:
                        answer_ability_logits = torch.randn([len(self.answering_abilities)]).numpy()
                        print(1)
                    else :
                        answer_ability_logits = output['answer_ability_logits']
                        answer_ability_logits = answer_ability_logits.detach().cpu().numpy()[i]
                    
                    #program  
                    ##############method 2
                    #logger.info(scores)
                    #answer_ability_input = torch.tensor(scores, device = output['input_ids'].device, dtype= torch.float, requires_grad=True)
                    #answer_ability_input = answer_ability_input.exp()
                    ##############method 1
                    #answer_ability_input = output['valid_methods_logits']
                    ##############
                    #answer_ability_logits = self._answer_predictor(answer_ability_input)
                    #answer_ability_logits = answer_ability_logits.detach().cpu().numpy()[i]

                    ###############
                    if 'program' in self.answering_abilities:
                        program_index = self.answering_abilities.index("program")
                        if answers[i][program_index][0] == "NULL" or answers[i][program_index][0] == "Error Program" :
                            answer_ability_logits[program_index] = np.min(answer_ability_logits) - 1.0
                    answer_index = np.argmax(answer_ability_logits)
                    predicted_ability_str = self.answering_abilities[answer_index]
                else:
                    predicted_ability_str = self.answering_abilities[0]
                answer_annotations = metadata[i].get('answer_annotations', [])
                predicted_answer = []
                
                for j in range(len(self.answering_abilities)):
                    predicted_answer.append(answers[i][j])
                eval_result = self._drop_metrics(predicted_answer, answer_annotations, predicted_ability_str, questionId)
                log['eval_result'].append(eval_result)

                valid_methods_logits = output['valid_methods_logits']
                valid_methods_logits = torch.sigmoid(valid_methods_logits).detach().cpu().numpy()[i]
                valid_methods_logits = list(valid_methods_logits>0.5)
                valid_methods_mask = output['valid_methods_mask'].detach().cpu().numpy()[i]
                valid_methods_mask = list(valid_methods_mask)
                valid_methods_result = 0.0
                for j in range(len(self.answering_abilities)):
                    if valid_methods_mask[j] == valid_methods_logits[j]:
                        valid_methods_result += 1.0
                log['valid_methods_result'].append(valid_methods_result/len(self.answering_abilities))
            
            return log
