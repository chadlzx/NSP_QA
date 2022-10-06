python baseline.py --do_train --do_eval --answering_abilities program --encoder_type=bart_encoder --num_decoder_layers=12 --max_epoch=20 --batch_size=8 --eval_batch_size=8 --warmup=-1 --name_of_this_trial=bart_training --num_eval_epoch=1 --lambda_list "1.0"  --save_model --learning_rate=1e-5 --bert_learning_rate=1e-5 --lambda_is_ok_for_spans=0.0  --lr_scheduler_constant --delete_no_number_answer



#  --do_train --do_eval --answering_abilities passagespan questionspan multispans program count --encoder_type=split --num_decoder_layers=12 --max_epoch=5 --batch_size=8 --eval_batch_size=8 --warmup=-1 --name_of_this_trial=classifier_train --num_eval_step=200 --delete_null_program --lambda_list "1.0" "1.0" "1.0" "1.0" "1.0" --save_model --no_add_number_token_text --learning_rate 1e-3 --gradient_accumulation_steps=1 --lambda_is_ok_for_spans=0.0 --load_checkpoint --checkpoint_path=output/roberta_training/state_5.pt --checkpoint_path2=output/bart_training/state_13.pt --loss_type=only_classifier --classifier_method=2 --get_old_train_answers --as_label=em


#  --do_train --do_eval --answering_abilities passagespan questionspan multispans count --encoder_type=split --num_decoder_layers=12 --max_epoch=5 --batch_size=8 --eval_batch_size=8 --warmup=-1 --name_of_this_trial=roberta_training --num_eval_epoch=1 --delete_null_program  --lambda_list "0.4" "0.4" "0.4" "0.4"   --save_model --no_add_number_token_text   --learning_rate 1e-4 --weight_decay 5e-5 --bert_learning_rate=1.5e-5 --bert_weight_decay=0.01 --gradient_accumulation_steps=1 --lambda_is_ok_for_spans=0.0 --use_original_loss

