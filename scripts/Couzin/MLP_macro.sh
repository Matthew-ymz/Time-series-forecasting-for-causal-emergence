#export CUDA_VISIBLE_DEVICES=0
  
model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Couzin/ \
  --data_path macro_1.csv \
  --model_id 2birds_macro_again \
  --model $model_name \
  --data Couzin \
  --data_partition 0.7 0.2 0.1 \
  --fold_loc 'normal' \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 8 \
  --prints 100 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 15 \
  --lradj type0 \
  --jacobian \
  --jac_init 0 \
  --jac_end 5996 \
  --jac_interval 1 \
  --cov_mean_num 5996 \
  --save_model \
