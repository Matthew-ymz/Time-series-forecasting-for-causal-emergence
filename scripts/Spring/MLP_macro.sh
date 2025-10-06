#export CUDA_VISIBLE_DEVICES=0
  
model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Spring/ \
  --data_path macro_12.csv \
  --model_id macro_dyn_seed30 \
  --model $model_name \
  --data Spring \
  --data_partition 0.8 0.1 0.1 \
  --no-data_scale \
  --fold_loc 'normal' \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --d_model 128 \
  --MLP_layers 3 \
  --batch_size 8 \
  --prints 400 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 50 \
  --lradj type0 \
  --freq_loss \
  --jacobian \
  --jac_init 0 \
  --jac_end 10000 \
  --jac_interval 1 \
  --cov_mean_num 10000 \
  --save_model \
