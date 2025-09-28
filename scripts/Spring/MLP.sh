#export CUDA_VISIBLE_DEVICES=0
  
model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Spring/ \
  --data_path 6_group2.csv \
  --model_id check_ep50 \
  --model $model_name \
  --data Spring \
  --data_partition 0.8 0.1 0.1 \
  --data_scale True \
  --inverse \
  --fold_loc 'normal' \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 24 \
  --c_out 24 \
  --des 'Exp' \
  --d_model 256 \
  --MLP_layers 2 \
  --batch_size 8 \
  --prints 500 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 50 \
  --lradj type0 \
  --jacobian \
  # --jac_init 0 \
  # --jac_end 6000 \
  # --jac_interval 1 \
  # --cov_mean_num 6000 \
