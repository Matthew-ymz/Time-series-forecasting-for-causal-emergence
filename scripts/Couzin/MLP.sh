#export CUDA_VISIBLE_DEVICES=0
  
model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/Couzin/ \
  --data_path couzin_simulation.csv \
  --model_id parallel \
  --model $model_name \
  --data Couzin \
  --data_partition 0.7 0.2 0.1 \
  --fold_loc 'normal' \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 60 \
  --c_out 60 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --batch_size 8 \
  --prints 100 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 15 \
  --lradj type0 \
  --jacobian \
  --jac_init 0 \
  --jac_end 5000 \
  --jac_interval 20 \
  --cov_mean_num 4000 \
