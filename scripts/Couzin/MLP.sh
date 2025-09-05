#export CUDA_VISIBLE_DEVICES=0
  
model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Couzin/ \
  --data_path couzin_simulation.csv \
  --model_id test_svd \
  --model $model_name \
  --data Couzin \
  --data_partition 0.8 0.1 0.1 \
  --fold_loc 'normal' \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 8 \
  --prints 10 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 15 \
  --inverse \
  --lradj type1 \
  --jacobian \
  --jac_init 0\
  --jac_end 1000 \
  --jac_interval 5 \
  --cov_mean \
