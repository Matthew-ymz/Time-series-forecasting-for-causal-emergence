#export CUDA_VISIBLE_DEVICES=0

model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id test_mape \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --model $model_name \
  --seq_len 40 \
  --pred_len 40 \
  --downsample 1 \
  --c_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 30\
  --print 300 \
  --lradj type0 \
  # --jacobian \
  # --cov_mean \
  # --jac_init 12600 \
  # --jac_end 14600 \
  # --jac_interval 40 \
