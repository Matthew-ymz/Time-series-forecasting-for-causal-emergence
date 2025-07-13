#export CUDA_VISIBLE_DEVICES=0
#inverse

model_name=NN_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id test_one \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --model $model_name \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 8 \
  --learning_rate 0.01 \
  --patience 7 \
  --itr 1 \
  --train_epochs 30\
  --cov_bool \
  --loss_lam 0.1 \
  --print 300 \
  --lradj type1 \
  --jacobian \
  --jac_init 14704 \
  --jac_end 14900 \
  --jac_interval 1 \
