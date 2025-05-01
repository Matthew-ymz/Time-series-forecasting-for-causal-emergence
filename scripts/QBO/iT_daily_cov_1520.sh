#export CUDA_VISIBLE_DEVICES=0
#2015-2016: 12419, 13880
  # --jac_mean \
  # --jac_init 12419 \
  # --jac_end 15492 \
  # --jac_interval 96 \
  # --jac_mean_interval 15 \
  
model_name=iTransformer_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id iT_cov_qbo_daily_1520 \
  --model $model_name \
  --data QBO \
  --data_partition 0.8 0.1 0.1 \
  --fold_loc 'vali_test_first' \
  --target stage \
  --seq_len 30 \
  --pred_len 30 \
  --downsample 1 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 8 \
  --prints 300 \
  --learning_rate 0.001 \
  --patience 1 \
  --itr 1 \
  --train_epochs 1 \
  --inverse \
  --lradj type1 \
  --cov_bool \
  --loss_lam 0.001
