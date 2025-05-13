#export CUDA_VISIBLE_DEVICES=0
#2015-2016: 12419, 13880
  # --jac_mean \
  # --jac_init 12419 \
  # --jac_end 15492 \
  # --jac_interval 96 \
  # --jac_mean_interval 15 \

seq_len=40
  
model_name=iTransformer_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id cov_qbo_daily_nofreq_loss \
  --model $model_name \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --seq_len $seq_len \
  --pred_len $seq_len \
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
  --patience 7 \
  --itr 1 \
  --train_epochs 30 \
  --inverse \
  --lradj type1 \
  --cov_bool \
  --loss_lam 0.001 \
  --jacobian \
  --jac_mean \
  --jac_init 12554 \
  --jac_end 15492 \
  --jac_interval 96 \
  --jac_mean_interval 15 \
