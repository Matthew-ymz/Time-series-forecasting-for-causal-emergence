#export CUDA_VISIBLE_DEVICES=0
#2015-2016: 12419, 13880
  # --jac_mean \
  # --jac_init 12419 \
  # --jac_end 15492 \
  # --jac_interval 96 \
  # --jac_mean_interval 15 \
  # --enc_in 862 \
  # --dec_in 862 \
  # --c_out 862 \
  # --d_layers 1 \
# jac_mean_interval=10
# task_name="test_jacmean_${1}"
model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id MSE_cov \
  --model $model_name \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --seq_len 40 \
  --pred_len 40 \
  --downsample 1 \
  --e_layers 4 \
  --factor 3 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --prints 200 \
  --learning_rate 0.01 \
  --patience 7 \
  --itr 1 \
  --train_epochs 30 \
  --lradj type1 \
  # --jacobian \
  # --jac_init 14600 \
  # --jac_end 15200 \
  # --jac_interval 40 \
  # --jacobian \
  # --jac_mean \
  # --jac_init 12554 \
  # --jac_end 15492 \
  # --jac_interval 100 \
  # --jac_mean_interval $jac_mean_interval \

