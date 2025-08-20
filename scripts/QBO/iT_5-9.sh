#export CUDA_VISIBLE_DEVICES=0
#2015-2016: 12419, 13880
  # --jac_mean \
  # --jac_init 12419 \
  # --jac_end 15492 \
  # --enc_in 862 \
  # --dec_in 862 \
  # --c_out 862 \
  # --d_layers 1 \
# jac_mean_interval=10
# task_name="test_jacmean_${1}"
model_name=iTransformer
lens=40

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_5-9.csv \
  --model_id train_part \
  --model $model_name \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --seq_len $lens \
  --pred_len $lens \
  --downsample 1 \
  --e_layers 4 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 16 \
  --prints 300 \
  --learning_rate 0.001 \
  --es_delta -100 \
  --itr 1 \
  --train_epochs 7 \
  --lradj type0 \
  --jacobian \
  --cov_mean \
  --jac_init  12005 \
  --jac_end 15005 \
  --jac_interval $lens \

  # --jac_init 14403 12800 \
  # --jac_end  14400\
  # --jac_interval 40 \

