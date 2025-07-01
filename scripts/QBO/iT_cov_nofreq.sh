#export CUDA_VISIBLE_DEVICES=0
#2015-2016: 12419, 13880
  # --jac_mean \
  # --jac_init 12419 \
  # --jac_end 15492 \
  # --jac_interval 96 \
  # --jac_mean_interval 15 \

jac_mean_interval=10
# task_name="test_jacmean_${1}"
model_name=iTransformer_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id new_frame_bt16_epoch8 \
  --model $model_name \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --seq_len 25 \
  --pred_len 25 \
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
  --batch_size 16 \
  --prints 300 \
  --learning_rate 0.001 \
  --patience 8 \
  --itr 1 \
  --train_epochs 8 \
  --inverse \
  --lradj type1 \
  --cov_bool \
  --loss_lam 0.001 \
  --jacobian \
  --jac_mean \
  --jac_init 0 \
  --jac_end 1000 \
  --jac_interval 100 \
  --jac_mean_interval 10 \

