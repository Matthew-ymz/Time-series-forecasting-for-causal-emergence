#export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id iT_qbo_daily \
  --model $model_name \
  --data QBO \
  --target stage \
  --features 5 6 7 8 9 \
  --seq_len 96 \
  --pred_len 96 \
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
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 30 \
  --inverse \
  --lradj type1 \
  --jacobian \
  --jac_init 12419 \
  --jac_end 13880 \
  --jac_interval 96 \
  --fold_loc vali_first \
