#export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path data_mean_996_37.csv \
  --model_id data_mean_996_37 \
  --model $model_name \
  --data QBO \
  --target stage \
  --features M \
  --seq_len 12 \
  --pred_len 12 \
  --downsample 1 \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 64 \
  --dec_in 64 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --patience 5 \
  --itr 1 \
  --train_epochs 30 \
  --inverse \
  --lradj type1 \
  --jacobian
