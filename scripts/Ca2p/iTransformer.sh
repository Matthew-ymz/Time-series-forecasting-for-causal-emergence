#export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Ca2p/ \
  --data_path C73dABCCa2pLP1_3_64G.csv \
  --model_id ca2pABCc73dlp1_3_64g_24_24 \
  --model $model_name \
  --data Ca2p \
  --target stage \
  --features M \
  --seq_len 24 \
  --pred_len 24 \
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
  --batch_size 128 \
  --learning_rate 0.001 \
  --itr 1 \
  --train_epochs 10
