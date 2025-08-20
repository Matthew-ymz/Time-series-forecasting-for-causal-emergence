#export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id test_cov_mean \
  --data QBO \
  --data_partition 0.79 0.1 0.11 \
  --fold_loc 'vali_first' \
  --target stage \
  --model $model_name \
  --seq_len 2 \
  --pred_len 2 \
  --downsample 1 \
  --c_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --es_delta -100 \
  --itr 1 \
  --train_epochs 7\
  --print 300 \
  --lradj type0 \
  --jacobian \
  --cov_mean \
  --jac_init 12800 \
  --jac_end 15400 \
  --jac_interval 40 \
