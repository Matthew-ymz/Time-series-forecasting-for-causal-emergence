#export CUDA_VISIBLE_DEVICES=0
#inverse

model_name=NN_cov

python -u run.py \
  --task_name nn_forecast \
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
  --c_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 12 \
  --learning_rate 0.001 \
  --patience 10 \
  --itr 1 \
  --train_epochs 25\
  --fold_loc 1 \
  --cov_bool \
  --print 20 \
  --lradj type1 \
  --jacobian \
