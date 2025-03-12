#export CUDA_VISIBLE_DEVICES=0

model_name=NN_cov

python -u run.py \
  --task_name nn_forecast \
  --is_training 1 \
  --root_path ./dataset/SIR/ \
  --model_id sir_iid \
  --model $model_name \
  --data SIR \
  --size_list 9000 \
  --steps 7 \
  --sigma 0.03 \
  --rho -0.5 \
  --beta 1 \
  --gamma 0.5 \
  --dt 0.01 \
  --target stage \
  --features M \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 64 \
  --learning_rate 0.01 \
  --patience 10 \
  --itr 1 \
  --train_epochs 10 \
  --fold_loc 1 \
  --seed 2050 \
  --cov_bool \
 