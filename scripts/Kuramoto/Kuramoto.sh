#export CUDA_VISIBLE_DEVICES=0

model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Kuramoto/ \
  --model_id Kuramoto \
  --model $model_name \
  --data Kuramoto \
  --sz_kuramoto 32 \
  --groups_kuramoto 2 \
  --batch_size_kuramoto 100 \
  --time_steps_kuramoto 1000 \
  --dt_kuramoto 0.01 \
  --sample_interval_kuramoto 1 \
  --coupling_strength 2.0 \
  --noise_level_kuramoto 10 \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 32 \
  --c_out 32 \
  --des 'Exp' \
  --d_model 128 \
  --MLP_layers 1 \
  --batch_size 64 \
  --prints 100 \
  --learning_rate 0.001 \
  --patience 50 \
  --itr 1 \
  --train_epochs 50 \
  --lradj type0 \
  --jacobian \
  --jac_init 0 \
  --jac_end 5000 \
  --jac_interval 10 \
  --cov_mean_num 5000 \