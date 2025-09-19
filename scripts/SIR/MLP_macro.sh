#export CUDA_VISIBLE_DEVICES=0

model_name=NN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/SIR/ \
  --data_path macro_2.npy \
  --model_id check_macro \
  --model $model_name \
  --data SIR \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 2 \
  --c_out 2 \
  --des 'Exp' \
  --d_model 128 \
  --MLP_layers 1 \
  --batch_size 32 \
  --print 100 \
  --learning_rate 0.001 \
  --patience 5 \
  --itr 1 \
  --train_epochs 15 \
  --lradj type0 \
  --save_model \
  # --jacobian \
  # --jac_init 0 \
  # --jac_end 6000 \
  # --jac_interval 5 \
  # --cov_mean_num 6000 \

 