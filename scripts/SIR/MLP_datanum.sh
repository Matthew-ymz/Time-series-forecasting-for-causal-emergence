#export CUDA_VISIBLE_DEVICES=0

model_name=NN
nums=40

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/SIR/ \
  --model_id compare_100step \
  --model $model_name \
  --data SIR \
  --size_list $nums \
  --steps 101 \
  --no-data_scale \
  --sigma 0.01 \
  --rho -0.5 \
  --beta 1 \
  --gamma 0.5 \
  --dt 0.01 \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 4 \
  --c_out 4 \
  --des 'Exp' \
  --d_model 128 \
  --MLP_layers 1 \
  --batch_size 16 \
  --print 100 \
  --learning_rate 0.001 \
  --patience 5 \
  --itr 1 \
  --train_epochs 15 \
  --lradj type0 \
  --save_model \
  --jacobian \
  --jac_init 0 \
  --jac_end 4000 \
  --jac_interval 1 \
  --cov_mean_num 4000 \

 