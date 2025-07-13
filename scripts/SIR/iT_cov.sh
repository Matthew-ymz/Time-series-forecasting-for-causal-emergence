#export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/SIR/ \
  --model_id sir_review \
  --model $model_name \
  --data SIR \
  --size_list 9000 \
  --steps 7 \
  --sigma 0.1 \
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
  --batch_size 100 \
  --learning_rate 0.001 \
  --patience 5 \
  --itr 1 \
  --train_epochs 20 \
  --seed 2050 \
  --cov_bool \
  --loss_lam 1 \
  --lradj type0 \
  --jacobian \
  --jac_init 0 \
  --jac_end 54001 \
  --jac_interval 3000 \
  # --jac_mean \
  # --jac_mean_interval 2000 \

 