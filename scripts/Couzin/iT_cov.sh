#export CUDA_VISIBLE_DEVICES=0
  
model_name=iTransformer_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Couzin/ \
  --data_path swarm_30_10000.csv \
  --model_id iT_cov_swarm \
  --model $model_name \
  --data Couzin \
  --data_partition 0.7 0.1 0.2 \
  --fold_loc 'normal' \
  --target stage \
  --seq_len 20 \
  --pred_len 20 \
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
  --batch_size 8 \
  --prints 300 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 7 \
  --inverse \
  --lradj type1 \
  --cov_bool \
  --loss_lam 0.001 \
  --jacobian \
  --jac_mean \
  --jac_init 5000 \
  --jac_end 7000 \
  --jac_interval 100 \
  --jac_mean_interval 10 \
