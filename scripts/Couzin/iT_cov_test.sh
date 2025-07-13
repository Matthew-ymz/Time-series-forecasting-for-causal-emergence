#export CUDA_VISIBLE_DEVICES=0
  
model_name=iTransformer_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Couzin/ \
  --data_path swarm_to_pa_10000.csv \
  --model_id MSE_loss \
  --model $model_name \
  --data Couzin \
  --data_partition 0.8 0.1 0.1 \
  --fold_loc 'vali_first' \
  --target stage \
  --seq_len 1 \
  --pred_len 25 \
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
  --train_epochs 30 \
  --inverse \
  --lradj type1 \
  --cov_bool \
  --loss_lam 0.1 \
  --jacobian \
  --jac_mean \
  --jac_init 7960 \
  --jac_end 8060 \
  --jac_interval 4 \
  --jac_mean_interval 1 \
  --causal_net \
