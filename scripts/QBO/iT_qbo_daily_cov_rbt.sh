#export CUDA_VISIBLE_DEVICES=0
#2015-2016: 12419, 13880

model_name=iTransformer_cov

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/QBO/ \
  --data_path daily_1979_2023_16436_37.csv \
  --model_id iT_cov_qbo_daily_1520_rbt \
  --model $model_name \
  --data QBO \
  --target stage \
  --features 5 6 7 8 9 \
  --seq_len 96 \
  --pred_len 96 \
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
  --jacobian \
  --jac_init 12420 \
  --jac_end 15492 \
  --jac_interval 96 \
  --fold_loc vali_first \
  --cov_bool \
  --loss_lam 0.001
