#export CUDA_VISIBLE_DEVICES=0
  
model_name=NN
length=10000
data_path="data_${length}_no_noise.csv"
model_id="lorzen_${length}_no_noise"

nohup python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Lorzen/ \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data Lorzen \
  --data_partition 0.8 0.1 0.1 \
  --fold_loc 'normal' \
  --target stage \
  --seq_len 1 \
  --pred_len 1 \
  --downsample 1 \
  --c_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 128 \
  --prints 10 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 15 \
  --inverse \
  --lradj type0 \
  --jacobian \
  --jac_init 0\
  --jac_end 10000 \
  --jac_interval 1000 \
  --cov_mean_num 10000 &


