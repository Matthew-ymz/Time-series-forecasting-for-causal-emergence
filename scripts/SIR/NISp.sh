#export CUDA_VISIBLE_DEVICES=0

model_name=NISp

python -u run.py \
  --task_name maxei \
  --is_training 1 \
  --root_path ./dataset/SIR/ \
  --model_id sir_iid_maxei \
  --model $model_name \
  --prints 20 \
  --data SIR \
  --size_list 4000 \
  --steps 2 \
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
  --d_model 64 \
  --batch_size 64 \
  --learning_rate 0.002 \
  --patience 30 \
  --itr 1 \
  --train_epochs 30 \
  --fold_loc 1 \
  --EI \
  --latent_size 2 \
  --lambdas 0.1 \
  --first_stage 2 \