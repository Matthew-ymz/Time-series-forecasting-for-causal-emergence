#export CUDA_VISIBLE_DEVICES=0
#couzin_simulation.csv
model_name=NN

python -u run.py \
  --task_name coarse_graining \
  --is_training 1 \
  --root_path ./dataset/Spring/ \
  --data_path  6_group2_k0.csv\
  --model_id k0 \
  --model $model_name \
  --data Spring \
  --c_in 24 \
  --c_out 12 \
  --des 'Exp' \
  --d_model 256 \
  --MLP_layers 2 \
  --batch_size 8 \
  --prints 400 \
  --learning_rate 0.001 \
  --patience 5 \
  --itr 1 \
  --train_epochs 20 \
  --lradj type0 \
  --ig_output \
  --ig_baseline mean \
