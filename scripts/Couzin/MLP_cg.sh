#export CUDA_VISIBLE_DEVICES=0
#couzin_simulation.csv
model_name=NN

python -u run.py \
  --task_name coarse_graining \
  --is_training 1 \
  --root_path ./dataset/Couzin/ \
  --data_path  macro_8.csv\
  --model_id twobird \
  --model $model_name \
  --data Couzin \
  --c_in 8 \
  --c_out 2 \
  --des 'Exp' \
  --d_model 256 \
  --batch_size 8 \
  --prints 1 \
  --learning_rate 0.001 \
  --patience 7 \
  --itr 1 \
  --train_epochs 15 \
  --lradj type0 \
  --ig_output \
  --ig_baseline zero \
