#export CUDA_VISIBLE_DEVICES=0
#couzin_simulation.csv
model_name=NN

python -u run.py \
  --task_name coarse_graining \
  --is_training 1 \
  --root_path ./dataset/Couzin/ \
  --data_path  couzin_simulation.csv\
  --model_id twobird_moresample \
  --model $model_name \
  --data Couzin \
  --c_in 12 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 256 \
  --MLP_layers 2 \
  --batch_size 1 \
  --prints 50 \
  --learning_rate 0.001 \
  --patience 5 \
  --itr 1 \
  --train_epochs 20 \
  --lradj type0 \
  --ig_output \
  --ig_baseline mean \
