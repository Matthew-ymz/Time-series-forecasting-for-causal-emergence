#export CUDA_VISIBLE_DEVICES=0
#couzin_simulation.csv
model_name=iTransformer

python -u run.py \
  --task_name coarse_graining \
  --is_training 1 \
  --root_path ./dataset/SIR/ \
  --data_path  train_7000_0.01.npy\
  --model_id for_macro_dyn \
  --model $model_name \
  --data SIR \
  --c_in 4 \
  --c_out 2 \
  --des 'Exp' \
  --d_model 256 \
  --MLP_layers 2 \
  --batch_size 1 \
  --prints 1000 \
  --learning_rate 0.001 \
  --use_amp \
  --patience 5 \
  --itr 1 \
  --train_epochs 20 \
  --lradj type0 \
  --ig_output \
  --ig_baseline zero \
  --one_serie \
