# pretrain + fine tune
# carpet -> grid (id 2 -> 13)

python train_DRAEM.py --data_path datasets/mvtec --obj_id 2 --sample_rate 1.0 --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 100 --log_path logs/carpet_sp_100

# Ratio 100% AP 74.3
# Ratio 50% AP 73.3
# Ratio 30% AP 71.5
# Ratio 10% AP 62.6
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 1.0 --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 100 --log_path logs/grid_sp_100
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.5 --anomaly_source_path datasets/dtd/images \
  --gpu_id 1 --lr 0.0001 --bs 6 --epochs 200 --log_path logs/grid_sp_50
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.3 --anomaly_source_path datasets/dtd/images \
  --gpu_id 2 --lr 0.0001 --bs 6 --epochs 300 --log_path logs/grid_sp_30
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --gpu_id 3 --lr 0.0001 --bs 6 --epochs 1000 --log_path logs/grid_sp_10

# TODO zero-shot eval
# fine-tuning

# both generator and discriminator
# 63.1
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --pretrained-generative logs/carpet_sp_100/checkpoints/latest_generative.pth \
  --pretrained-discriminative logs/carpet_sp_100/checkpoints/latest_discriminative.pth \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 1000 --log_path finetune/both

# discriminator only
# 68.5
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --pretrained-discriminative logs/carpet_sp_100/checkpoints/latest_discriminative.pth \
  --gpu_id 1 --lr 0.0001 --bs 6 --epochs 1000 --log_path finetune/disc

# generator only
# 66.9
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --pretrained-generative logs/carpet_sp_100/checkpoints/latest_generative.pth \
  --gpu_id 2 --lr 0.0001 --bs 6 --epochs 1000 --log_path finetune/gene

# baseline
# 62.4
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --gpu_id 3 --lr 0.0001 --bs 6 --epochs 1000 --log_path finetune/baseline

# joint training

# carpet only 66.7
python train_DRAEM.py --data_path datasets/mvtec --obj_id 2 --sample_rate 1 --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 100 --log_path joint_train/carpet_only

# grid only 72.2
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 1 --anomaly_source_path datasets/dtd/images \
  --gpu_id 1 --lr 0.0001 --bs 6 --epochs 100 --log_path joint_train/grid_only

# carpet + grid 57.3 71.8
python joint_train.py --data_path datasets/mvtec --anomaly_source_path datasets/dtd/images \
  --gpu_id 2 --lr 0.0001 --bs 6 --epochs 100 --log_path joint_train/carpet_grid

# carpet + grid shared discriminator 63.2 56.5
python joint_train.py --data_path datasets/mvtec --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 200 --log_path joint_train/shared_disc

# TODO why not using standard augmentations
  # T.Normalize()
  # T.ToTensor()
# TODO based on this
  # Try T.ColorJitter()
