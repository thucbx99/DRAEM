# pretrain + fine tune
# carpet -> grid (id 2 -> 13)

# Image Auc 95.9 AP 98.8
# Pixel Auc 95.2 AP 56.5
python train_DRAEM.py --data_path datasets/mvtec --obj_id 2 --sample_rate 1.0 --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 100 --log_path logs/carpet_sp_100

# Ratio 100% Image Auc 100 Pixel AP 66.3
# Ratio 50% Image Auc 99.3 Pixel AP 61.7
# Ratio 30% Image Auc 99.8 Pixel AP 64.0
# Ratio 10% Image Auc 99.6 Pixel AP 53.0
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
# with both generator and discriminator
# Ratio 100% Pixel AP 59.4
# Ratio 10% Pixel AP 49.6
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --pretrained-generative logs/carpet_sp_100/checkpoints/latest_generative.pth \
  --pretrained-discriminative logs/carpet_sp_100/checkpoints/latest_discriminative.pth \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 1000 --log_path logs/grid_sp_10_finetune_ori_lr

python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --pretrained-generative logs/carpet_sp_100/checkpoints/latest_generative.pth \
  --pretrained-discriminative logs/carpet_sp_100/checkpoints/latest_discriminative.pth \
  --gpu_id 1 --lr 0.00003 --bs 6 --epochs 1000 --log_path logs/grid_sp_10_finetune_0_3_lr

# discriminator only
# 70.0
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --pretrained-discriminative logs/carpet_sp_100/checkpoints/latest_discriminative.pth \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 1000 --log_path logs/grid_sp_10_finetune_ori_lr

# 55.4
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --pretrained-discriminative logs/carpet_sp_100/checkpoints/latest_discriminative.pth \
  --gpu_id 1 --lr 0.00003 --bs 6 --epochs 1000 --log_path logs/grid_sp_10_finetune_0_3_lr

# baseline
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --sample_rate 0.1 --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 1000 --log_path logs/grid_sp_10

# joint training

# carpet only
python train_DRAEM.py --data_path datasets/mvtec --obj_id 2 --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 100 --log_path joint_train/carpet_only

# grid only
python train_DRAEM.py --data_path datasets/mvtec --obj_id 13 --anomaly_source_path datasets/dtd/images \
  --gpu_id 1 --lr 0.0001 --bs 6 --epochs 100 --log_path joint_train/grid_only

# carpet + grid
python joint_train.py --data_path datasets/mvtec --anomaly_source_path datasets/dtd/images \
  --gpu_id 2 --lr 0.0001 --bs 6 --epochs 100 --log_path joint_train/carpet_grid

# TODO why not using standard augmentations
  # T.Normalize()
  # T.ToTensor()
# TODO based on this
  # Try T.ColorJitter()
