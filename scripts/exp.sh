# pretrain + fine tune
# carpet -> grid (id 2 -> 13)

# Image Auc 95.9 AP 98.8
# Pixel Auc 95.2 AP 56.5
python train_DRAEM.py --data_path datasets/mvtec --obj_id 2 --sample_rate 1.0 --anomaly_source_path datasets/dtd/images \
  --gpu_id 0 --lr 0.0001 --bs 6 --epochs 100 --log_path logs/carpet_sp_100

# joint training

# TODO why not using standard augmentations
  # T.Normalize()
  # T.ToTensor()
# TODO based on this
  # Try T.ColorJitter()
