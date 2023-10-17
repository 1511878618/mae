

1. `python main_linprobe.py --data_path ../cadFace/data/imgFolder/new_recrop256 --nb_classes 2 --output_dir output_dir/linear --log_dir output_dir/linear --finetune /storage/xutingfeng/cadFace/models/MAEFaceCAD_az_v1.pt --batch_size 128`
2. `python main_finetune.py --batch_size 8 --epochs 40 --input_size 224 --lr 1e-3 --data_path ../cadFace/data/imgFolder/new_recrop256 --nb_classes 2 --finetune ../cadFace/models/MAEFaceCAD_az_v1.pt`