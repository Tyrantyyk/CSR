## Coordinated Sparse Recovery of Label Noise(CSR)

**在cifar10n数据集上rand1噪声类型的运行指令**：
python train_cifarN.py --batch_size 128 --noise_type rand1 --num_epochs 300 --lr 0.05 \
--dataset cifar10 --num_class 10 --rho_range 0.8,0.8 --threshold 0.9 --tau 0.95 --pretrain_ep 10 --seed 0 --lr_u 10 --lr_v 10 \
--lr_trans 0.00001 --start_expand 250 --data_path ../data/cifar-10-batches-py --noise_path ../data/cifar10N/CIFAR-10_human.pt
