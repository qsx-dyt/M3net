# Indian
python main.py --dataset=IndianPines --patch_size=11 --epochs=20 --SpaMask_ratio=0.6 --flag=pretrain
python main.py --dataset=IndianPines --patch_size=11 --samples=5 --epochs=50 --pt_model=IndianPines_11_0.6_pt.pth --flag=finetune --runs=5
python main.py --dataset=IndianPines --patch_size=11 --samples=5 --model_path=output/IndianPines_seed0.pth --flag=test --seed=0

# PaviaU
python main.py --dataset=PaviaU --patch_size=11 --epochs=20 --SpaMask_ratio=0.6 --flag=pretrain
python main.py --dataset=PaviaU --patch_size=11 --samples=5 --epochs=50 --pt_model=PaviaU_11_0.6_pt.pth --flag=finetune --runs=5
python main.py --dataset=PaviaU --patch_size=11 --samples=5 --model_path=output/PaviaU_seed1111.pth --flag=test --seed=1111

# HC
python main.py --dataset=HC --patch_size=11 --epochs=20 --SpaMask_ratio=0.6 --flag=pretrain
python main.py --dataset=HC --patch_size=11 --samples=5 --epochs=50 --pt_model=HC_11_0.6_pt.pth --flag=finetune --runs=5
python main.py --dataset=HC --patch_size=11 --samples=5 --model_path=output/HC_seed1111.pth --flag=test --seed=1111


# test: model_path和seed与微调一致
python main.py --dataset=? --patch_size=11 --samples=5 --model_path=output/? --flag=test --seed=?
