# DSA-PFL 代码说明

本目录是准备上传 GitHub 的独立版 **DSA-PFL（Dynamic Subspace Adaptation
for Personalized Federated Learning，动态子空间适应个性化联邦学习）**。

早期实验代码中的方法名是 DSR-PFL。DSA-PFL 只是论文命名更新，核心算法
没有变化：客户端用固定本地校准样本构造逐层 Conceptor，利用相邻通信轮的
Conceptor 漂移估计表示成熟度，再对服务器提供的全局模型差分进行方向级调节。

## 快速运行

```bash
pip install -r requirements.txt
python train.py --self-test
python -m unittest discover -s tests -v

python train.py --dataset cifar10 --split dirichlet --alpha 0.1 \
  --method dsapfl --rounds 100 --download
```

默认参数与已经完成的 DSR-PFL/DSA-PFL 实验保持一致：10 客户端、100 轮、
每轮 1 个本地 epoch、SGD 学习率 0.001、momentum 0.9、batch size 64、
torch seed 543、partition seed 500，以及 `warmup=5 / aperture=10 /
tau=0.25 / lambda=1`。

## 与旧实验严格对应的三个设置

```bash
# Dirichlet(0.1)
python train.py --dataset cifar10 --split dirichlet --alpha 0.1 --download

# Dirichlet(0.5)
python train.py --dataset cifar10 --split dirichlet --alpha 0.5 --download

# 公开代码兼容的 50-shard Pathological
python train.py --dataset cifar10 --split pathological --num-shards 50 --download
```

将 `cifar10` 改为 `cifar100` 或 `fashionmnist` 即可运行相应数据集。
Tiny-ImageNet 需要把 `--data-root` 指向解压后的 `tiny-imagenet-200`。

## 必须注意的协议事实

为了复现已有结果，默认值保留了旧实验的实际行为：

1. 服务器使用客户端模型的等权平均，而不是按样本量加权；
2. 客户端与服务器模型按固定随机数顺序分别初始化，而不是把一个初始权重
   复制给所有客户端；
3. 默认 `fedssm-compatible` 划分会让客户端独立抽取类别样本或 shards，
   因而不同客户端之间可能出现样本重叠，也可能有样本未被分配；
4. 官方训练集和测试集先合并，再进行客户端分配和客户端内部 8:2 切分。

这些行为是历史实验协议的一部分，不能在不重跑全部对比方法的情况下悄悄
修改。新实验可以显式使用：

```bash
--partition-mode disjoint   # 客户端之间不重叠
--aggregation sample       # 按本地训练样本数加权
--initialization shared    # 所有客户端和服务器共享同一初始权重
```

论文中的实验设置必须与实际命令一致。如果论文写的是“样本加权聚合”“共同
初始化”或“无重叠划分”，应当修改论文表述，或者在新协议下重新运行所有方法。

## 发布前检查

- 在 `CITATION.cff` 中填写作者、论文、仓库和 DOI；
- 选择开源许可证，并用正式 `LICENSE` 替换 `LICENSE.template`；
- 不上传 `data/`、`runs/`、模型权重、SSH 脚本、密码或服务器日志；
- 至少执行自检、单元测试和一次 1 轮 CPU/GPU 冒烟实验；
- 论文统一使用 DSA-PFL，不再混用 DSR-PFL。

