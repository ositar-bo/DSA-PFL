# DSA-PFL

Official implementation of **Dynamic Subspace Adaptation for Personalized
Federated Learning (DSA-PFL)**.

DSA-PFL personalizes the global model update with client-specific temporal
subspaces. Each client builds layer-wise Conceptors from local calibration
features, estimates subspace stability across communication rounds, and filters
the global update before the next round of local training.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- NumPy

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

The default configuration uses 10 clients, 100 communication rounds, one local
epoch, SGD with learning rate `0.001`, momentum `0.9`, and batch size `64`.

```bash
# CIFAR-10, Dirichlet(0.1)
python train.py --dataset cifar10 --split dirichlet --alpha 0.1 --download

# CIFAR-10, Dirichlet(0.5)
python train.py --dataset cifar10 --split dirichlet --alpha 0.5 --download

# CIFAR-10, pathological 50-shard split
python train.py --dataset cifar10 --split pathological --num-shards 50 --download

# CIFAR-100
python train.py --dataset cifar100 --split dirichlet --alpha 0.1 --download
```

For Tiny-ImageNet, provide the extracted `tiny-imagenet-200` directory:

```bash
python train.py --dataset tinyimagenet \
  --data-root /path/to/tiny-imagenet-200 \
  --split dirichlet --alpha 0.1
```

Run `python train.py --help` to view all options.

## Main parameters

- warm-up rounds: `5`
- Conceptor aperture: `10`
- temporal scale (`tau`): `0.25`
- subspace strength (`lambda`): `1.0`
- torch seed: `543`
- partition seed: `500`

Training outputs are written to `runs/` by default.
