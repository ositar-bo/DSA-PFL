# DSA-PFL

Official research code preparation for **Dynamic Subspace Adaptation for
Personalized Federated Learning (DSA-PFL)**.

DSA-PFL was called **DSR-PFL** (Dynamic Subspace Routing) in the original
experiment scripts. The rename does not change the algorithm: a client builds a
layer-wise Conceptor from fixed local calibration samples, estimates temporal
subspace maturity from consecutive communication rounds, and filters the global
model proposal in the output-channel space of each convolutional or linear
layer.

## Method in one equation

For client `i`, layer `l`, and communication round `t`, let

```text
Delta = W_global - W_local
mu    = exp(-drift / tau)
```

where `drift` is the normalized Frobenius distance between consecutive
Conceptors. DSA-PFL solves

```text
min_Z  0.5 ||Z - Delta||_F^2 + 0.5 * lambda * mu * tr(Z^T C Z),
```

whose unique solution is

```text
Z* = (I + lambda * mu * C)^(-1) Delta.
```

The personalized initialization for the next round is `W_local + Z*`. Batch
normalization state and other non-Conv/Linear state follow the global aggregate
and are not Conceptor-routed.

## Repository layout

```text
DSA-PFL/
├── dsapfl/
│   ├── config.py       # frozen experiment configuration and validation
│   ├── data.py         # CIFAR/FashionMNIST/Tiny-ImageNet and partitions
│   ├── models.py       # ResNet-18 variants used by the experiments
│   ├── routing.py      # Conceptor construction, drift, and DSA fusion
│   └── trainer.py      # federated training, metrics, and checkpoints
├── tests/
│   └── test_routing.py # numerical invariants of the routing operator
├── train.py            # command-line entry point
├── requirements.txt
├── pyproject.toml
└── README_CN.md
```

The repository intentionally excludes datasets, checkpoints, experiment
outputs, SSH helpers, credentials, and the FedSSM/FedSSM-o reproduction code.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the no-data algorithm self-test before downloading a dataset:

```bash
python train.py --self-test
python -m unittest discover -s tests -v
```

## Reproduce the released DSA-PFL protocol

The following defaults match the completed DSR-PFL/DSA-PFL experiment runner:

- 10 clients and full participation;
- 100 rounds, one local epoch;
- SGD, learning rate `0.001`, momentum `0.9`, batch size `64`;
- torch seed `543`, partition seed `500`;
- warm-up `5`, aperture `10`, `tau=0.25`, `lambda=1`;
- at most 256 fixed calibration samples per client;
- uniform server aggregation;
- independently instantiated client/server models in a deterministic RNG
  sequence;
- the public-code-compatible FedSSM partition procedure and merged official
  train/test pools, followed by an 80/20 split inside each client.

```bash
# CIFAR-10, Dirichlet(0.1)
python train.py --dataset cifar10 --split dirichlet --alpha 0.1 \
  --method dsapfl --rounds 100 --download

# CIFAR-10, Dirichlet(0.5)
python train.py --dataset cifar10 --split dirichlet --alpha 0.5 \
  --method dsapfl --rounds 100 --download

# CIFAR-10, public-code-compatible 50-shard setting
python train.py --dataset cifar10 --split pathological --num-shards 50 \
  --method dsapfl --rounds 100 --download

# CIFAR-100
python train.py --dataset cifar100 --split dirichlet --alpha 0.1 \
  --method dsapfl --rounds 100 --download
```

Use `--method both` to run a same-code-path FedAvg sanity baseline before
DSA-PFL. This is **not** FedSSM or FedSSM-o.

Tiny-ImageNet expects an extracted `tiny-imagenet-200` directory:

```bash
python train.py --dataset tinyimagenet --data-root /path/to/tiny-imagenet-200 \
  --split dirichlet --alpha 0.1 --method dsapfl --rounds 100
```

## Protocol switches

The release exposes alternatives without silently changing the historical
defaults:

```bash
# Disjoint client allocation instead of the public-code-compatible allocation
python train.py ... --partition-mode disjoint

# Sample-count-weighted aggregation
python train.py ... --aggregation sample

# A common initial state for every client and the server
python train.py ... --initialization shared

# Algebraically equivalent dual Conceptor construction when d > m
python train.py ... --conceptor-solver auto
```

These switches define different experimental protocols. Results generated with
them must not be mixed with the historical default results without rerunning all
compared methods.

## Important reproducibility note

The public-code-compatible partition intentionally reproduces the earlier
comparison environment. In that implementation, clients independently sample
class indices or shards, so samples can overlap across clients and some samples
can remain unassigned. It is retained solely to reproduce the completed paired
experiments. For new studies, prefer `--partition-mode disjoint` and state the
choice in the paper.

Likewise, the historical code used uniform aggregation and deterministic but
independent model instantiation. If a manuscript states sample-weighted
aggregation, a common initial client model, or a disjoint partition, either
correct the manuscript or rerun every baseline under those settings.

## Outputs

Each run writes:

- `metadata.json`: full configuration and partition/initialization hashes;
- `results.csv`: one row per method and communication round;
- `client_results.csv`: per-client round metrics;
- `conceptor_stats.csv`: layer-wise drift, maturity, and occupancy;
- `summary.json`: final, best, last-10, worst-client, and stability metrics;
- `best_dsapfl.pt`: best DSA-PFL checkpoint.

## Citation and license

Update `CITATION.cff` with the final authors, paper venue, DOI, and repository URL
before release. No open-source license has been selected yet; choose one and
replace `LICENSE.template` before making the repository public.
