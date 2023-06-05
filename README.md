This is the official repo for the paper [Distilling Efficient Language-Specific Models for Cross-Lingual Transfer](https://arxiv.org/abs/2306.01709), forked from [composable-sft](https://github.com/cambridgeltl/composable-sft).

## Installation

First, install Python >= 3.9 and PyTorch >= 1.9 (earlier versions may work but haven't been tested), e.g. using conda:
```
conda create -n bistil python=3.10
conda activate bistil
pip install torch
```

Then download and install composable-sft:
```
git clone https://github.com/AlanAnsell/bistil.git
cd bistil
pip install -e .
```
