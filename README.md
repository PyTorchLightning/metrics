<div align="center">

<img src="docs/source/_static/images/logo.png" width="400px">

**Machine learning metrics for distributed, scalable PyTorch applications.**

---

<p align="center">
  <a href="#what-is-torchmetrics">What is Torchmetrics</a> •
  <a href="#implementing-your-own-metric">Implementing a metric</a> •
  <a href="#build-in-metrics">Built-in metrics</a> •
  <a href="https://torchmetrics.readthedocs.io/en/stable/">Docs</a> •
  <a href="#community">Community</a> •
  <a href="#license">License</a>
</p>

---

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchmetrics)](https://pypi.org/project/torchmetrics/)
[![PyPI Status](https://badge.fury.io/py/torchmetrics.svg)](https://badge.fury.io/py/torchmetrics)
[![PyPI Status](https://pepy.tech/badge/torchmetrics)](https://pepy.tech/project/torchmetrics)
[![Conda](https://img.shields.io/conda/v/conda-forge/torchmetrics?label=conda&color=success)](https://anaconda.org/conda-forge/torchmetrics)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/torchmetrics/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/metrics/blob/master/LICENSE)

[![CI testing - base](https://github.com/PyTorchLightning/metrics/actions/workflows/ci_test-base.yml/badge.svg?branch=master&event=push)](https://github.com/PyTorchLightning/metrics/actions/workflows/ci_test-base.yml)
[![Build Status](https://dev.azure.com/PytorchLightning/Metrics/_apis/build/status/PyTorchLightning.metrics?branchName=master)](https://dev.azure.com/PytorchLightning/Metrics/_build/latest?definitionId=3&branchName=master)
[![codecov](https://codecov.io/gh/PyTorchLightning/metrics/branch/master/graph/badge.svg?token=NER6LPI3HS)](https://codecov.io/gh/PyTorchLightning/metrics)
[![Documentation Status](https://readthedocs.org/projects/torchmetrics/badge/?version=latest)](https://torchmetrics.readthedocs.io/en/latest/?badge=latest)

---

</div>


## Installation

Simple installation from PyPI
```bash
pip install torchmetrics
```

<details>
  <summary>Other installations</summary>

Install using conda
```bash
conda install torchmetrics
```

Pip from source

```bash
# with git
pip install git+https://github.com/PytorchLightning/metrics.git@master
```

Pip from archive
```bash
pip install https://github.com/PyTorchLightning/metrics/archive/master.zip
```

</details>

---

## What is Torchmetrics
TorchMetrics is a collection of 25+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:

* A standardized interface to increase reproducibility
* Reduces boilerplate
* Automatic accumulation over batches
* Metrics optimized for distributed-training
* Automatic synchronization between multiple devices

You can use TorchMetrics with any PyTorch model or with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) to enjoy additional features such as:

* Module metrics are automatically placed on the correct device.
* Native support for logging metrics in Lightning to reduce even more boilerplate.

## Using TorchMetrics

### Module metrics

The [module-based metrics](https://pytorchlightning.github.io/metrics/references/modules.html) contain internal metric states (similar to the parameters of the PyTorch module) that automate accumulation and synchronization across devices!

* Automatic accumulation over multiple batches
* Automatic synchronization between multiple devices
* Metric arithmetic

**This can be run on CPU, single GPU or multi-GPUs!**

For the single GPU/CPU case:

```python
import torch
# import our library
import torchmetrics 

# initialize metric
metric = torchmetrics.Accuracy()

n_batches = 10
for i in range(n_batches):
    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))

    # metric on current batch
    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")    

# metric on all batches using custom accumulation
acc = metric.compute()
print(f"Accuracy on all data: {acc}")
```

Module metric usage remains the same when using multiple GPUs or multiple nodes. 

<details>
  <summary>Example using DDP</summary>

```python

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torchmetrics

def metric_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # initialize model
    metric = torchmetrics.Accuracy()

    # define a model and append your metric to it
    # this allows metric states to be placed on correct accelerators when
    # .to(device) is called on the model
    model = nn.Linear(10, 10)
    model.metric = metric
    model = model.to(rank)

    # initialize DDP
    model = DDP(model, device_ids=[rank])

    n_epochs = 5
    # this shows iteration over multiple training epochs
    for n in range(n_epochs):

        # this will be replaced by a DataLoader with a DistributedSampler
        n_batches = 10
        for i in range(n_batches):
            # simulate a classification problem
            preds = torch.randn(10, 5).softmax(dim=-1)
            target = torch.randint(5, (10,))

            # metric on current batch
            acc = metric(preds, target)
            if rank == 0:  # print only for rank 0
                print(f"Accuracy on batch {i}: {acc}")    

        # metric on all batches and all accelerators using custom accumulation
        # accuracy is same across both accelerators
        acc = metric.compute()
        print(f"Accuracy on all data: {acc}, accelerator rank: {rank}")

        # Reseting internal state such that metric ready for new data
        metric.reset()
    
    # cleanup
    dist.destroy_process_group()

world_size = 2   # number of gpus to parallize over
mp.spawn(metric_ddp, args=(world_size,), nprocs=world_size, join=True)

```
</details>

### Implementing your own Module metric

Implementing your own metric is as easy as subclassing an [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). Simply, subclass `torchmetrics.Metric`
and implement the following methods:

```python
import torch
from torchmetrics import Metric

class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce 
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total
```

### Functional metrics

Similar to [`torch.nn`](https://pytorch.org/docs/stable/nn.html), most metrics have both a [module-based](https://torchmetrics.readthedocs.io/en/latest/references/modules.html) and a [functional](https://torchmetrics.readthedocs.io/en/latest/references/functional.html) version.
The functional versions are simple python functions that as input take [torch.tensors](https://pytorch.org/docs/stable/tensors.html) and return the corresponding metric as a [torch.tensor](https://pytorch.org/docs/stable/tensors.html).

``` python
import torch
# import our library
import torchmetrics

# simulate a classification problem
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))

acc = torchmetrics.functional.accuracy(preds, target)
```

### Implemented metrics

* [Accuracy](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#accuracy)
* [AveragePrecision](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#averageprecision)
* [AUC](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#auc)
* [AUROC](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#auroc)
* [F1](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#f1) 
* [Hamming Distance](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#hamming-distance)
* [ROC](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#roc)
* [ExplainedVariance](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#explainedvariance)
* [MeanSquaredError](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#meansquarederror)
* [R2Score](https://torchmetrics.readthedocs.io/en/latest/references/modules.html#r2score)
* [bleu_score](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#bleu-score-func)
* [embedding_similarity](https://torchmetrics.readthedocs.io/en/latest/references/functional.html#embedding-similarity-func)

And many more!

## Contribute!
The lightning + torchmetric team is hard at work adding even more metrics. 
But we're looking for incredible contributors like you to submit new metrics
and improve existing ones!

Join our [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A) 
to get help becoming a contributor!

## Community
For help or questions, join our huge community on [Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)!

## Citations
We’re excited to continue the strong legacy of open source software and have been inspired over the years by 
Caffe, Theano, Keras, PyTorch, torchbearer, ignite, sklearn and fast.ai. When/if a paper is written about this, 
we’ll be happy to cite these frameworks and the corresponding authors.

## License
Please observe the Apache 2.0 license that is listed in this repository. In addition
the Lightning framework is Patent Pending.
