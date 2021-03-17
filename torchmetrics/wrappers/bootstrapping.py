# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional, Union, List, Callable
from copy import deepcopy

import torch
from torch import nn

from torchmetrics.metric import Metric
from torchmetrics.utilities import apply_to_collection


def _bootstrap_sampler(tensor: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
    """  """
    if size is None:
        size = tensor.shape[0]
    idx = torch.multinomial(
        torch.ones(tensor.shape[0], device=tensor.device),
        num_samples=size,
        replacement=True
    )
    print('pytorch', idx)
    return tensor[idx]


class BootStrapper(Metric):
    def __init__(self, base_metric: Metric, 
                 num_bootstraps: int = 10,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None
    ) -> None:
        """ 
        Use to turn a metric into a bootstrapped metric that can automate the process of getting confidence
        intervals for metric values. This wrapper class basically keeps multiple copies of the same base metric
        in memory and whenever ``update`` or ``forward`` is called, all input tensors are resampled
        (with replacement) along the first dimension.
     
        .. note:: Different from all other metrics, bootstrapped metrics has additional
            arguments in its ``compute``  method determining what should be returned.
        
        Args:
            base_metric: 
                base metric class to wrap
            compute_on_step:
                Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
            dist_sync_on_step:
                Synchronize metric state across processes at each ``forward()``
                before returning the value at the step
            process_group:
                Specify the process group on which synchronization is called.
                default: ``None`` (which selects the entire world)
            dist_sync_fn:
                Callback that performs the allgather operation on the metric state. When ``None``, DDP
                will be used to perform the allgather.
        
        Example::
            >>> from torchmetrics.wrappers import BootStrapper
            >>> from torchmetrics import Accuracy
            >>> _ = torch.manual_seed(0)
            >>> base_metric = Accuracy()
            >>> bootstrap = BootStrapper(base_metric, num_bootstraps=20)
            >>> bootstrap.update(torch.randint(5, (20,)), torch.randint(5, (20,)))
            >>> output = bootstrap.compute(mean=True, std=True)
            >>> mean, std = output
            >>> print(mean, std)
            tensor(0.4950) tensor(0.1677)
        
        """
        super().__init__(
            compute_on_step,
            dist_sync_on_step,
            process_group,
            dist_sync_fn
        )
        if not isinstance(base_metric, Metric):
            raise ValueError("Expected base metric to be an instance of torchmetrics.Metric"
                            f" but received {base_metric}")

        self.metrics = nn.ModuleList([deepcopy(base_metric) for _ in range(num_bootstraps)])
        self.num_bootstraps = num_bootstraps
        
    def update(self, *args, **kwargs):
        """ Updates the state of the base metric. Any tensor passed in will be bootstrapped
        along dimension 0
        """
        for idx in range(self.num_bootstraps):
            args = apply_to_collection(args, torch.Tensor, _bootstrap_sampler)
            kwargs = apply_to_collection(kwargs, torch.Tensor, _bootstrap_sampler)
            self.metrics[idx].update(*args, **kwargs)

    def compute(
            self, 
            mean: bool = True, 
            std: bool = True, 
            quantile: Optional[Union[float, torch.Tensor]] = None,
            raw: bool = False
        ) -> List[torch.Tensor]:
        """ Computes the metric value.
        Args:
            mean: if `True` return the mean of the bootstraps
            std: if `True` return the standard diviation of the bootstraps
            quantile: if given, returns the quantile of the bootstraps
            raw: if `True`, return all bootstrapped values
        """
        computed_vals = torch.stack([m.compute() for m in self.metrics], dim=0)
        output = []
        if mean:
            output.append(computed_vals.mean(dim=0))
        if std:
            output.append(computed_vals.std(dim=0))
        if quantile is not None:
            output.append(torch.quantile(computed_vals, quantile))
        if raw:
            output.append(computed_vals)
        return output
