from typing import Callable, List

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch import Tensor


def _compute_sklearn_metric(
    metric: Callable, target: List[np.ndarray], preds: List[np.ndarray], behaviour: str
) -> Tensor:
    """ Compute metric with multiple iterations over documents predictions. """
    sk_results = []

    for b, a in zip(target, preds):
        if b.sum() == 0:
            if behaviour == 'skip':
                pass
            elif behaviour == 'pos':
                sk_results.append(1.0)
            else:
                sk_results.append(0.0)
        else:
            res = metric(b, a)
            sk_results.append(res)

    if len(sk_results) > 0:
        return np.mean(sk_results)
    return np.array(0.0)


def _test_against_sklearn(
    sklearn_metric,
    torch_metric,
    size,
    n_documents,
    query_without_relevant_docs_options
) -> None:
    """ Compare PL metrics to standard version. """
    seed_everything(0)

    metric = torch_metric(query_without_relevant_docs=query_without_relevant_docs_options)
    shape = (size, )

    indexes = []
    preds = []
    target = []

    for i in range(n_documents):
        indexes.append(np.ones(shape, dtype=int) * i)
        preds.append(np.random.randn(*shape))
        target.append(np.random.randn(*shape) > 0)

    sk_results = _compute_sklearn_metric(sklearn_metric, target, preds, query_without_relevant_docs_options)
    sk_results = torch.tensor(sk_results)

    indexes_tensor = torch.cat([torch.tensor(i) for i in indexes])
    preds_tensor = torch.cat([torch.tensor(p) for p in preds])
    target_tensor = torch.cat([torch.tensor(t) for t in target])

    # lets assume data are not ordered
    perm = torch.randperm(indexes_tensor.nelement())
    indexes_tensor = indexes_tensor.view(-1)[perm].view(indexes_tensor.size())
    preds_tensor = preds_tensor.view(-1)[perm].view(preds_tensor.size())
    target_tensor = target_tensor.view(-1)[perm].view(target_tensor.size())

    # shuffle ids to require also sorting of documents ability from the lightning metric
    pl_result = metric(indexes_tensor, preds_tensor, target_tensor)

    assert torch.allclose(sk_results.float(), pl_result.float(), equal_nan=False), (
        f"Test failed comparing metric {sklearn_metric} with {torch_metric}: "
        f"{sk_results.float()} vs {pl_result.float()}. "
        f"indexes: {indexes}, preds: {preds}, target: {target}"
    )


def _test_dtypes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    length = 10  # not important in this test

    # check error when `query_without_relevant_docs='error'` is raised correctly
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.rand(size=(length, ), device=device, dtype=torch.float32)
    target = torch.tensor([False] * length, device=device, dtype=torch.bool)

    metric = torchmetric(query_without_relevant_docs='error')
    _assert_error(metric, ValueError, indexes, preds, target)

    # check ValueError with invalid `query_without_relevant_docs` argument
    _assert_error(torchmetric, ValueError, query_without_relevant_docs='casual_argument')

    # check input dtypes
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.tensor([0] * length, device=device, dtype=torch.float32)
    target = torch.tensor([0] * length, device=device, dtype=torch.int64)

    metric = torchmetric(query_without_relevant_docs='error')

    # check error on input dtypes are raised correctly
    _assert_error(metric, ValueError, indexes.bool(), preds, target)
    _assert_error(metric, ValueError, indexes, preds.bool(), target)
    _assert_error(metric, ValueError, indexes, preds, target.float())


def _test_input_shapes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    metric = torchmetric(query_without_relevant_docs='error')

    # check input shapes are checked correclty
    elements_1, elements_2, elements_3 = np.random.choice(20, size=3, replace=False)
    indexes = torch.tensor([0] * elements_1, device=device, dtype=torch.int64)
    preds = torch.tensor([0] * elements_2, device=device, dtype=torch.float32)
    target = torch.tensor([0] * elements_3, device=device, dtype=torch.int64)

    _assert_error(metric, ValueError, indexes, preds, target)


def _assert_error(function, error, *args, **kwargs):
    """ Assert that `function(*args, **kwargs)` raises `error`. """
    try:
        function(*args, **kwargs)
        assert False  # assert exception is raised
    except Exception as e:
        assert isinstance(e, error)
