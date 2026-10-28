"""Post-hoc calibration on held-out data (never on the test split).

The handoff: the confidence head's output sets the solver's data-term
weights, those weights feed the Hessian, and so a miscalibrated confidence
head produces miscalibrated intervals. Calibrate before trusting them.

Two scalars are fitted by minimising held-out negative log-likelihood:

* conf_temperature T:  p = sigmoid(logit / T)
* level_std_scale s:   sigma_level = s * exp(logvar / 2)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, iters: int = 200) -> float:
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def fit_std_scale(err: np.ndarray, std: np.ndarray) -> float:
    """Closed-form maximum-likelihood scale for Gaussian errors: s^2 = mean((e/sigma)^2)."""
    return float(np.sqrt(np.mean((err / np.maximum(std, 1e-6)) ** 2)))


def calibrate(model, results: list[dict]) -> dict:
    """`results` from evaluate.run_inference on held-out data with T=1, s=1."""
    logits = np.concatenate([r["conf_logit"] for r in results])
    tgt = np.concatenate([r["conf_target"] for r in results])
    T = fit_temperature(logits, tgt)

    errs, stds = [], []
    for r in results:
        m = r["level_mask"].astype(bool)
        errs.append(r["level_mu"][m] - r["level"][m])
        stds.append(r["level_std"][m] / float(model.level_std_scale))
    s = fit_std_scale(np.concatenate(errs), np.concatenate(stds))

    model.conf_temperature.fill_(T)
    model.level_std_scale.fill_(s)
    log.info("fitted conf temperature %.3f, level std scale %.3f", T, s)
    return {"conf_temperature": T, "level_std_scale": s}
