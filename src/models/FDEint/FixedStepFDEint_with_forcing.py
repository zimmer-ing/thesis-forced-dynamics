# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Bernd Zimmering
# See licenses/FDEint-MIT.txt for the original license.
# modified from FDEint version 0.1.1 URL:https://github.com/zimmer-ing/FDEint/blob/6b3bb11fe38f260f6d5fa05f8d3149b4b08e37c7/FDEint/FixedStepFDEint.py

import torch
from tqdm import tqdm
import warnings
import torch.nn as nn

DEBUG = False


def torchgamma(x):
    return torch.exp(torch.special.gammaln(x))


def _prepare_time_tensor(t: torch.Tensor, y0: torch.Tensor, dtype=torch.float32):
    """
    Ensures t has shape (batch, n_time, 1).
    """
    if len(y0.shape) == 1:
        batch_size = 1
    else:
        batch_size = y0.shape[0]

    if len(t.shape) == 1:
        t = t.unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1)
    elif len(t.shape) == 2:
        if t.shape[-1] == 1:
            t = t.unsqueeze(0)
        else:
            t = t.unsqueeze(-1)
    return t.to(dtype)


def _prepare_input_tensor(u: torch.Tensor, t: torch.Tensor, y0: torch.Tensor, dtype=torch.float32):
    """
    Ensures u has shape (batch, n_time, dim_u), aligned with t.
    Expected use case: u is given on the same discrete time grid as t.
    """
    if u is None:
        return None

    if len(y0.shape) == 1:
        batch_size = 1
    else:
        batch_size = y0.shape[0]


    t = _prepare_time_tensor(t, y0, dtype=dtype)
    n_time = t.shape[1]

    if len(u.shape) == 1:
        u = u.unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1)
    elif len(u.shape) == 2:
        if u.shape[0] == n_time:
            u = u.unsqueeze(0).repeat(batch_size, 1, 1)
        elif u.shape[0] == batch_size and u.shape[1] == n_time:
            u = u.unsqueeze(-1)
        else:
            raise ValueError("Ambiguous shape for u. Expected (n_time, dim_u) or (batch, n_time).")
    elif len(u.shape) == 3:
        pass
    else:
        raise ValueError("u must have shape (n_time,), (n_time, dim_u), (batch, n_time), or (batch, n_time, dim_u).")

    if u.shape[0] != batch_size:
        raise ValueError("Batch size of u must match batch size of y0.")
    if u.shape[1] != n_time:
        raise ValueError("Time dimension of u must match time dimension of t.")

    return u.to(dtype)


def _linear_interpolate_batched(t_query: torch.Tensor, t_grid: torch.Tensor, values: torch.Tensor):
    """
    Linear interpolation for batched time series.
    """
    batch_size, m, _ = t_query.shape
    _, n, dim_u = values.shape
    device = t_query.device
    out = torch.zeros((batch_size, m, dim_u), device=device, dtype=values.dtype)

    for b in range(batch_size):
        tq = t_query[b, :, 0]
        tg = t_grid[b, :, 0]
        vg = values[b]

        idx = torch.searchsorted(tg, tq, right=True)
        idx = torch.clamp(idx, 1, n - 1)

        t0 = tg[idx - 1]
        t1 = tg[idx]
        v0 = vg[idx - 1]
        v1 = vg[idx]

        w = ((tq - t0) / (t1 - t0)).unsqueeze(-1)
        out[b] = v0 + w * (v1 - v0)

    return out


def _call_f(f, t_in, y_in, u_in=None):
    """
    Calls f either as f(t, y) or f(t, y, u).
    """
    if u_in is None:
        return f(t_in, y_in)
    return f(t_in, y_in, u_in)


def FDEint(
    f: nn.Module,
    t: torch.Tensor,
    y0: torch.Tensor,
    alpha: torch.Tensor,
    h=None,
    dtype=torch.float32,
    DEBUG=False,
    return_internals=False,
    u: torch.Tensor = None,
    u_fn=None,
):
    """
    Solves a fractional differential equation using a predictor-corrector method.
    """

    assert 0 < alpha <= 1, "Alpha must be between 0 and 1"
    assert h is None or h > 0, "Step size must be greater than 0 if provided"
    assert len(y0.shape) == 1 or len(y0.shape) == 2, "Initial condition y0 must have shape (dim_y,) or (batch_size, dim_y)"
    assert not (u is not None and u_fn is not None), "Use either u or u_fn, not both."

    device = y0.device

    if len(y0.shape) == 1:
        y0 = y0.unsqueeze(0)

    t = _prepare_time_tensor(t, y0, dtype=dtype).to(device)
    u = _prepare_input_tensor(u, t, y0, dtype=dtype)
    y0 = y0.to(dtype)
    alpha = alpha.to(dtype).squeeze()

    assert y0.shape[0] == t.shape[0], "Batch sizes of y0 and t must be the same"

    alpha_is_in_autograd = alpha.requires_grad

    dt_min = torch.min(torch.diff(t.squeeze()))
    if h is None:
        h = dt_min
    elif dt_min < h and dt_min - h > 1e-6:
        warnings.warn(
            f"The minimum time difference in desired time points ({dt_min}) is smaller than the step size h ({h}). Adjusting h to {dt_min}.",
            UserWarning
        )
        h = dt_min

    batch_size, dim_y = y0.shape
    N = int(torch.ceil((t.max() - t.min()) / h).item()) + 1

    y_internal = torch.zeros((batch_size, N + 1, dim_y), device=device, dtype=dtype)
    t_internal = torch.zeros((batch_size, N + 1), device=device, dtype=dtype)
    fk_mem = torch.zeros_like(y_internal)

    y_internal[:, 0, :] = y0
    t_internal[:, 0] = t[:, 0, 0]

    k = torch.arange(1, N + 1, device=device, dtype=dtype)
    b = ((k ** alpha) - (k - 1) ** alpha).unsqueeze(-1)
    a = ((k + 1) ** (alpha + 1) - 2 * k ** (alpha + 1) + (k - 1) ** (alpha + 1)).unsqueeze(-1)
    b = torch.cat([torch.zeros((1, 1), device=device, dtype=dtype), b], dim=0)
    a = torch.cat([torch.zeros((1, 1), device=device, dtype=dtype), a], dim=0)

    gamma_alpha1 = torchgamma(alpha + 1)
    gamma_alpha2 = torchgamma(alpha + 2)

    # initial forcing
    if u_fn is not None:
        u0 = u_fn(t[:, 0, :])
    elif u is not None:
        u0 = u[:, 0, :]
    else:
        u0 = None

    f0 = _call_f(f, t[:, 0, :], y0, u0)
    fk_mem[:, 0, :] = f0.clone() if alpha_is_in_autograd else f0

    y_new = y0
    t_prev_exp = t[:, 0, :]
    u_prev = u0
    kn = torch.arange(0, N, device=device, dtype=torch.long)


    t_query_cache = None
    if u is not None:
        t_query_cache = t_internal.unsqueeze(-1)

    for j in tqdm(range(1, N + 1), desc="Time Steps Progress", disable=not DEBUG):
        if j > 1:
            fkj = _call_f(f, t_prev_exp, y_new, u_prev)
            fk_mem[:, j - 1, :] = fkj.clone() if alpha_is_in_autograd else fkj

        t_act = (j * h).repeat(batch_size).to(dtype)
        t_internal[:, j] = t_act

        t_act_exp = t_act.unsqueeze(-1)


        if u_fn is not None:
            u_act = u_fn(t_act_exp)
        elif u is not None:

            tq = t_act_exp.unsqueeze(1)
            u_act = _linear_interpolate_batched(tq, t, u).squeeze(1)
        else:
            u_act = None

        fk = fk_mem[:, :j, :].clone() if alpha_is_in_autograd else fk_mem[:, :j, :]

        bjk = b[j - kn[:j]]
        y_p = y0 + (h ** alpha / gamma_alpha1) * torch.sum(bjk * fk, dim=1)

        ajk = a[(j - kn[:j])[1:]]

        f_pred = _call_f(f, t_act_exp, y_p, u_act)

        y_new = y0 + (h ** alpha / gamma_alpha2) * (
            f_pred
            + ((j - 1) ** (alpha + 1) - (j - 1 - alpha) * j ** alpha) * f0
            + torch.sum(ajk * fk[:, 1:, :], dim=1)
        )

        y_internal[:, j, :] = y_new.clone() if alpha_is_in_autograd else y_new
        t_prev_exp = t_act_exp
        u_prev = u_act

    sol = get_outputs(y_internal, t_internal, t)
    if not return_internals:
        return sol
    return sol, t_internal, y_internal


def get_outputs(y_internal, t_internal, t):
    batch_size, num_internal_points = t_internal.shape
    _, num_desired_points, _ = t.shape
    _, _, features = y_internal.shape

    idx = torch.searchsorted(t_internal, t.squeeze(-1), right=True)
    idx = torch.clamp(idx, 1, t_internal.shape[1] - 1)
    idx_y = idx.unsqueeze(-1).repeat(1, 1, features)

    t0 = torch.gather(t_internal, 1, idx - 1).unsqueeze(-1)
    t1 = torch.gather(t_internal, 1, idx).unsqueeze(-1)
    y0 = torch.gather(y_internal, 1, idx_y - 1)
    y1 = torch.gather(y_internal, 1, idx_y)

    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
