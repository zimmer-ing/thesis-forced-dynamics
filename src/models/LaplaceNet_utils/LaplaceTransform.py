import torch


def discrete_laplace_transform(x, s_in, t, mode="sample", chunk_size=1000):
    """
    Computes the discrete Laplace transform of a signal x(t), such as the known forcing.

    Args:
        x (Tensor): Real signal, (B, N, F).
        s_in (Tensor): Complex query points, (B, num_s, 1, ilt_terms),
                       shared across signal features.
        t (Tensor): Time points, (B, N, 1), with at least two points and
                    the same constant time step across the batch.
        mode (str): "batch" evaluates all query points together, "chunk"
                    evaluates query-point chunks, and "sample" loops over batch items.
        chunk_size (int): Maximum query points per chunk in "chunk" mode.

    Returns:
        Tensor: Complex transform values, (B, num_s, ilt_terms, F).
    """
    dt = t[0, 1, 0] - t[0, 0, 0]
    B, N, F = x.shape
    _, S, _, T_terms = s_in.shape

    if mode == "sample":
        s_res = torch.zeros(B, S, F, T_terms, device=x.device, dtype=s_in.dtype)
        for b in range(B):
            t_s = t[b].view(1, -1)
            x_b = x[b]
            s_flat = s_in[b].view(-1, 1)
            kernel = torch.exp(-s_flat * t_s)
            result = torch.sum(x_b.unsqueeze(0) * kernel.unsqueeze(-1), dim=1) * dt
            s_res[b] = result.view(S, T_terms, F).permute(0, 2, 1)
        return s_res.permute(0, 1, 3, 2)

    if mode not in {"batch", "chunk"}:
        raise ValueError("Invalid mode. Use 'sample', 'batch', or 'chunk'.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    query_count = S * T_terms
    effective_chunk_size = query_count if mode == "batch" else min(chunk_size, query_count)
    s_flat = s_in.reshape(B, query_count)
    t_batched = t.transpose(1, 2)
    x_complex = x.to(s_in.dtype)
    transformed_chunks = []
    for start in range(0, query_count, effective_chunk_size):
        end = min(start + effective_chunk_size, query_count)
        kernel = torch.exp(-s_flat[:, start:end].unsqueeze(-1) * t_batched)
        transformed_chunks.append(torch.bmm(kernel, x_complex) * dt)

    transformed = torch.cat(transformed_chunks, dim=1)
    return transformed.view(B, S, T_terms, F)


def laplace_from_fft(x_t, s, t, chunk_size=1000, mode="sample"):
    """
    Computes X(s) from Fourier coefficients, including the start-time factor exp(-s * t0).

    Args:
        x_t (Tensor): Real signal, such as the known forcing, (B, N, F).
        s (Tensor): Complex query points, (B, num_s, 1, ilt_terms),
                    shared across signal features.
        t (Tensor): Time points, (B, N, 1), with at least two points and
                    the same constant time step across the batch.
        chunk_size (int): Maximum query points per chunk in "chunk" mode.
        mode (str): "batch" evaluates all query points together, "chunk"
                    evaluates query-point chunks, and "sample" loops over batch items.

    Returns:
        Tensor: Complex transform values, (B, num_s, ilt_terms, F).
    """
    import torch

    B, num_s, one, ilt_terms = s.shape
    F = x_t.shape[-1]
    dt = t[0, 1, 0] - t[0, 0, 0]  # Time step (assumed constant)

    t0 = t[:, 0, 0].view(B, 1)

    N = x_t.shape[1]

    # Fourier coefficients normalized by N * dt.
    alpha = torch.fft.fft(x_t, dim=1) / (N * dt)

    # Negative angular FFT frequencies for the denominator s + i*w.
    w_fft = -2.0 * torch.pi * torch.fft.fftfreq(N, d=dt).to(alpha.device)

    S = num_s * ilt_terms
    s_reshaped = s.view(B, S)

    if mode == "chunk":
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        Xs_chunks = []
        for start in range(0, S, chunk_size):
            s_chunk = s_reshaped[:, start:min(start + chunk_size, S)]
            denom_chunk = s_chunk.unsqueeze(-1) + 1j * w_fft.view(1, 1, N)  # (B, S_chunk, N)

            Xs_chunk = torch.matmul(1 / denom_chunk, alpha) * dt

            shift_factor = torch.exp(-s_chunk * t0)
            Xs_chunk = Xs_chunk * shift_factor.unsqueeze(-1)

            Xs_chunks.append(Xs_chunk)
        Xs = torch.cat(Xs_chunks, dim=1)

    elif mode == "batch":
        denom = s_reshaped.unsqueeze(-1) + 1j * w_fft.view(1, 1, N)
        Xs = torch.matmul(denom.reciprocal(), alpha) * dt
        shift_factor = torch.exp(-s_reshaped * t0)
        Xs = Xs * shift_factor.unsqueeze(-1)

    elif mode == "sample":
        Xs_list = []
        for b in range(B):
            alpha_b = alpha[b]
            s_b = s_reshaped[b].unsqueeze(-1)
            w_b = w_fft.unsqueeze(0)
            denom = s_b + 1j * w_b
            X_b = (alpha_b.unsqueeze(0) / denom.unsqueeze(-1)).sum(dim=1) * dt
            shift_factor = torch.exp(-s_b * t0[b])
            X_b = X_b * shift_factor
            Xs_list.append(X_b)
        Xs = torch.stack(Xs_list, dim=0)

    else:
        raise ValueError("Invalid mode. Use 'chunk', 'batch', or 'sample'.")

    Xs = Xs.view(B, num_s, ilt_terms, F)
    return Xs
