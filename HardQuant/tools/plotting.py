import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm

def _surface_tokens_x_channels(mat: np.ndarray,
                               title: str,
                               clip_percentile: float | None = None,
                               elev: float = 30,
                               azim: float = -60,
                               auto_stride: bool = True,
                               ax=None,
                               *,
                               zlim: tuple[float, float] | None = None,
                               color_norm=None,
                               cmap_name: str = "coolwarm"):
    """
    mat is (T, C). Plot a 3D surface with X=channels, Y=tokens, Z=values.
    If zlim and color_norm are provided, they are used (so multiple plots share scale).
    """
    T, C = mat.shape
    sx = sy = 1
    if auto_stride:
        target = 200_000
        cur = T * C
        if cur > target:
            scale = (cur / target) ** 0.5
            sy = max(1, int(np.floor(scale)))
            sx = max(1, int(np.floor(scale)))
        if C > 768:
            sx = max(sx, int(np.ceil(C / 768)))

    y_idx = np.arange(0, T, sy)  # tokens
    x_idx = np.arange(0, C, sx)  # channels
    Xg, Yg = np.meshgrid(x_idx, y_idx, indexing="xy")
    Z = mat[np.ix_(y_idx, x_idx)].astype(np.float32, copy=False)

    # Local (per-plot) limits only if shared ones weren't provided
    if zlim is None or color_norm is None:
        if clip_percentile is None:
            z_min, z_max = float(np.nanmin(Z)), float(np.nanmax(Z))
        else:
            z_min = float(np.nanpercentile(Z, 100 - clip_percentile))
            z_max = float(np.nanpercentile(Z, clip_percentile))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min = float(np.nanmin(Z))
            z_max = float(np.nanmax(Z))
            if z_min == z_max:
                z_max = z_min + 1e-12
        zlim = (z_min, z_max)
        if zlim[0] < 0 < zlim[1]:
            color_norm = TwoSlopeNorm(vmin=zlim[0], vcenter=0.0, vmax=zlim[1])
        else:
            color_norm = Normalize(vmin=zlim[0], vmax=zlim[1])

    cmap = cm.get_cmap(cmap_name)
    facecolors = cmap(color_norm(Z))

    if ax is None:
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        own_fig = True
    else:
        own_fig = False

    ax.plot_surface(
        Xg, Yg, Z,
        facecolors=facecolors,
        linewidth=0, antialiased=True,
        shade=False
    )
    mappable = cm.ScalarMappable(norm=color_norm, cmap=cmap)
    mappable.set_array(Z)
    plt.colorbar(mappable, ax=ax, shrink=0.65, pad=0.06, label="activation")

    ax.set_xlabel("Channel index (0 … C-1)")
    ax.set_ylabel("Token index (0 … T-1)")
    ax.set_zlabel("|activation|" if np.all(Z >= 0) else "activation")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(*zlim)  # <-- shared z-axis limits

    if len(x_idx) > 12:
        ax.set_xticks(x_idx[:: max(1, len(x_idx)//12)])
    else:
        ax.set_xticks(x_idx)
    if len(y_idx) > 12:
        ax.set_yticks(y_idx[:: max(1, len(y_idx)//12)])
    else:
        ax.set_yticks(y_idx)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


@torch.no_grad()
def plot_layer_3d(
    model: nn.Module,
    enc_batch: dict,
    layer_name: str,
    which: str = "inputs",
    model_b: nn.Module | None = None,
    title_a: str = "model",
    title_b: str = "model_ft",
    max_channels: int | None = None,
    max_tokens: int | None = None,
    use_abs: bool = True,
    clip_percentile: float | None = None,
    elev: float = 30,
    azim: float = -60,
):
    """
    Capture the selected module's inputs/outputs and plot a 3D surface (T x C).
    If model_b is provided, both panels share the SAME z-axis and color scale.
    """
    def _capture(model_, batch_, name_, which_):
        mod = _get_module_by_name(model_, name_)
        captured = {}

        def hook_pre(m, args):
            x = args[0] if isinstance(args, (tuple, list)) else args
            captured["X"] = x.detach().cpu()

        def hook_post(m, args, out):
            y = out
            if isinstance(y, (list, tuple)):
                y = next((t for t in y if isinstance(t, torch.Tensor)), None)
            if isinstance(y, torch.Tensor):
                captured["X"] = y.detach().cpu()

        h = (mod.register_forward_pre_hook(hook_pre) if which_ == "inputs"
             else mod.register_forward_hook(hook_post))

        _ = model_(**batch_)  # run forward to trigger hook
        h.remove()
        if "X" not in captured:
            raise RuntimeError(f"Hook for '{name_}' ({which_}) did not fire.")
        t = _to_TC(captured["X"])
        if use_abs:
            t = t.abs()
        mat = t.numpy().astype(np.float32, copy=False)
        T, C = mat.shape
        if max_tokens   is not None: T = min(T, max_tokens)
        if max_channels is not None: C = min(C, max_channels)
        mat = mat[:T, :C]
        R = _rowwise_R(mat)
        return R, mat

    def _to_device(batch, device):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                out[k] = type(v)(t.to(device) for t in v)
            else:
                out[k] = v
        return out

    # Capture A
    model.eval()
    enc_a = _to_device(enc_batch, next(model.parameters()).device)
    Ra, mata = _capture(model, enc_a, layer_name, which)
    result = {"A": {"R": Ra, "mat": mata}}

    # If B exists, capture it too, then compute SHARED limits
    matb = None
    if model_b is not None:
        model_b.eval()
        enc_b = _to_device(enc_batch, next(model_b.parameters()).device)
        Rb, matb = _capture(model_b, enc_b, layer_name, which)
        result["B"] = {"R": Rb, "mat": matb}

        # Compute shared z-limits (optionally percentile-clipped) across BOTH
        if clip_percentile is None:
            z_min = float(np.nanmin([mata.min(), matb.min()]))
            z_max = float(np.nanmax([mata.max(), matb.max()]))
        else:
            lo_a = np.nanpercentile(mata, 100 - clip_percentile)
            hi_a = np.nanpercentile(mata, clip_percentile)
            lo_b = np.nanpercentile(matb, 100 - clip_percentile)
            hi_b = np.nanpercentile(matb, clip_percentile)
            z_min = float(min(lo_a, lo_b))
            z_max = float(max(hi_a, hi_b))

        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min = float(min(np.nanmin(mata), np.nanmin(matb)))
            z_max = float(max(np.nanmax(mata), np.nanmax(matb)))
            if z_min == z_max:
                z_max = z_min + 1e-12

        # Shared color normalization
        if z_min < 0 < z_max:
            shared_norm = TwoSlopeNorm(vmin=z_min, vcenter=0.0, vmax=z_max)
        else:
            shared_norm = Normalize(vmin=z_min, vmax=z_max)
        shared_zlim = (z_min, z_max)
    else:
        shared_norm = None
        shared_zlim = None

    # Plot
    ncols = 2 if model_b is not None else 1
    fig = plt.figure(figsize=(12*ncols, 7))
    ax1 = fig.add_subplot(1, ncols, 1, projection="3d")
    _surface_tokens_x_channels(
        mata,
        title=f"{title_a} • {layer_name} • {which} • shape={mata.shape}",
        clip_percentile=clip_percentile, elev=elev, azim=azim, ax=ax1,
        zlim=shared_zlim, color_norm=shared_norm
    )

    if model_b is not None:
        ax2 = fig.add_subplot(1, ncols, 2, projection="3d")
        _surface_tokens_x_channels(
            matb,
            title=f"{title_b} • {layer_name} • {which} • shape={matb.shape}",
            clip_percentile=clip_percentile, elev=elev, azim=azim, ax=ax2,
            zlim=shared_zlim, color_norm=shared_norm
        )

    plt.tight_layout()
    plt.show()
    return result


def _get_module_by_name(model: nn.Module, dotted: str) -> nn.Module:
    """Resolve 'transformer.h.4.mlp.c_proj' -> module object."""
    cur = model
    if not dotted:
        return cur
    for part in dotted.split('.'):
        # handle numeric indices inside ModuleList/Sequential, e.g., 'h.4'
        if part.isdigit():
            cur = cur[int(part)]
        else:
            if not hasattr(cur, part):
                raise AttributeError(f"Module '{dotted}' not found at '{part}'")
            cur = getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise TypeError(f"Resolved object for '{dotted}' is not a nn.Module")
    return cur

def _to_TC(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure a (T, C) 2D view for visualization.
    Accepts shapes like (B,T,C), (T,C), (N,C), (C,), etc.
    """
    if t.dim() == 3:      # (B, T, C)
        return t[0]
    elif t.dim() == 2:    # (T, C) or (N, C)
        return t
    elif t.dim() == 1:    # (C,)
        return t.view(1, -1)
    else:
        return t.view(-1, t.size(-1))

def _rowwise_R(mat: np.ndarray) -> np.ndarray:
    """
    For (T,C): R_t = (||x_t||_inf / (||x_t||_2 / sqrt(C)))^2
    """
    T, C = mat.shape
    l_inf = np.max(np.abs(mat), axis=1)
    l2 = np.linalg.norm(mat, axis=1)
    denom = l2 / np.sqrt(C)
    R = np.zeros(T, dtype=np.float64)
    mask = denom > 0
    R[mask] = (l_inf[mask] / denom[mask])**2
    return R
