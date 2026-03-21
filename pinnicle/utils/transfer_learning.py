from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except Exception:  
    torch = None
    nn = None

# container to summarize what transfer learning did
@dataclass
class TransferLearningReport:
    loaded: bool    # bool if weights were loaded
    frozen: int     # number of frozen parameters
    trainable: int  # number of trainable parameters
    missing_keys: List[str]     # keys expected but not found
    unexpected_keys: List[str]  # keys unexpected but found
    used_strict: bool   # bool if strict loading mode used 
    how_frozen: str     # description of freezing strategy used

"""General steps:
    1. Check if PyTorch is available and is the backend being used
    2. Gets the underlying nn based on what is passed in
        - If you pass PINNICLE object it extracts model.net
        - If you pass a raw PyTorch model it will be used directly
    ## Code works directly on PyTorch nn if either PINNICLE object or raw PyTorch model is passed in ##
    3. If transfer learning is enabled 
        - load the weights
        - strip prefixes (so the saved weights names match the names in the current model)
    4. Load weights into the model
    5. Apply freezing
        - Choose the mode (freeze all layers, freeze a fraction, freeze the first n, none unfreezes everything)
        - Chosse the target (linears=layers (default), params=raw parameters)
    6. Return report
"""

# check if PyTorch is available
def _is_torch_available() -> bool:
    return torch is not None and nn is not None


def _as_net(pinn_or_net: Any) -> "nn.Module":
    """Accepts either
      -a PINNICLE object with `.model.net`
      -a torch.nn.Module
    """
    if not _is_torch_available():
        raise RuntimeError(
            "PyTorch is not available. Transfer learning requires PyTorch backend. "
        )

    if isinstance(pinn_or_net, nn.Module):
        return pinn_or_net

    # handle a PINNICLE convention
    model = getattr(pinn_or_net, "model", None)
    if model is not None:
        net = getattr(model, "net", None)
        if isinstance(net, nn.Module):
            return net

    raise TypeError(
        "Expected a torch.nn.Module or an object with `.model.net`."
    )

# normalize prefix_strip input into a list of strings
def _normalize_prefix_strip(prefix_strip: Any) -> List[str]:
    if prefix_strip is None:
        return []
    if isinstance(prefix_strip, (list, tuple)):
        return [str(p) for p in prefix_strip]
    return [str(prefix_strip)]

# remove prefixes (module., net.) from state_dict keys and returna new stat_dict where prefixes are removed from keys
def _strip_prefixes_from_state_dict(
    state: Mapping[str, Any], prefix_strip: List[str]
) -> Dict[str, Any]:
    """remove prefixes (module., net.) from state_dict keys and returna new stat_dict 
    where prefixes are removed from keys
    """
    if not prefix_strip:
        return dict(state)

    new_state: Dict[str, Any] = {}
    for k, v in state.items():
        nk = k
        changed = True
        # repeatedly strip to handle nested prefixes
        while changed:
            changed = False
            for p in prefix_strip:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        new_state[nk] = v
    return new_state

# load saved PyTorch state_dict 
def _load_state_dict_file(weights_path: str) -> Mapping[str, Any]:
    if not _is_torch_available():
        raise RuntimeError("torch not available")
    
    obj = torch.load(weights_path, map_location="cpu")
    
    # handle common patterns
    if isinstance(obj, Mapping) and "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
        return obj["state_dict"]
    if isinstance(obj, Mapping):
        return obj
    raise TypeError(
        f"Unsupported weights file format at {weights_path!r}. "
    )

# freeze or unfreeze all parameters
def _freeze_all_params(net: "nn.Module", freeze: bool) -> Tuple[int, int]:
    params = list(net.parameters())
    for p in params:
        p.requires_grad = not freeze
    #count frozen vs. trainable
    n_frozen = sum(1 for p in params if not p.requires_grad)
    n_train = sum(1 for p in params if p.requires_grad)
    return n_frozen, n_train

# freeze a fraction of parameters (in parameter order).
def _freeze_by_fraction_params(net: "nn.Module", fraction: float) -> Tuple[int, int]:
    params = list(net.parameters())
    if not params:
        return 0, 0

    # clamp to [0,1]
    fraction = float(fraction)
    if fraction < 0.0:
        fraction = 0.0
    if fraction > 1.0:
        fraction = 1.0

    n_total = len(params)
    n_freeze = int(round(fraction * n_total))

    for i, p in enumerate(params):
        p.requires_grad = False if i < n_freeze else True

    n_frozen = sum(1 for p in params if not p.requires_grad)
    n_train = sum(1 for p in params if p.requires_grad)
    return n_frozen, n_train

# freeze first N parameters
def _freeze_first_n_params(net: "nn.Module", first_n: int) -> Tuple[int, int]:
    params = list(net.parameters())
    if not params:
        return 0, 0

    n_total = len(params)
    n_freeze = max(0, min(int(first_n), n_total))

    for i, p in enumerate(params):
        p.requires_grad = False if i < n_freeze else True

    n_frozen = sum(1 for p in params if not p.requires_grad)
    n_train = sum(1 for p in params if p.requires_grad)
    return n_frozen, n_train

# collect nn.linear layers in the model 
def _iter_linear_modules(net: "nn.Module") -> List["nn.Linear"]:
    if not _is_torch_available():
        return []
    linears: List[nn.Linear] = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            linears.append(m)
    return linears

# freeze based on linear layers not parameters
def _freeze_linear_layers(
    net: "nn.Module",
    mode: str,
    fraction: float = 0.0,
    first_n: int = 0,
) -> Tuple[int, int]:
    linears = _iter_linear_modules(net)
    if not linears:
        # if no linear layers found do nothing
        params = list(net.parameters())
        return (
            sum(1 for p in params if not p.requires_grad),
            sum(1 for p in params if p.requires_grad),
        )

    # decide how many Linear layers to freeze
    if mode == "fraction":
        frac = max(0.0, min(1.0, float(fraction)))
        n_freeze_layers = int(round(frac * len(linears)))
    elif mode == "first_n":
        n_freeze_layers = max(0, min(int(first_n), len(linears)))
    elif mode == "all":
        n_freeze_layers = len(linears)
    elif mode == "none":
        n_freeze_layers = 0
    else:
        n_freeze_layers = 0

    # apply freezing to only selected layers
    for i, layer in enumerate(linears):
        freeze_this = i < n_freeze_layers
        for p in layer.parameters(recurse=False):
            p.requires_grad = not freeze_this

    params = list(net.parameters())
    n_frozen = sum(1 for p in params if not p.requires_grad)
    n_train = sum(1 for p in params if p.requires_grad)
    return n_frozen, n_train

# Main function. apply transfer learning
def apply_transfer_learning(
    pinn_or_net: Any,
    cfg: Optional[Mapping[str, Any]] = None,
) -> TransferLearningReport:
    """
    expected cfg shape:
    {
      "enabled": True,
      "weights_path": "path/to/state.pt",
      "strict": False,
      "prefix_strip": ["net.", "model.", "module."],
      "freeze": {
        "mode": "fraction" | "first_n" | "all" | "none",
        "fraction": 0.8,
        "first_n": 0,
        "target": "linears" | "params" | "all"
      }
    }
    """
    import warnings

    if cfg is None:
        cfg = {}

    enabled = bool(cfg.get("enabled", True)) # enabled:True so apply transfer learning 
    net = _as_net(pinn_or_net)

    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    used_strict = bool(cfg.get("strict", False)) # PyTorch will still run if some keys are missing or do not match

    loaded = False

    # Load weights 
    weights_path = str(cfg.get("weights_path", "") or "").strip()
    if enabled and weights_path:
        state = _load_state_dict_file(weights_path)
        prefix_strip = _normalize_prefix_strip(cfg.get("prefix_strip", []))
        state = _strip_prefixes_from_state_dict(state, prefix_strip)

        res = net.load_state_dict(state, strict=used_strict)

        try:
            missing_keys = list(getattr(res, "missing_keys", []) or [])
            unexpected_keys = list(getattr(res, "unexpected_keys", []) or [])
        except Exception:
            missing_keys, unexpected_keys = [], []

        loaded = True

    # freeze parameters if enabled
    frozen = 0
    trainable = 0
    how_frozen = "none"

    freeze_cfg = cfg.get("freeze", {}) or {}
    mode = str(freeze_cfg.get("mode", "none") or "none").lower()

    # default is linears
    target = str(freeze_cfg.get("target", "linears") or "linears").lower()

    # "all" target behaved like param-based freezing
    if target == "all":
        target = "params"

    if target == "parameters":
        target = "params"

    if target == "params":
        warnings.warn(
            "Freezing by parameters may not correspond to layer structure."
        )

    fraction = float(freeze_cfg.get("fraction", 0.0) or 0.0)
    first_n = int(freeze_cfg.get("first_n", 0) or 0)

    if enabled:
        if mode in ("none", ""):
            frozen, trainable = _freeze_all_params(net, freeze=False)
            how_frozen = "unfreeze_all"

        elif mode == "all":
            if target == "linears":
                frozen, trainable = _freeze_linear_layers(net, mode="all")
                how_frozen = "freeze_all_linears"
            else:
                frozen, trainable = _freeze_all_params(net, freeze=True)
                how_frozen = "freeze_all_params"

        elif mode == "fraction":
            if target == "linears":
                frozen, trainable = _freeze_linear_layers(
                    net, mode="fraction", fraction=fraction
                )
                how_frozen = f"freeze_fraction_linears({fraction})"
            else:
                frozen, trainable = _freeze_by_fraction_params(net, fraction=fraction)
                how_frozen = f"freeze_fraction_params({fraction})"

        elif mode == "first_n":
            if target == "linears":
                frozen, trainable = _freeze_linear_layers(
                    net, mode="first_n", first_n=first_n
                )
                how_frozen = f"freeze_first_n_linears({first_n})"
            else:
                frozen, trainable = _freeze_first_n_params(net, first_n=first_n)
                how_frozen = f"freeze_first_n_params({first_n})"

        else:
            # unknown mode then keep current requires_grad (do nothing)
            params = list(net.parameters())
            frozen = sum(1 for p in params if not p.requires_grad)
            trainable = sum(1 for p in params if p.requires_grad)
            how_frozen = f"unknown_mode({mode})"            
    # return summary report
    return TransferLearningReport(
        loaded=loaded,
        frozen=frozen,
        trainable=trainable,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        used_strict=used_strict,
        how_frozen=how_frozen,
    )