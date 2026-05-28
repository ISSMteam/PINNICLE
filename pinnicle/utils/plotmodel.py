import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
import deepxde.backend as bkd
import numpy as np


def plotmodel(pinn, path="", filename="", **kwargs):
    """plotmodel"""
    pass


def _as_1d_float_array(a):
    """Return a 1-D ndarray; masked values are converted to NaN."""
    if np.ma.isMaskedArray(a):
        return np.asarray(np.ma.asarray(a).filled(np.nan)).ravel()
    return np.asarray(a).ravel()


def _get_nan_mask(a):
    """Return a 1-D boolean mask for masked or non-finite values."""
    if np.ma.isMaskedArray(a):
        arr = np.ma.asarray(a)
        return (np.ma.getmaskarray(arr) | ~np.isfinite(arr.filled(np.nan))).ravel()
    return ~np.isfinite(np.asarray(a).ravel())


def _predict_in_batches(model, X_nn, batch_size=None):
    """Call model.model.predict once, or in chunks to avoid memory spikes."""
    if batch_size is None or X_nn.shape[0] <= batch_size:
        return model.model.predict(X_nn)

    parts = []
    for i in range(0, X_nn.shape[0], batch_size):
        parts.append(model.model.predict(X_nn[i:i + batch_size]))
    return np.concatenate(parts, axis=0)


def _extract_prediction(sol_pred, keylist, key):
    """Extract a named output from the NN prediction matrix."""
    if key in keylist:
        ind = keylist.index(key)
        return sol_pred[:, ind].ravel()

    if key == "bed":
        ind_s = keylist.index("s")
        ind_H = keylist.index("H")
        return (sol_pred[:, ind_s] - sol_pred[:, ind_H]).ravel()

    raise ValueError(f"Key {key} not found in model output variables and is not 'bed'.")


def _build_plot_cache(X, Y, prefer_grid=True):
    """Precompute either a regular-grid layout or one triangulation, then reuse it."""
    cache = {
        "X": X,
        "Y": Y,
        "is_grid": False,
        "order": None,
        "shape": None,
        "Xg": None,
        "Yg": None,
        "triangles": None,
    }

    if prefer_grid:
        x_unique = np.unique(X)
        y_unique = np.unique(Y)
        nx = x_unique.size
        ny = y_unique.size

        # This detects the common case where X/Y come from a flattened meshgrid.
        if nx * ny == X.size:
            order = np.lexsort((X, Y))
            Xs = X[order].reshape(ny, nx)
            Ys = Y[order].reshape(ny, nx)
            if np.allclose(Xs, x_unique[None, :]) and np.allclose(Ys, y_unique[:, None]):
                cache.update(
                    {
                        "is_grid": True,
                        "order": order,
                        "shape": (ny, nx),
                        "Xg": Xs,
                        "Yg": Ys,
                    }
                )
                return cache

    # Fallback for genuinely scattered points. Build this only once.
    cache["triangles"] = mpl.tri.Triangulation(X, Y)
    return cache


def _plot_from_cache(ax, cache, data, mask=None, scaling=1.0, iscatter=False,
                     rasterized=True, scatter_size=1, **kwargs):
    """Fast plotting helper used by data, prediction, and difference panels."""
    X = cache["X"]
    Y = cache["Y"]
    z = np.asarray(data).ravel().astype(float, copy=True) * scaling

    if mask is None:
        mask = ~np.isfinite(z)
    else:
        mask = np.asarray(mask).ravel() | ~np.isfinite(z)

    if iscatter:
        # Avoid drawing invalid points; this is faster for large scatter plots.
        keep = ~mask
        scatter_kwargs = kwargs.copy()
        scatter_kwargs.setdefault("s", scatter_size)
        scatter_kwargs.setdefault("rasterized", rasterized)
        return ax.scatter(X[keep], Y[keep], c=z[keep], **scatter_kwargs)

    if cache["is_grid"]:
        # Much faster than tripcolor for regular gridded data.
        Zg = z[cache["order"]].reshape(cache["shape"])
        Mg = mask[cache["order"]].reshape(cache["shape"])
        Zg = np.where(Mg, np.nan, Zg)

        mesh_kwargs = kwargs.copy()
        shading = mesh_kwargs.pop("shading", "auto")
        mesh_kwargs.setdefault("rasterized", rasterized)
        return ax.pcolormesh(cache["Xg"], cache["Yg"], Zg, shading=shading, **mesh_kwargs)

    # Scattered fallback: reuse one triangulation instead of rebuilding it.
    z_ma = np.ma.array(z, mask=mask)
    tri_kwargs = kwargs.copy()
    tri_kwargs.setdefault("rasterized", rasterized)
    return ax.tripcolor(cache["triangles"], z_ma, **tri_kwargs)


def plotmodelcompare(model, dataname, output, scaling=1, diffrange=None,
                     iscatter=False, **kwargs):
    """Plot data, PINN prediction, and prediction-data difference.

    Performance-oriented changes compared with the original version:
      1. Predict only once at the data coordinates.
      2. Reuse a single triangulation for all panels when the data are scattered.
      3. Use pcolormesh automatically when X/Y form a regular grid.
      4. Rasterize heavy artists by default, which keeps notebooks/PDFs responsive.

    Extra kwargs
    ------------
    batch_size : int or None
        Predict in chunks. Useful for very large grids to reduce memory use.
    prefer_grid : bool
        Try to detect a regular grid and use pcolormesh. Default is True.
    rasterized : bool
        Rasterize pcolormesh/tripcolor/scatter artists. Default is True.
    scatter_size : float
        Marker size used when iscatter=True. Default is 1.
    max_points : int or None
        Optional quick-look downsampling. If set and data contain more points,
        an evenly spaced subset is plotted and predicted.
    """
    figsize = kwargs.pop("figsize", (12, 5))
    batch_size = kwargs.pop("batch_size", None)
    prefer_grid = kwargs.pop("prefer_grid", True)
    rasterized = kwargs.pop("rasterized", True)
    scatter_size = kwargs.pop("scatter_size", 1)
    max_points = kwargs.pop("max_points", None)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    data_obj = model.model_data.data[dataname]
    X = _as_1d_float_array(data_obj.X_dict["x"])
    Y = _as_1d_float_array(data_obj.X_dict["y"])
    data_raw = data_obj.data_dict[output]
    data = _as_1d_float_array(data_raw).astype(float, copy=False)
    mask = _get_nan_mask(data_raw)

    # Remove points with invalid coordinates before triangulation/prediction.
    valid_xy = np.isfinite(X) & np.isfinite(Y)
    X = X[valid_xy]
    Y = Y[valid_xy]
    data = data[valid_xy]
    mask = mask[valid_xy]

    if max_points is not None and X.size > max_points:
        idx = np.linspace(0, X.size - 1, int(max_points), dtype=int)
        X = X[idx]
        Y = Y[idx]
        data = data[idx]
        mask = mask[idx]

    X_nn = np.column_stack((X, Y))
    sol_pred = _predict_in_batches(model, X_nn, batch_size=batch_size)
    keylist = model.params.nn.output_variables
    pred = _extract_prediction(sol_pred, keylist, output)

    if diffrange is None:
        diffrange = np.nanmax(np.abs(data[~mask])) * scaling

    cache = _build_plot_cache(X, Y, prefer_grid=(prefer_grid and not iscatter))

    # Data
    im = _plot_from_cache(
        axs[0], cache, data, mask=mask, scaling=scaling,
        iscatter=iscatter, rasterized=rasterized, scatter_size=scatter_size,
        **kwargs,
    )
    axs[0].set_title(f"Data: {output}")
    fig.colorbar(im, ax=axs[0], shrink=0.8, location="top")

    # Prediction at the same coordinates as the data. This avoids a second
    # expensive prediction on a separate plotting grid.
    im = _plot_from_cache(
        axs[1], cache, pred, mask=mask, scaling=scaling,
        iscatter=iscatter, rasterized=rasterized, scatter_size=scatter_size,
        **kwargs,
    )
    axs[1].set_title("Prediction")
    fig.colorbar(im, ax=axs[1], shrink=0.8, location="top")

    # Difference: prediction - data
    diff_kwargs = kwargs.copy()
    diff_kwargs.update({"cmap": "bwr", "vmin": -0.1 * diffrange, "vmax": 0.1 * diffrange})
    im = _plot_from_cache(
        axs[2], cache, pred - data, mask=mask, scaling=scaling,
        iscatter=iscatter, rasterized=rasterized, scatter_size=scatter_size,
        **diff_kwargs,
    )
    axs[2].set_title("Difference")
    fig.colorbar(im, ax=axs[2], shrink=0.8, location="top")

    for ax in axs:
        ax.set_aspect("equal", adjustable="box")

    return axs


# Backward-compatible wrappers -------------------------------------------------

def plotprediction(axs, model, key, X=None, Y=None, scaling=1, resolution=200,
                   operator=None, **kwargs):
    """Plot predictions of a model output variable."""
    if X is None or Y is None:
        bbox = model.domain.bbox()
        x = np.linspace(bbox[0, 0], bbox[1, 0], resolution)
        y = np.linspace(bbox[0, 1], bbox[1, 1], resolution)
        X, Y = np.meshgrid(x, y)

    X = _as_1d_float_array(X)
    Y = _as_1d_float_array(Y)
    valid_xy = np.isfinite(X) & np.isfinite(Y)
    X = X[valid_xy]
    Y = Y[valid_xy]

    sol_pred = _predict_in_batches(model, np.column_stack((X, Y)), kwargs.pop("batch_size", None))
    data = _extract_prediction(sol_pred, model.params.nn.output_variables, key)

    if operator is not None:
        data = operator(data)

    return plot2d(axs, X, Y, data, scaling=scaling, **kwargs)


def plotdiff(axs, model, X, Y, data, key, scaling=1, iscatter=False, **kwargs):
    """Plot prediction minus data."""
    X = _as_1d_float_array(X)
    Y = _as_1d_float_array(Y)
    data = _as_1d_float_array(data).astype(float, copy=False)
    mask = _get_nan_mask(data)

    valid_xy = np.isfinite(X) & np.isfinite(Y)
    X = X[valid_xy]
    Y = Y[valid_xy]
    data = data[valid_xy]
    mask = mask[valid_xy]

    sol_pred = _predict_in_batches(model, np.column_stack((X, Y)), kwargs.pop("batch_size", None))
    pred = _extract_prediction(sol_pred, model.params.nn.output_variables, key)

    if iscatter:
        return plotscatter(axs, X[~mask], Y[~mask], scaling * (pred[~mask] - data[~mask]), **kwargs)
    return plot2d(axs, X, Y, pred - data, mask=mask, scaling=scaling, **kwargs)


def plot2d(axs, X, Y, data, mask=None, scaling=1, **kwargs):
    """Plot 2D data using pcolormesh for regular grids and tripcolor otherwise."""
    X = _as_1d_float_array(X)
    Y = _as_1d_float_array(Y)
    data = _as_1d_float_array(data).astype(float, copy=False)

    valid_xy = np.isfinite(X) & np.isfinite(Y)
    X = X[valid_xy]
    Y = Y[valid_xy]
    data = data[valid_xy]
    if mask is not None:
        mask = np.asarray(mask).ravel()[valid_xy]

    cache = _build_plot_cache(X, Y, prefer_grid=kwargs.pop("prefer_grid", True))
    return _plot_from_cache(axs, cache, data, mask=mask, scaling=scaling, **kwargs)


def plottriangle(axs, triangles, data, **kwargs):
    """Plot a triangular mesh."""
    kwargs.setdefault("rasterized", True)
    return axs.tripcolor(triangles, data, **kwargs)


def plotscatter(axs, X, Y, data, **kwargs):
    """Plot 2D data as scattered points."""
    kwargs.setdefault("s", 1)
    kwargs.setdefault("rasterized", True)
    return axs.scatter(X, Y, c=data, **kwargs)
