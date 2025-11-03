import numpy as np


def _plt():
    """Late import matplotlib to avoid hard dependency during pure preprocessing."""
    from matplotlib import pyplot as plt  # type: ignore import-error

    return plt


def plot_price_hist_log(price: np.ndarray) -> None:
    """Histogram gia tren thang log; clip tren 99.5%."""
    plt = _plt()
    mask = np.isfinite(price) & (price > 0)
    price = price[mask]
    if price.size == 0:
        return
    hi = np.percentile(price, 99.5)
    bins = np.logspace(
        np.log10(max(price.min(), 1e-6)),
        np.log10(max(hi, 1e-6)),
        40,
    )
    plt.figure()
    plt.hist(np.clip(price, None, hi), bins=bins, edgecolor="none")
    plt.xscale("log")
    plt.xlabel("price (log scale)")
    plt.ylabel("count")
    plt.title("Phan phoi gia (log)")
    plt.tight_layout()
    plt.show()


def plot_min_nights_hist(min_nights: np.ndarray) -> None:
    """Histogram minimum_nights (clip 99%)."""
    plt = _plt()
    mn = min_nights[np.isfinite(min_nights)]
    if mn.size == 0:
        return
    hi = np.percentile(mn, 99.0)
    plt.figure()
    plt.hist(np.clip(mn, None, hi), bins=40, edgecolor="none")
    plt.xlabel("minimum_nights (clipped 99%)")
    plt.ylabel("count")
    plt.title("Phan phoi minimum_nights")
    plt.tight_layout()
    plt.show()


def plot_scatter_map(lon: np.ndarray, lat: np.ndarray, neigh_group: np.ndarray) -> None:
    """Scatter theo toa do, to mau theo neighbourhood_group."""
    plt = _plt()
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    ng = neigh_group[ok]
    if lon.size == 0:
        return
    groups, inv = np.unique(ng, return_inverse=True)
    plt.figure()
    sc = plt.scatter(lon, lat, c=inv, s=4, alpha=0.5, cmap="tab10")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("Diem cho thue theo khu")
    cbar = plt.colorbar(sc, ticks=np.arange(groups.size))
    cbar.ax.set_yticklabels(groups.tolist())
    plt.tight_layout()
    plt.show()
