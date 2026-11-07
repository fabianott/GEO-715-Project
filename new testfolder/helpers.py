"""Helper functions for saving figures."""

from pathlib import Path

import matplotlib.pyplot as plt

# The figures folder sits next to this file (in the same testfolder)
FIG_DIR = Path(__file__).resolve().parent / "figures"


def save_fig(name, fig=None, fig_dir=FIG_DIR, dpi=300, formats=("png", "pdf"), **kwargs):
    """Save a figure as PNG and PDF into the figures folder.

    Parameters
    ----------
    name : str
        File name without extension, e.g. "ring_analysis".
    fig : matplotlib.figure.Figure, optional
        Figure to save. If None, the current figure (plt.gcf()) is used.
    fig_dir : str or Path, optional
        Output folder. Defaults to the "figures" folder next to this file.
    dpi : int, optional
        Resolution for raster formats (PNG). Default 300.
    formats : tuple of str, optional
        File formats to save. Default ("png", "pdf").
    **kwargs
        Passed on to fig.savefig().

    Returns
    -------
    list of Path
        Paths of the saved files.
    """
    if fig is None:
        fig = plt.gcf()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    kwargs.setdefault("bbox_inches", "tight")

    saved = []
    for ext in formats:
        path = fig_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, **kwargs)
        saved.append(path)

    return saved
