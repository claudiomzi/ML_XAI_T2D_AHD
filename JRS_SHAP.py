"""
================================================================================
Multi-Panel Publication Visualization for SHAP Attribution Plots
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    Utility module for composing cross-regional multi-panel SHAP summary figures
    (Panels A, B, and C representing Tuscany, Lombardy, and Sicily, respectively)
    into publication-quality vector PDFs (300 DPI) for the JRSS-C manuscript.
================================================================================
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Config_RETIN import SHAP_VALUE


def create_shap_multipanel_from_images(cohort: str = "Global") -> None:
    """
    Compose regional SHAP summary plots into a unified horizontal multi-panel figure.

    Parameters
    ----------
    cohort : str, default="Global"
        Sub-cohort identifier: 'Global', 'M' (Male), or 'F' (Female).

    Returns
    -------
    None
        Saves 'SHAP_multipanel_{cohort}.pdf' in the designated SHAP_VALUE directory.
    """
    img_paths = [
        Path(SHAP_VALUE) / f"SHAP_TY_{cohort}.png",
        Path(SHAP_VALUE) / f"SHAP_LY_{cohort}.png",
        Path(SHAP_VALUE) / f"SHAP_SY_{cohort}.png"
    ]

    panel_labels = ["(A) Tuscany", "(B) Lombardy", "(C) Sicily"]

    fig, axs = plt.subplots(1, 3, figsize=(22, 7))

    for ax, path, label in zip(axs, img_paths, panel_labels):
        if not path.exists():
            print(f"[Warning] Image not found at {path}. Skipping panel.")
            ax.axis("off")
            continue

        img = mpimg.imread(str(path))
        ax.imshow(img)
        ax.axis("off")

        # Overlay publication panel label
        ax.text(
            0.02, 0.98, label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="left",
            color="black",
            bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.8)
        )

    plt.tight_layout()
    outpath = Path(SHAP_VALUE) / f"SHAP_multipanel_{cohort}.pdf"
    plt.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Visualization] Multi-panel figure successfully saved to: {outpath}")


if __name__ == "__main__":
    create_shap_multipanel_from_images("Global")
