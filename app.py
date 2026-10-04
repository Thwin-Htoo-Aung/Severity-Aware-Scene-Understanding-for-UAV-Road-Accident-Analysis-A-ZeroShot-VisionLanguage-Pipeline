"""
YOLO Dataset Explorer — Streamlit app
======================================
Two tabs:
  📊 Statistics  –  run analysis, show all plots inline
  🔍 Explorer    –  browse images with bounding-box overlays

Run:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# ── make sure the package is importable from the project root ──────────────
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image as PILImage, ImageDraw, ImageFont

import streamlit as st

from vista.stats import (
    YOLODatasetAnalyzer,
    _label_path_for_image,
    _parse_label_file,
    _resolve_split_paths,
    ALL_SECTIONS,
    analyze_yolo_dataset,
)

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Dataset Explorer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global style tweaks ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── colour helpers ─────────────────────────────────────────────────────────
MPL_STYLE = "seaborn-v0_8-whitegrid"

# 20 maximally distinct colours (sashamaps.net/docs/resources/20-colors/).
# Each class_id always maps to the same entry → consistent across every plot
# and the image explorer.
_DISTINCT_HEX = [
    "#e6194b",  # 0  red
    "#3cb44b",  # 1  green
    "#4363d8",  # 2  blue
    "#f58231",  # 3  orange
    "#911eb4",  # 4  purple
    "#42d4f4",  # 5  cyan
    "#f032e6",  # 6  magenta
    "#bfef45",  # 7  lime
    "#9a6324",  # 8  brown
    "#469990",  # 9  teal
    "#800000",  # 10 maroon
    "#aaffc3",  # 11 mint
    "#808000",  # 12 olive
    "#dcbeff",  # 13 lavender
    "#ffd8b1",  # 14 apricot
    "#000075",  # 15 navy
    "#a9a9a9",  # 16 grey
    "#fffac8",  # 17 beige
    "#fabed4",  # 18 pink
    "#000000",  # 19 black
]


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def build_class_palette(
    class_names: Dict[int, str],
) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int, int]]]:
    """
    Return two dicts keyed by class_id:
      hex_map  – matplotlib-compatible hex string
      rgb_map  – PIL-compatible (R, G, B) tuple

    Class IDs that exceed the built-in palette cycle through it.
    The mapping is stable regardless of how many classes are present or
    what order they appear in, so every plot and the explorer always use
    the exact same colour for each class.
    """
    hex_map: Dict[int, str] = {}
    rgb_map: Dict[int, Tuple[int, int, int]] = {}
    for cls_id in sorted(class_names.keys()):
        h = _DISTINCT_HEX[cls_id % len(_DISTINCT_HEX)]
        hex_map[cls_id] = h
        rgb_map[cls_id] = _hex_to_rgb(h)
    return hex_map, rgb_map


def _split_colors_mpl(n: int) -> List[str]:
    """Separate palette used for split-comparison bars (not class colours)."""
    cmap = plt.get_cmap("Set2")
    return [matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# ── cached data loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Parsing dataset …")
def load_analyzer(yaml_path: str) -> YOLODatasetAnalyzer:
    az = YOLODatasetAnalyzer(
        yaml_path,
        output_dir=str(Path(yaml_path).parent / "dataset_stats"),
        verbose=False,
        save_json=False,
        save_csv=False,
    )
    az._load()
    az._parse()
    return az


@st.cache_data(show_spinner="Drawing image …")
def render_image_with_boxes(
    image_path: str,
    annotations: List[Tuple[int, float, float, float, float]],
    class_names: Dict[int, str],
    rgb_map: Dict[int, Tuple[int, int, int]],
    line_width: int = 2,
    show_labels: bool = True,
    label_bg: bool = True,
    max_size: int = 1024,
) -> PILImage.Image:
    img = PILImage.open(image_path).convert("RGB")
    W, H = img.size
    if max(W, H) > max_size:
        scale = max_size / max(W, H)
        img = img.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for cls, cx, cy, w, h in annotations:
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        # fall back to a neutral grey for any unseen class id
        color = rgb_map.get(cls, _hex_to_rgb(_DISTINCT_HEX[cls % len(_DISTINCT_HEX)]))
        draw.rectangle([x1, y1, x2, y2], outline=color + (230,), width=line_width)

        if show_labels:
            label = class_names.get(cls, str(cls))
            bbox_text = draw.textbbox((x1, y1), label, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            ty = max(0, y1 - th - 4)
            if label_bg:
                draw.rectangle(
                    [x1, ty, x1 + tw + 6, ty + th + 4],
                    fill=color + (200,),
                )
            draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255, 255), font=font)

    return img


def pil_to_st_image(img: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def fig_to_bytes(fig, fmt: str = "svg", dpi: int = 150) -> bytes:
    """Serialise a matplotlib figure to bytes in the requested format."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=dpi)
    return buf.getvalue()


def _plot_download_buttons(fig, stem: str, dpi: int = 150) -> None:
    """Render SVG + PNG download buttons for a matplotlib figure."""
    c1, c2, *_ = st.columns([1, 1, 6])
    c1.download_button(
        "⬇ SVG", data=fig_to_bytes(fig, "svg", dpi),
        file_name=f"{stem}.svg", mime="image/svg+xml",
        key=f"dl_svg_{stem}",
    )
    c2.download_button(
        "⬇ PNG", data=fig_to_bytes(fig, "png", dpi),
        file_name=f"{stem}.png", mime="image/png",
        key=f"dl_png_{stem}",
    )


# ── matplotlib figure helpers (inline rendering in Streamlit) ──────────────

def fig_class_distribution(az: YOLODatasetAnalyzer, hex_map: Dict[int, str],
                            orientation: str = "horizontal",
                            figsize: Tuple[float, float] = None):
    from collections import Counter
    counts = Counter(a.cls for a in az.annotations)
    sorted_ids = sorted(counts, key=lambda c: counts[c], reverse=True)
    names  = [az._cls_name(c) for c in sorted_ids]
    vals   = [counts[c] for c in sorted_ids]
    colors = [hex_map[c] for c in sorted_ids]

    with plt.style.context(MPL_STYLE):
        h = max(4, len(names) * 0.35)
        fig, ax = plt.subplots(figsize=figsize or (10, h))
        if orientation == "horizontal":
            bars = ax.barh(names, vals, color=colors)
            ax.set_xlabel("Annotations")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + max(vals) * 0.005,
                        bar.get_y() + bar.get_height() / 2,
                        str(v), va="center", fontsize=8)
        else:
            bars = ax.bar(names, vals, color=colors)
            ax.set_ylabel("Annotations")
            plt.xticks(rotation=45, ha="right")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.005,
                        str(v), ha="center", fontsize=8)
        ax.set_title("Class Distribution")
        fig.tight_layout()
    return fig


def fig_spatial_heatmaps(az: YOLODatasetAnalyzer, bins: int = 64,
                          cmap: str = "hot", ncols: int = 3,
                          figsize: Tuple[float, float] = None):
    """
    One figure with subplots: first panel is global (all classes),
    then one panel per class, laid out in rows of `ncols` columns.
    """
    cls_ids = sorted(az.class_names.keys())
    # panels: (title, annotation list)
    panels = [("All classes", az.annotations)] + [
        (az._cls_name(c), [a for a in az.annotations if a.cls == c])
        for c in cls_ids
    ]
    # drop empty per-class panels
    panels = [(t, a) for t, a in panels if a]

    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    cell = 3.5  # inches per cell
    default_fs = (cell * ncols, cell * nrows)

    # pre-compute all heatmaps to find the shared max
    heatmaps = []
    for _, ann in panels:
        cxs = np.array([a.cx for a in ann])
        cys = np.array([a.cy for a in ann])
        hm, _, _ = np.histogram2d(cxs, cys, bins=bins, range=[[0, 1], [0, 1]])
        heatmaps.append(hm)
    vmax = max(hm.max() for hm in heatmaps)

    with plt.style.context(MPL_STYLE):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=figsize or default_fs,
                                 squeeze=False)
        last_im = None
        for idx, ((title, _), hm) in enumerate(zip(panels, heatmaps)):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            last_im = ax.imshow(hm.T, origin="lower", extent=[0, 1, 0, 1],
                                cmap=cmap, aspect="equal",
                                interpolation="bilinear",
                                vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("cx", fontsize=7)
            ax.set_ylabel("cy", fontsize=7)
            ax.tick_params(labelsize=6)
        # hide unused axes
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)
        fig.suptitle("Spatial Heatmaps of BBox Centres", fontsize=11)
        fig.tight_layout()
        # reserve a fixed strip on the right, then place the colorbar there
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(last_im, cax=cbar_ax, label="Annotation count")
    return fig


def fig_combined_main(
    az: YOLODatasetAnalyzer,
    hex_map: Dict[int, str],
    *,
    scatter_alpha: float = 0.35,
    ann_split_filter: Optional[str] = None,
    heatmap_bins: int = 64,
    heatmap_cmap: str = "hot",
    heatmap_ncols: int = 3,
    figsize: Tuple[float, float] = None,
) -> plt.Figure:
    """
    Single figure combining:
      Row 0          – BBox W×H scatter (left)  |  Annotations per Image histogram (right)
      Rows 1 .. end  – Spatial heatmap grid (global + one panel per class, `heatmap_ncols` wide)
    Heatmaps share one colorbar anchored to the right.
    """
    import matplotlib.gridspec as gridspec

    # ── heatmap panels ──────────────────────────────────────────────────────
    cls_ids = sorted(az.class_names.keys())
    hmap_panels = [("All classes", az.annotations)] + [
        (az._cls_name(c), [a for a in az.annotations if a.cls == c]) for c in cls_ids
    ]
    hmap_panels = [(t, a) for t, a in hmap_panels if a]

    n_hmap = len(hmap_panels)
    ncols   = max(heatmap_ncols, 2)          # at least 2 so top row fits
    hmap_rows = (n_hmap + ncols - 1) // ncols

    # pre-compute heatmaps & shared scale
    heatmaps = []
    for _, ann in hmap_panels:
        cxs = np.array([a.cx for a in ann])
        cys = np.array([a.cy for a in ann])
        hm, _, _ = np.histogram2d(cxs, cys, bins=heatmap_bins,
                                   range=[[0, 1], [0, 1]])
        heatmaps.append(hm)
    vmax = max(hm.max() for hm in heatmaps)

    # ── figure & gridspec ──────────────────────────────────────────────────
    # An extra thin column at index `ncols` holds the colorbar exclusively.
    # Both the top row and the heatmap rows are constrained to [0:ncols], so
    # their left and right edges are pixel-perfect aligned.
    total_rows  = 1 + hmap_rows
    cbar_ratio  = 0.08
    cell        = 3.2
    default_fs  = (cell * ncols, cell * total_rows)

    with plt.style.context(MPL_STYLE):
        fig = plt.figure(figsize=figsize or default_fs)
        gs = gridspec.GridSpec(
            total_rows, ncols + 1,          # +1 dedicated colorbar column
            figure=fig,
            width_ratios=[1] * ncols + [cbar_ratio],
            height_ratios=[1] * total_rows,
            hspace=0.45, wspace=0.35,
        )

        # ── scatter: left half of top row ──────────────────────────────────
        n_left = max(1, ncols // 2)
        ax_scatter = fig.add_subplot(gs[0, :n_left])
        ann_all = az.annotations
        ws   = np.array([a.w   for a in ann_all])
        hs   = np.array([a.h   for a in ann_all])
        cids = np.array([a.cls for a in ann_all])
        for c in sorted(set(cids.tolist())):
            mask = cids == c
            ax_scatter.scatter(ws[mask], hs[mask], s=5, alpha=scatter_alpha,
                               color=hex_map[c], label=az._cls_name(c))
        ax_scatter.set_xlabel("Width (norm.)", fontsize=8)
        ax_scatter.set_ylabel("Height (norm.)", fontsize=8)
        ax_scatter.set_title("BBox Width vs Height", fontsize=9)
        ax_scatter.tick_params(labelsize=7)
        if len(az.class_names) <= 20:
            ax_scatter.legend(fontsize=7, markerscale=2.5,
                              loc="upper right", framealpha=0.7)

        # ── histogram: right half of top row (stops before cbar column) ────
        ax_hist = fig.add_subplot(gs[0, n_left:ncols+1])
        recs = az.images if not ann_split_filter else \
               [r for r in az.images if r.split == ann_split_filter]
        counts = [r.n_annotations for r in recs]
        if counts:
            ax_hist.hist(counts, bins=range(0, max(counts) + 2), align="left",
                         color="#55a868", edgecolor="white", linewidth=0.4)
        ax_hist.set_xlabel("Annotations per Image", fontsize=8)
        ax_hist.set_ylabel("Images", fontsize=8)
        hist_title = "Annotations distribution"
        if ann_split_filter:
            hist_title += f" ({ann_split_filter})"
        ax_hist.set_title(hist_title, fontsize=9)
        ax_hist.tick_params(labelsize=7)

        # ── heatmaps ───────────────────────────────────────────────────────
        last_im = None
        for idx, ((title, _), hm) in enumerate(zip(hmap_panels, heatmaps)):
            row, col = divmod(idx, ncols)
            ax = fig.add_subplot(gs[1 + row, col])
            last_im = ax.imshow(hm.T, origin="lower", extent=[0, 1, 0, 1],
                                cmap=heatmap_cmap, aspect="equal",
                                interpolation="bilinear", vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("cx", fontsize=7)
            ax.set_ylabel("cy", fontsize=7)
            ax.tick_params(labelsize=6)

        # hide leftover heatmap cells
        for idx in range(n_hmap, hmap_rows * ncols):
            row, col = divmod(idx, ncols)
            fig.add_subplot(gs[1 + row, col]).set_visible(False)

        # colorbar occupies the full height of the heatmap rows in the cbar col
        cbar_ax = fig.add_subplot(gs[1:, ncols])
        fig.colorbar(last_im, cax=cbar_ax, label="Annotation count")

        fig.tight_layout()

    return fig


def fig_size_scatter(az: YOLODatasetAnalyzer, hex_map: Dict[int, str],
                     alpha: float = 0.35, cls_filter: Optional[List[int]] = None,
                     figsize: Tuple[float, float] = None):
    ann = az.annotations
    if cls_filter:
        ann = [a for a in ann if a.cls in cls_filter]
    if not ann:
        return None

    ws = np.array([a.w for a in ann])
    hs = np.array([a.h for a in ann])
    cls_ids = np.array([a.cls for a in ann])
    unique_cls = sorted(set(cls_ids.tolist()))

    with plt.style.context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
        for c in unique_cls:
            mask = cls_ids == c
            ax.scatter(ws[mask], hs[mask], s=6, alpha=alpha,
                       color=hex_map[c], label=az._cls_name(c))
        ax.set_xlabel("Width (norm.)"); ax.set_ylabel("Height (norm.)")
        ax.set_title("BBox Width vs Height")
        if len(unique_cls) <= 20:
            ax.legend(fontsize=8, markerscale=3)
        fig.tight_layout()
    return fig


def fig_aspect_ratio(az: YOLODatasetAnalyzer, bins: int = 40,
                     cls_filter: Optional[List[int]] = None,
                     figsize: Tuple[float, float] = None):
    ann = az.annotations
    if cls_filter:
        ann = [a for a in ann if a.cls in cls_filter]
    ars = np.array([a.aspect_ratio for a in ann])
    ars = ars[~np.isnan(ars) & ~np.isinf(ars)]
    if not len(ars):
        return None

    with plt.style.context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.hist(ars, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Aspect ratio (w/h)"); ax.set_ylabel("Count")
        ax.set_title("Aspect-Ratio Distribution")
        fig.tight_layout()
    return fig


def fig_annotations_per_image(az: YOLODatasetAnalyzer,
                               split_filter: Optional[str] = None,
                               figsize: Tuple[float, float] = None):
    recs = az.images
    if split_filter:
        recs = [r for r in recs if r.split == split_filter]
    counts = [r.n_annotations for r in recs]
    if not counts:
        return None

    with plt.style.context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.hist(counts, bins=range(0, max(counts) + 2), align="left",
                color="#55a868", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Annotations per image"); ax.set_ylabel("Images")
        title = "Annotations per Image"
        if split_filter:
            title += f" ({split_filter})"
        ax.set_title(title)
        fig.tight_layout()
    return fig


def fig_cooccurrence(az: YOLODatasetAnalyzer, figsize: Tuple[float, float] = None):
    n_cls = len(az.class_names)
    if n_cls < 2:
        return None
    cls_ids_sorted = sorted(az.class_names.keys())
    idx_map = {c: i for i, c in enumerate(cls_ids_sorted)}
    img_classes: Dict[str, set] = defaultdict(set)
    for a in az.annotations:
        img_classes[a.image].add(a.cls)
    matrix = np.zeros((n_cls, n_cls), dtype=int)
    for cls_set in img_classes.values():
        cl = sorted(cls_set)
        for i, ci in enumerate(cl):
            for cj in cl[i:]:
                ii, jj = idx_map[ci], idx_map[cj]
                matrix[ii, jj] += 1
                if ii != jj:
                    matrix[jj, ii] += 1
    diag = np.diag(matrix).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(diag[:, None] > 0, matrix / diag[:, None], 0.0)
    names = [az._cls_name(c) for c in cls_ids_sorted]

    with plt.style.context(MPL_STYLE):
        fig_w = max(6, n_cls * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=figsize or (fig_w, fig_w * 0.9))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="Fraction")
        ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Class Co-occurrence")
        fig.tight_layout()
    return fig


def fig_split_comparison(az: YOLODatasetAnalyzer, figsize: Tuple[float, float] = None):
    splits_present = [s for s in az.splits if any(r.split == s for r in az.images)]
    if len(splits_present) < 2:
        return None
    cls_ids_sorted = sorted(az.class_names.keys())
    n_cls = len(cls_ids_sorted)
    idx_map = {c: i for i, c in enumerate(cls_ids_sorted)}
    split_counts = {}
    for split in splits_present:
        arr = np.zeros(n_cls, dtype=int)
        for a in az.annotations:
            if a.split == split and a.cls in idx_map:
                arr[idx_map[a.cls]] += 1
        split_counts[split] = arr
    names = [az._cls_name(c) for c in cls_ids_sorted]
    x = np.arange(n_cls)
    bar_w = 0.8 / len(splits_present)
    # split bars use a dedicated palette — not class colours
    colors = _split_colors_mpl(len(splits_present))

    with plt.style.context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize or (max(10, n_cls * 0.5 + 3), 5))
        for i, (split, arr) in enumerate(split_counts.items()):
            offsets = x + i * bar_w - (len(splits_present) - 1) * bar_w / 2
            ax.bar(offsets, arr, width=bar_w * 0.9, label=split, color=colors[i], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Annotations"); ax.set_title("Per-Class Count by Split")
        ax.legend()
        fig.tight_layout()
    return fig


def fig_bbox_area_hist(az: YOLODatasetAnalyzer, bins: int = 40,
                        cls_filter: Optional[List[int]] = None,
                        figsize: Tuple[float, float] = None):
    ann = az.annotations
    if cls_filter:
        ann = [a for a in ann if a.cls in cls_filter]
    if not ann:
        return None
    areas = np.array([a.area for a in ann])
    with plt.style.context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=figsize or (8, 4))
        ax.hist(areas, bins=bins, color="#c44e52", edgecolor="white", linewidth=0.4)
        ax.set_xlabel("BBox area (normalised²)"); ax.set_ylabel("Count")
        ax.set_title("BBox Area Distribution")
        fig.tight_layout()
    return fig


# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📦 YOLO Explorer")
    st.markdown("---")

    yaml_input = st.text_input(
        "Dataset YAML path",
        placeholder="/path/to/dataset.yaml",
        help="Path to an Ultralytics YOLO-format dataset YAML file.",
    )

    az: Optional[YOLODatasetAnalyzer] = None
    if yaml_input and Path(yaml_input).is_file():
        try:
            az = load_analyzer(yaml_input)
            st.success(
                f"✅ {len(az.images)} images · "
                f"{len(az.annotations)} annotations · "
                f"{len(az.class_names)} classes"
            )
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
    elif yaml_input:
        st.warning("File not found.")

    st.markdown("---")
    st.caption("YOLO Dataset Explorer · built with Streamlit")


# ── guard: no dataset loaded ───────────────────────────────────────────────
if az is None:
    st.markdown(
        """
        ## 👈 Enter a dataset YAML path to get started

        This app analyses any Ultralytics YOLO-format dataset.

        **Expected YAML structure:**
        ```yaml
        path: /abs/or/relative/dataset/root
        train: images/train
        val:   images/val
        test:  images/test   # optional

        names:
          0: class_a
          1: class_b
        ```
        Labels must live in a `labels/` folder that mirrors the `images/` tree.
        """
    )
    st.stop()

# ── single palette for the whole session ──────────────────────────────────
# Built once from the loaded class_names; keyed by class_id so every plot
# and the explorer always use the exact same colour for each class.
hex_map, rgb_map = build_class_palette(az.class_names)

# ── tabs ───────────────────────────────────────────────────────────────────
tab_stats, tab_explore = st.tabs(["📊 Statistics", "🔍 Explorer"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
with tab_stats:

    # ── overview metrics ──────────────────────────────────────────────────
    st.subheader("Dataset Overview")
    n_splits = sum(1 for s in az.splits if any(r.split == s for r in az.images))
    n_empty  = sum(1 for r in az.images if r.n_annotations == 0)
    mean_ann = len(az.annotations) / max(len(az.images), 1)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Classes",     len(az.class_names))
    col2.metric("Total Images", len(az.images))
    col3.metric("Annotations", len(az.annotations))
    col4.metric("Empty Images", n_empty)
    col5.metric("Ann / Image",  f"{mean_ann:.1f}")

    # per-split breakdown
    with st.expander("Per-split breakdown", expanded=False):
        cols = st.columns(len([s for s in az.splits
                                if any(r.split == s for r in az.images)]) or 1)
        ci = 0
        for split in az.splits:
            imgs   = [r for r in az.images      if r.split == split]
            anns   = [a for a in az.annotations if a.split == split]
            empty  = sum(1 for r in imgs if r.n_annotations == 0)
            if not imgs:
                continue
            with cols[ci]:
                st.markdown(f"**{split}**")
                st.metric("Images",        len(imgs))
                st.metric("Annotations",   len(anns))
                st.metric("Empty",         empty)
                st.metric("Ann / img",     f"{len(anns)/len(imgs):.1f}")
            ci += 1

    st.markdown("---")

    # ── plot controls ──────────────────────────────────────────────────────
    with st.expander("⚙️ Plot settings", expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        bar_orient = pc1.radio("Bar orientation", ["horizontal", "vertical"], horizontal=True)
        hmap_bins  = pc2.slider("Heatmap bins",   16, 128, 64, 8)
        hmap_cmap  = pc3.selectbox("Heatmap colourmap",
                                   ["hot", "viridis", "plasma", "magma", "inferno", "Blues"])
        hist_bins  = pc1.slider("Histogram bins", 10, 100, 40, 5)
        scatter_a  = pc2.slider("Scatter opacity", 0.05, 1.0, 0.35, 0.05)
        dl_dpi     = pc3.number_input("Download DPI (PNG)", min_value=72, max_value=600,
                                      value=300, step=50)

        cls_filter_stats = st.multiselect(
            "Filter classes (size/AR/heatmap/area plots)",
            options=sorted(az.class_names.keys()),
            format_func=az._cls_name,
            default=[],
        )
        cls_filter_stats = cls_filter_stats or None

    # ── helper: compact width/height inputs ────────────────────────────────
    def _size_inputs(key: str, default_w: float, default_h: float
                     ) -> Tuple[float, float]:
        """Render two small number inputs and return (w, h)."""
        c1, c2, _ = st.columns([1, 1, 4])
        w = c1.number_input("W (in)", value=default_w, min_value=1.0,
                            max_value=40.0, step=0.5, key=f"{key}_w")
        h = c2.number_input("H (in)", value=default_h, min_value=1.0,
                            max_value=40.0, step=0.5, key=f"{key}_h")
        return float(w), float(h)

    # ── plots ──────────────────────────────────────────────────────────────
    st.subheader("Class Distribution")
    fs = _size_inputs("cls_dist", 10.0, max(4.0, len(az.class_names) * 0.35))
    f = fig_class_distribution(az, hex_map, bar_orient, figsize=fs)
    st.pyplot(f, use_container_width=False)
    _plot_download_buttons(f, "class_distribution", dpi=dl_dpi)

    st.markdown("---")
    st.subheader("BBox Width vs Height  ·  Annotations per Image  ·  Spatial Heatmaps")

    comb_c1, comb_c2, comb_c3 = st.columns(3)
    hmap_ncols = comb_c1.slider("Heatmap columns", 1, 6, 3, key="hmap_ncols")
    split_sel = comb_c2.selectbox(
        "Annotations split",
        ["all"] + [s for s in az.splits if any(r.split == s for r in az.images)],
        key="ann_per_img_split",
    )
    n_cls      = len(az.class_names)
    n_hmap_rows = ((n_cls + 1) + hmap_ncols - 1) // hmap_ncols
    ncols_eff   = max(hmap_ncols, 2)
    default_cw  = round(3.2 * ncols_eff, 1)
    default_ch  = round(3.2 * (1 + n_hmap_rows), 1)
    fs = _size_inputs("combined", default_cw, default_ch)

    f = fig_combined_main(
        az, hex_map,
        scatter_alpha=scatter_a,
        ann_split_filter=None if split_sel == "all" else split_sel,
        heatmap_bins=hmap_bins,
        heatmap_cmap=hmap_cmap,
        heatmap_ncols=hmap_ncols,
        figsize=fs,
    )
    if f:
        st.pyplot(f, use_container_width=False)
        _plot_download_buttons(f, "combined_main", dpi=dl_dpi)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Aspect-Ratio Distribution")
        fs = _size_inputs("ar", 8.0, 4.0)
        f = fig_aspect_ratio(az, bins=hist_bins,
                             cls_filter=cls_filter_stats, figsize=fs)
        if f:
            st.pyplot(f, use_container_width=False)
            _plot_download_buttons(f, "aspect_ratio", dpi=dl_dpi)

    with col_b:
        st.subheader("BBox Area Distribution")
        fs = _size_inputs("area", 8.0, 4.0)
        f = fig_bbox_area_hist(az, bins=hist_bins,
                               cls_filter=cls_filter_stats, figsize=fs)
        if f:
            st.pyplot(f, use_container_width=False)
            _plot_download_buttons(f, "bbox_area", dpi=dl_dpi)

    if len(az.class_names) >= 2:
        st.markdown("---")
        st.subheader("Class Co-occurrence Matrix")
        auto_w = max(6.0, len(az.class_names) * 0.5 + 1.5)
        fs = _size_inputs("cooc", auto_w, round(auto_w * 0.9, 1))
        f = fig_cooccurrence(az, figsize=fs)
        if f:
            st.pyplot(f, use_container_width=False)
            _plot_download_buttons(f, "cooccurrence", dpi=dl_dpi)

    n_splits_present = sum(
        1 for s in az.splits if any(r.split == s for r in az.images)
    )
    if n_splits_present >= 2:
        st.markdown("---")
        st.subheader("Per-Class Count by Split")
        auto_w = max(10.0, len(az.class_names) * 0.5 + 3)
        fs = _size_inputs("split_cmp", auto_w, 5.0)
        f = fig_split_comparison(az, figsize=fs)
        if f:
            st.pyplot(f, use_container_width=False)
            _plot_download_buttons(f, "split_comparison", dpi=dl_dpi)

    # ── per-class bbox stats table ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("BBox Statistics per Class")
    rows = []
    for cls_id in sorted(az.class_names.keys()):
        cls_ann = [a for a in az.annotations if a.cls == cls_id]
        if not cls_ann:
            continue
        ws    = np.array([a.w    for a in cls_ann])
        hs    = np.array([a.h    for a in cls_ann])
        areas = ws * hs
        ars   = np.where(hs > 0, ws / hs, np.nan)
        rows.append({
            "Class":      az._cls_name(cls_id),
            "Count":      len(cls_ann),
            "W mean":     round(float(ws.mean()),    4),
            "W std":      round(float(ws.std()),     4),
            "H mean":     round(float(hs.mean()),    4),
            "H std":      round(float(hs.std()),     4),
            "Area mean":  round(float(areas.mean()), 5),
            "AR mean":    round(float(np.nanmean(ars)), 3),
        })
    if rows:
        try:
            import pandas as pd
            df_stats = pd.DataFrame(rows)
            st.dataframe(df_stats, use_container_width=True)
            st.download_button(
                "⬇ Download table (CSV)",
                data=df_stats.to_csv(index=False).encode(),
                file_name="bbox_stats_per_class.csv",
                mime="text/csv",
                key="dl_bbox_stats_csv",
            )
        except ImportError:
            for r in rows:
                st.text(r)

    st.markdown("---")

    # ── save to disk ───────────────────────────────────────────────────────
    with st.expander("💾 Save full analysis to disk"):
        out_dir_input = st.text_input(
            "Output directory",
            value=str(Path(yaml_input).parent / "dataset_stats"),
        )
        save_json = st.checkbox("Save stats.json", value=True)
        save_csv  = st.checkbox("Save annotations.csv", value=True)
        if st.button("Run & Save"):
            with st.spinner("Running analysis …"):
                try:
                    analyze_yolo_dataset(
                        yaml_path=yaml_input,
                        output_dir=out_dir_input,
                        save_json=save_json,
                        save_csv=save_csv,
                        verbose=False,
                    )
                    st.success(f"Saved to `{out_dir_input}`")
                except Exception as e:
                    st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tab_explore:

    # ── filter sidebar inside tab ──────────────────────────────────────────
    with st.expander("🔧 Filters & display options", expanded=True):
        fx1, fx2, fx3 = st.columns(3)

        available_splits = sorted({r.split for r in az.images})
        split_sel_exp = fx1.multiselect(
            "Split", available_splits, default=available_splits,
        )

        cls_sel_exp = fx2.multiselect(
            "Show only classes",
            options=sorted(az.class_names.keys()),
            format_func=az._cls_name,
            default=[],
            help="Leave empty to show all classes.",
        )

        ann_min, ann_max = 0, max((r.n_annotations for r in az.images), default=0)
        ann_range = fx3.slider(
            "Annotations per image",
            ann_min, max(ann_max, 1), (ann_min, ann_max),
        )

        fx4, fx5, fx6 = st.columns(3)
        sort_by  = fx4.selectbox("Sort by", ["name", "annotations ↑", "annotations ↓", "random"])
        show_lbl = fx5.checkbox("Show labels", value=True)
        lbl_bg   = fx5.checkbox("Label background", value=True)
        line_w   = fx6.slider("Box line width", 1, 6, 2)
        thumb_size = fx6.slider("Max image size (px)", 256, 2048, 800, 128)

        view_mode = fx4.radio("View mode", ["Single image", "Grid"], horizontal=True)
        grid_cols  = fx5.slider("Grid columns", 1, 6, 3) if view_mode == "Grid" else 3
        page_size  = fx6.number_input("Images per page", min_value=1, max_value=200, value=16, step=4) if view_mode == "Grid" else 1

    # ── build filtered + sorted image list ────────────────────────────────
    # index annotations by image name for fast lookup
    ann_by_image: Dict[str, List] = defaultdict(list)
    for a in az.annotations:
        ann_by_image[a.image].append((a.cls, a.cx, a.cy, a.w, a.h))

    filtered = [
        r for r in az.images
        if r.split in split_sel_exp
        and ann_range[0] <= r.n_annotations <= ann_range[1]
    ]

    # if class filter active, keep images that have ≥1 annotation in filter
    if cls_sel_exp:
        cls_set = set(cls_sel_exp)
        filtered = [
            r for r in filtered
            if any(c in cls_set for (c, *_) in ann_by_image[r.name])
        ]

    if sort_by == "name":
        filtered = sorted(filtered, key=lambda r: r.name)
    elif sort_by == "annotations ↑":
        filtered = sorted(filtered, key=lambda r: r.n_annotations)
    elif sort_by == "annotations ↓":
        filtered = sorted(filtered, key=lambda r: r.n_annotations, reverse=True)
    elif sort_by == "random":
        import random
        random.seed(42)
        random.shuffle(filtered)

    st.markdown(f"**{len(filtered)}** images match filters")

    if not filtered:
        st.info("No images match the current filters.")
        st.stop()

    # ── annotation filter per image (class mask) ───────────────────────────
    def get_ann_for_image(rec) -> List:
        anns = ann_by_image[rec.name]
        if cls_sel_exp:
            anns = [(c, cx, cy, w, h) for (c, cx, cy, w, h) in anns if c in set(cls_sel_exp)]
        return anns

    # ── SINGLE IMAGE VIEW ──────────────────────────────────────────────────
    if view_mode == "Single image":
        # session state for navigation index
        if "explorer_idx" not in st.session_state:
            st.session_state.explorer_idx = 0
        st.session_state.explorer_idx = min(
            st.session_state.explorer_idx, len(filtered) - 1
        )

        nav1, nav2, nav3, nav4 = st.columns([1, 1, 6, 2])
        if nav1.button("◀ Prev"):
            st.session_state.explorer_idx = max(0, st.session_state.explorer_idx - 1)
        if nav2.button("Next ▶"):
            st.session_state.explorer_idx = min(
                len(filtered) - 1, st.session_state.explorer_idx + 1
            )
        idx = nav4.number_input(
            "Jump to #", 0, len(filtered) - 1,
            value=st.session_state.explorer_idx, step=1, key="jump_box",
        )
        st.session_state.explorer_idx = int(idx)

        idx_slider = st.slider(
            "Image index", 0, len(filtered) - 1,
            value=st.session_state.explorer_idx, key="idx_slider",
        )
        st.session_state.explorer_idx = idx_slider

        rec = filtered[st.session_state.explorer_idx]
        anns = get_ann_for_image(rec)

        # metadata row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Split",       rec.split)
        m2.metric("Annotations", len(anns))
        m3.metric("Image",       f"{st.session_state.explorer_idx + 1}/{len(filtered)}")
        m4.markdown(f"**File:** `{rec.name}`")

        if rec.path.is_file():
            img = render_image_with_boxes(
                str(rec.path), anns, az.class_names, rgb_map,
                line_width=line_w, show_labels=show_lbl,
                label_bg=lbl_bg, max_size=thumb_size,
            )
            img_bytes = pil_to_st_image(img)
            st.image(img_bytes, use_container_width=True)
            st.download_button(
                "⬇ Download image (PNG)",
                data=img_bytes,
                file_name=rec.name,
                mime="image/png",
                key="dl_explorer_img",
            )

            # annotation detail table
            if anns:
                with st.expander("Annotations detail"):
                    try:
                        import pandas as pd
                        df = pd.DataFrame(
                            [{"class": az._cls_name(c), "cx": cx, "cy": cy,
                              "w": w, "h": h, "area": round(w*h, 5)}
                             for (c, cx, cy, w, h) in anns]
                        )
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "⬇ Download annotations (CSV)",
                            data=df.to_csv(index=False).encode(),
                            file_name=f"{rec.path.stem}_annotations.csv",
                            mime="text/csv",
                            key="dl_explorer_ann_csv",
                        )
                    except ImportError:
                        for a in anns:
                            st.text(a)
        else:
            st.warning(f"Image file not found: `{rec.path}`")

    # ── GRID VIEW ──────────────────────────────────────────────────────────
    else:
        n_pages = max(1, (len(filtered) + page_size - 1) // page_size)

        page = st.number_input("Page", 1, n_pages, 1, key="grid_page") - 1
        start = page * page_size
        page_recs = filtered[start: start + page_size]

        for row_start in range(0, len(page_recs), grid_cols):
            row_recs = page_recs[row_start: row_start + grid_cols]
            cols = st.columns(grid_cols)
            for col, rec in zip(cols, row_recs):
                anns = get_ann_for_image(rec)
                with col:
                    if rec.path.is_file():
                        img = render_image_with_boxes(
                            str(rec.path), anns, az.class_names, rgb_map,
                            line_width=line_w, show_labels=show_lbl,
                            label_bg=lbl_bg, max_size=512,
                        )
                        st.image(pil_to_st_image(img), use_container_width=True)
                    else:
                        st.warning("missing")
                    st.caption(
                        f"`{rec.name}` · {rec.split} · {len(anns)} ann"
                    )

        st.markdown(f"Page {page + 1} / {n_pages}  "
                    f"(images {start + 1}–{min(start + page_size, len(filtered))} "
                    f"of {len(filtered)})")
