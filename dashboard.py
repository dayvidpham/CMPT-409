import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path
import io

# Import your local modules
from utils.read_results import ResultsReader
from engine.strategies import AxisScale, PlotStrategy, PlotContext
from engine.colors import ColorManagerFactory

st.set_page_config(layout="wide", page_title="Optimizer Results Dashboard")

# --- Helper: Data Loading ---
@st.cache_resource
def load_data(filepath):
    """Load and cache the results reader."""
    return ResultsReader(filepath)

def get_available_npz_files():
    """Recursively find all results.npz files in experiments/prayers/."""
    base_path = Path("experiments/prayers/")
    if not base_path.exists():
        st.error(f"Directory not found: {base_path}. Please run from project root.")
        return []
    return list(base_path.rglob("results.npz"))

# --- Helper: Exact Legend Replica from plotting.py ---
def add_custom_split_legend(ax, color_manager, lrs, rhos, show_styles=False):
    """
    Replicates the exact legend logic from plotting.py.
    Uses Patches for headers and Line2D for entries.
    """
    legend_elements = []

    # 1. Split Style Indicators (Optional)
    if show_styles:
        legend_elements.append(Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="SAM (Solid)"))
        legend_elements.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, alpha=0.5, label="Base (Dashed)"))
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="   ")) # Spacer

    # 2. Learning Rate (Hue)
    if lrs:
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="Learning Rate (Hue):"))
        for lr in lrs:
            # Use bright color (rho=0 equivalent or max vibrancy) for legend
            c = color_manager.color_lr(lr) 
            legend_elements.append(Line2D([0], [0], color=c, linewidth=3, label=f"  lr={lr}"))

    # 3. Rho (Vibrancy)
    # Only show if we have rhos > 0 or multiple rhos
    valid_rhos = sorted(list(set(rhos)))
    if len(valid_rhos) > 1 or (len(valid_rhos) == 1 and valid_rhos[0] > 0):
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="   ")) # Spacer
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="rho (Vibrancy, SAM only):"))
        
        # Use a sample LR to show rho effect
        sample_lr = lrs[0] if lrs else 0.1
        
        for rho in valid_rhos:
            # Compute color config
            c = color_manager.color_config(sample_lr, rho)
            legend_elements.append(Line2D([0], [0], color=c, linewidth=3, alpha=0.8, label=f"  rho={rho}"))

    # Configure the legend above the plot
    # Dynamic ncol calculation to keep it readable
    total_items = len(legend_elements)
    ncols = min(total_items, 8) 
    
    # Place legend above the plot (y=1.02)
    ax.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncols, 
        frameon=True,
        fontsize='small',
        handletextpad=0.2,
        columnspacing=1.0,
        edgecolor='lightgray'
    )

# --- Sidebar: Configuration ---
st.sidebar.title("Experiment Config")

# File Selection
available_files = get_available_npz_files()
if not available_files:
    st.error("No 'results.npz' files found in experiments/prayers/.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select Results File", 
    available_files, 
    format_func=lambda x: str(x)
)

try:
    reader = load_data(str(selected_file))
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Global Data Parsing
all_optimizers = reader.optimizers
all_metrics = reader.metrics
all_params = reader.hyperparams

# Detect experiment type from file path
experiment_path = selected_file.parent.name  # e.g., "2025-12-15_12-41-06"
experiment_type = selected_file.parent.parent.name  # e.g., "soudry_gd" or "soudry_sgd"

# Determine if this is an SGD experiment (stochastic gradient descent)
is_sgd_experiment = "_sgd" in experiment_type.lower()

# Helper function to map optimizer names for SGD experiments
def map_optimizer_name(opt_name):
    """Map optimizer names for SGD experiments."""
    if not is_sgd_experiment:
        return opt_name

    # Apply SGD mappings
    mappings = {
        "GD": "SGD",
        "VecNGD": "VecNSGD",
        "LossNGD": "LossNSGD",
        "SAM": "SAM_SGD",  # Base SAM uses GD
        "SAM_VecNGD": "SAM_VecNSGD",
        "SAM_LossNGD": "SAM_LossNSGD",
    }
    return mappings.get(opt_name, opt_name)

# --- Sidebar: Filtering ---
st.sidebar.header("Global Filters")
selected_opts = st.sidebar.multiselect("Select Optimizers", all_optimizers, default=all_optimizers)

# Extract LRs and Rhos for Color Management
all_lrs = set()
all_rhos = set()
for opt in selected_opts:
    for p in all_params.get(opt, []):
        if 'lr' in p: all_lrs.add(p['lr'])
        if 'rho' in p: all_rhos.add(p['rho'])

sorted_lrs = sorted(list(all_lrs))
sorted_rhos = sorted(list(all_rhos))

# --- Main Interface ---
st.title("Experiment Findings Explorer")

# ==============================================================================
# GLOBAL CONTROLS (Apply to all findings)
# ==============================================================================
st.subheader("Global Settings")

col1, col2 = st.columns(2)
with col1:
    # Experiment type toggle (global)
    global_experiment_choice = st.radio(
        "Experiment Type",
        ["GD (Gradient Descent)", "SGD (Stochastic Gradient Descent)"],
        horizontal=True,
        key="global_experiment_choice"
    )

with col2:
    global_show_legend = st.checkbox("Show Legend", value=True, key="global_legend")

# Set global paths and parameters based on selection
if "GD" in global_experiment_choice and "SGD" not in global_experiment_choice:
    global_data_path = Path("experiments/prayers/soudry_gd/2025-12-15_12-41-06/results.npz")
    is_sgd_global = False
else:
    global_data_path = Path("experiments/prayers/soudry_sgd/2025-12-15_13-04-25/results.npz")
    is_sgd_global = True

if not global_data_path.exists():
    st.error(f"Data not found: {global_data_path}")
    st.stop()

# Load the global data
try:
    reader_global = load_data(str(global_data_path))
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Set global labels and optimizer names
global_x_label = "Epoch" if is_sgd_global else "Iteration"
global_base_opt_name = "SGD" if is_sgd_global else "GD"
global_sam_opt_name = "SAM_SGD" if is_sgd_global else "SAM_GD"

# Get data from global reader
all_optimizers_global = reader_global.optimizers
all_metrics_global = reader_global.metrics
all_params_global = reader_global.hyperparams

# Global optimizer selectors
col3, col4 = st.columns(2)
with col3:
    base_opts_global = [o for o in all_optimizers_global if "SAM" not in o]
    global_base_opt = st.selectbox(
        "Base Optimizer",
        base_opts_global,
        index=base_opts_global.index("GD") if "GD" in base_opts_global else 0,
        key="global_base_opt"
    )
with col4:
    sam_opts_global = [o for o in all_optimizers_global if "SAM" in o]
    global_sam_opt = st.selectbox(
        "SAM Variant",
        sam_opts_global,
        index=sam_opts_global.index("SAM") if "SAM" in sam_opts_global else 0,
        key="global_sam_opt"
    )

# Global title
global_default_title = f"Angle Difference from the Max-Margin Solution, for {global_base_opt_name} vs {global_sam_opt_name} for a Linear Model"
global_title = st.text_input("Plot Title (applies to all findings)", global_default_title, key="global_title")

st.markdown("---")

tabs = st.tabs(["Trajectory Analysis (Finding 1)", "SAM vs Base (Finding 2)", "Hyperparam Grid (Finding 3)", "Stability Analysis"])

# ==============================================================================
# TAB 1: General Trajectory Analysis
# ==============================================================================
with tabs[0]:
    st.header("Trajectory Analysis")

    # Local controls specific to Finding 1
    col1, col2 = st.columns(2)
    with col1:
        plot_metric = st.selectbox("Metric (Y-Axis)", all_metrics_global, index=all_metrics_global.index("angle") if "angle" in all_metrics_global else 0, key="f1_metric")
    with col2:
        x_axis_type = st.radio("X-Axis Scale", ["Log", "Linear"], horizontal=True, key="f1_x_scale")

    strategy = PlotStrategy(
        x_scale=AxisScale.Log if x_axis_type == "Log" else AxisScale.Linear,
        y_scale=AxisScale.Log if "loss" in plot_metric or "distance" in plot_metric or "angle" in plot_metric else AxisScale.Linear
    )

    # Helper function from plotting.py to compute colors with rho vibrancy
    def compute_rho_vibrancy_color(lr, rho, all_lrs, all_rhos):
        """Match the exact color computation from engine/plotting.py"""
        import colorsys
        try:
            import hsluv
        except ImportError:
            hsluv = None

        # Map LR to hue position
        sorted_lrs_local = sorted(all_lrs)
        n_lrs = len(sorted_lrs_local)
        if lr in sorted_lrs_local:
            lr_rank = sorted_lrs_local.index(lr)
            lr_normalized = lr_rank / max(1, n_lrs - 1)
        else:
            lr_normalized = 0.5

        base_hue = (lr_normalized * 330.0) % 360.0
        hue = base_hue

        if rho == 0.0:
            saturation = 100.0
            lightness = 40.0
        else:
            sorted_rhos_local = sorted([r for r in all_rhos if r > 0.0])
            n_rhos = len(sorted_rhos_local)
            if n_rhos > 0 and rho in sorted_rhos_local:
                rho_rank = sorted_rhos_local.index(rho)
                rho_normalized = rho_rank / max(1, n_rhos - 1)
            else:
                rho_normalized = 0.5

            hue_shift = (rho_normalized - 0.5) * 30.0
            hue = (base_hue + hue_shift) % 360.0
            saturation = 15.0 + 70.0 * rho_normalized
            lightness = 85.0 - 45.0 * rho_normalized

        if hsluv is None:
            rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, lightness / 100.0)
        else:
            rgb = hsluv.hsluv_to_rgb((hue, saturation, lightness))

        return (rgb[0], rgb[1], rgb[2])

    # Extract LRs and rhos from global data for the selected optimizers
    selected_opts_f1 = [global_base_opt, global_sam_opt]
    all_lrs_f1 = set()
    all_rhos_f1 = set()
    for opt in selected_opts_f1:
        for p in all_params_global.get(opt, []):
            if 'lr' in p: all_lrs_f1.add(p['lr'])
            if 'rho' in p: all_rhos_f1.add(p['rho'])

    sorted_lrs_f1 = sorted(list(all_lrs_f1))
    sorted_rhos_f1 = sorted(list(all_rhos_f1))

    # Create two-column layout: Base and SAM
    fig, (ax_gd, ax_sam) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Configure both axes
    strategy.configure_axis(ax_gd, base_label=plot_metric)
    strategy.configure_axis(ax_sam, base_label=plot_metric)

    # Set titles with optimizer names (apply mapping for display)
    display_base_name = global_base_opt_name if global_base_opt == "GD" else global_base_opt
    display_sam_name = global_sam_opt_name if global_sam_opt == "SAM" else global_sam_opt
    ax_gd.set_title(f"{display_base_name} (Base)", fontsize=12, pad=10)
    ax_sam.set_title(display_sam_name, fontsize=12, pad=10)

    # Set x-axis labels
    ax_gd.set_xlabel(global_x_label, fontsize=14)
    ax_sam.set_xlabel(global_x_label, fontsize=14)

    # Set y-axis labels on both subplots
    ax_gd.set_ylabel("Angle Difference (radians)", fontsize=14)
    ax_sam.set_ylabel("Angle Difference (radians)", fontsize=14)

    # Set tick label font sizes
    ax_gd.tick_params(axis='both', which='major', labelsize=14)
    ax_sam.tick_params(axis='both', which='major', labelsize=14)

    count_gd = 0
    count_sam = 0

    # Plot base optimizer
    opt_params = all_params_global.get(global_base_opt, [])
    for params in opt_params:
        lr = params.get('lr')
        rho = 0.0  # Base optimizers have rho=0

        run_data = []
        steps = None
        for seed in reader_global.seeds:
            try:
                data = reader_global.get_data(global_base_opt, params, seed, plot_metric)
                steps_data = reader_global.get_data(global_base_opt, params, seed, 'steps')
                run_data.append(data)
                steps = steps_data
            except KeyError: pass

        if run_data and steps is not None:
            mean_data = np.mean(np.stack(run_data), axis=0)
            c = compute_rho_vibrancy_color(lr, rho, sorted_lrs_f1, sorted_rhos_f1)

            ctx = PlotContext(
                ax=ax_gd, x=steps, y=mean_data,
                label="_nolegend_",
                plot_kwargs={"color": c, "linewidth": 2.0, "alpha": 0.8}
            )
            strategy.plot(ctx)
            count_gd += 1

    # Plot SAM optimizer
    opt_params = all_params_global.get(global_sam_opt, [])
    for params in opt_params:
        lr = params.get('lr')
        rho = params.get('rho', 0.0)

        run_data = []
        steps = None
        for seed in reader_global.seeds:
            try:
                data = reader_global.get_data(global_sam_opt, params, seed, plot_metric)
                steps_data = reader_global.get_data(global_sam_opt, params, seed, 'steps')
                run_data.append(data)
                steps = steps_data
            except KeyError: pass

        if run_data and steps is not None:
            mean_data = np.mean(np.stack(run_data), axis=0)
            c = compute_rho_vibrancy_color(lr, rho, sorted_lrs_f1, sorted_rhos_f1)

            ctx = PlotContext(
                ax=ax_sam, x=steps, y=mean_data,
                label="_nolegend_",
                plot_kwargs={"color": c, "linewidth": 2.0, "alpha": 0.8}
            )
            strategy.plot(ctx)
            count_sam += 1

    if count_gd > 0 or count_sam > 0:
        # Add custom legend if enabled (matching sam_comparison style from plotting.py)
        if global_show_legend:
            legend_elements = []

            # Learning Rate (Hue)
            if sorted_lrs_f1:
                legend_elements.append(Patch(facecolor="none", edgecolor="none", label="Learning Rate (Hue):"))
                # Use mid-rho for LR color examples
                mid_rho_idx = len(sorted_rhos_f1) // 2 if sorted_rhos_f1 else 0
                sample_rho = sorted_rhos_f1[mid_rho_idx] if sorted_rhos_f1 else 0.0
                for lr in sorted_lrs_f1:
                    c_rgb = compute_rho_vibrancy_color(lr, sample_rho, sorted_lrs_f1, sorted_rhos_f1)
                    legend_elements.append(Line2D([0], [0], color=c_rgb, linewidth=3, label=f"  lr={lr}"))

            # Rho (Vibrancy)
            if sorted_rhos_f1 and len(sorted_rhos_f1) > 1:
                legend_elements.append(Patch(facecolor="none", edgecolor="none", label=""))
                rho_label = f"rho (Vibrancy, {global_sam_opt_name} only):"
                legend_elements.append(Patch(facecolor="none", edgecolor="none", label=rho_label))
                # Use sample LR for rho color examples
                sample_lr = sorted_lrs_f1[0] if sorted_lrs_f1 else 0.01
                for rho in sorted_rhos_f1:
                    c_rgb = compute_rho_vibrancy_color(sample_lr, rho, sorted_lrs_f1, sorted_rhos_f1)
                    legend_elements.append(Line2D([0], [0], color=c_rgb, linewidth=3, alpha=0.8, label=f"  rho={rho}"))

            if legend_elements:
                # Position legend above the plots using figure legend
                fig.legend(
                    handles=legend_elements,
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=min(len(legend_elements), 8),
                    frameon=True,
                    fontsize=11,
                    handlelength=2.0,
                    handleheight=0.7,
                    labelspacing=0.3,
                    columnspacing=1.5,
                )

        # Add title if provided
        if global_title:
            fig.suptitle(global_title, fontsize=14, y=1.10 if global_show_legend else 0.98)

        st.pyplot(fig)

        fn = f"{plot_metric}_trajectory.pdf"
        img = io.BytesIO()
        fig.savefig(img, format='pdf', bbox_inches='tight')
        st.download_button(label="Download PDF", data=img, file_name=fn, mime="application/pdf")

# ==============================================================================
# TAB 2: SAM vs Base (Finding 2)
# ==============================================================================
with tabs[1]:
    st.header("Finding 2: SAM vs Base Comparison")

    # Local controls specific to Finding 2
    metric_select = st.selectbox("Metric", [m for m in all_metrics_global], index=all_metrics_global.index("angle") if "angle" in all_metrics_global else 0, key="f2_metric")

    # Extract LRs and rhos from global data
    selected_opts_f2 = [global_base_opt, global_sam_opt]
    all_lrs_f2 = set()
    all_rhos_f2 = set()
    for opt in selected_opts_f2:
        for p in all_params_global.get(opt, []):
            if 'lr' in p: all_lrs_f2.add(p['lr'])
            if 'rho' in p: all_rhos_f2.add(p['rho'])

    sorted_lrs_f2 = sorted(list(all_lrs_f2))
    sorted_rhos_f2 = sorted(list(all_rhos_f2))

    # Helper function from Finding 1 to compute colors with rho vibrancy
    def compute_rho_vibrancy_color_f2(lr, rho, all_lrs, all_rhos):
        """Match the exact color computation from engine/plotting.py"""
        import colorsys
        try:
            import hsluv
        except ImportError:
            hsluv = None

        # Map LR to hue position
        sorted_lrs_local = sorted(all_lrs)
        n_lrs = len(sorted_lrs_local)
        if lr in sorted_lrs_local:
            lr_rank = sorted_lrs_local.index(lr)
            lr_normalized = lr_rank / max(1, n_lrs - 1)
        else:
            lr_normalized = 0.5

        base_hue = (lr_normalized * 330.0) % 360.0
        hue = base_hue

        if rho == 0.0:
            saturation = 100.0
            lightness = 40.0
        else:
            sorted_rhos_local = sorted([r for r in all_rhos if r > 0.0])
            n_rhos = len(sorted_rhos_local)
            if n_rhos > 0 and rho in sorted_rhos_local:
                rho_rank = sorted_rhos_local.index(rho)
                rho_normalized = rho_rank / max(1, n_rhos - 1)
            else:
                rho_normalized = 0.5

            hue_shift = (rho_normalized - 0.5) * 30.0
            hue = (base_hue + hue_shift) % 360.0
            saturation = 15.0 + 70.0 * rho_normalized
            lightness = 85.0 - 45.0 * rho_normalized

        if hsluv is None:
            rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, lightness / 100.0)
        else:
            rgb = hsluv.hsluv_to_rgb((hue, saturation, lightness))

        return (rgb[0], rgb[1], rgb[2])

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    strategy2 = PlotStrategy(
        x_scale=AxisScale.Log,
        y_scale=AxisScale.Log if "loss" in metric_select or "angle" in metric_select else AxisScale.Linear
    )
    strategy2.configure_axis(ax2, base_label=metric_select)

    # Set axis labels
    ax2.set_xlabel(global_x_label, fontsize=14)
    ax2.set_ylabel("Angle Difference (radians)", fontsize=14)

    # Set tick label font sizes
    ax2.tick_params(axis='both', which='major', labelsize=14)

    # Plot Base
    for params in all_params_global.get(global_base_opt, []):
        lr = params.get('lr')
        for seed in reader_global.seeds:
            try:
                y = reader_global.get_data(global_base_opt, params, seed, metric_select)
                x = reader_global.get_data(global_base_opt, params, seed, 'steps')
                c = compute_rho_vibrancy_color_f2(lr, 0.0, sorted_lrs_f2, sorted_rhos_f2)
                ax2.plot(x, y, color=c, linestyle='--', alpha=0.8, linewidth=1.5, marker='x', markersize=8, markevery=20, label="_nolegend_")
                break
            except: pass

    # Plot SAM
    for params in all_params_global.get(global_sam_opt, []):
        lr = params.get('lr')
        rho = params.get('rho', 0.0)
        for seed in reader_global.seeds:
            try:
                y = reader_global.get_data(global_sam_opt, params, seed, metric_select)
                x = reader_global.get_data(global_sam_opt, params, seed, 'steps')
                c = compute_rho_vibrancy_color_f2(lr, rho, sorted_lrs_f2, sorted_rhos_f2)
                ax2.plot(x, y, color=c, linestyle='-', alpha=0.9, linewidth=2.0, label="_nolegend_")
                break
            except: pass

    if global_show_legend:
        # Create legend similar to Finding 1
        legend_elements = []

        # Split style indicators
        legend_elements.append(Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="SAM (solid)"))
        legend_elements.append(Line2D([0], [0], color="black", linestyle="--", linewidth=2, alpha=0.5, label="Base (dashed)"))
        legend_elements.append(Patch(facecolor="none", edgecolor="none", label="   "))

        # Learning Rate (Hue)
        if sorted_lrs_f2:
            legend_elements.append(Patch(facecolor="none", edgecolor="none", label="Learning Rate (Hue):"))
            mid_rho_idx = len(sorted_rhos_f2) // 2 if sorted_rhos_f2 else 0
            sample_rho = sorted_rhos_f2[mid_rho_idx] if sorted_rhos_f2 else 0.0
            for lr in sorted_lrs_f2:
                c_rgb = compute_rho_vibrancy_color_f2(lr, sample_rho, sorted_lrs_f2, sorted_rhos_f2)
                legend_elements.append(Line2D([0], [0], color=c_rgb, linewidth=3, label=f"  lr={lr}"))

        # Rho (Vibrancy)
        if sorted_rhos_f2 and len(sorted_rhos_f2) > 1:
            legend_elements.append(Patch(facecolor="none", edgecolor="none", label=""))
            rho_label = f"rho (Vibrancy, {global_sam_opt_name} only):"
            legend_elements.append(Patch(facecolor="none", edgecolor="none", label=rho_label))
            sample_lr = sorted_lrs_f2[0] if sorted_lrs_f2 else 0.01
            for rho in sorted_rhos_f2:
                c_rgb = compute_rho_vibrancy_color_f2(sample_lr, rho, sorted_lrs_f2, sorted_rhos_f2)
                legend_elements.append(Line2D([0], [0], color=c_rgb, linewidth=3, alpha=0.8, label=f"  rho={rho}"))

        if legend_elements:
            ax2.legend(
                handles=legend_elements,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(len(legend_elements), 8),
                frameon=True,
                fontsize=11,
                handletextpad=0.2,
                columnspacing=1.0,
                edgecolor='lightgray'
            )

    # Add title if provided
    if global_title:
        fig2.suptitle(global_title, fontsize=14, y=1.10 if global_show_legend else 0.98)

    st.pyplot(fig2)

    fn2 = f"{global_base_opt}_vs_{global_sam_opt}.pdf"
    img2 = io.BytesIO()
    fig2.savefig(img2, format='pdf', bbox_inches='tight')
    st.download_button("Download Plot PDF", data=img2, file_name=fn2, mime="application/pdf")

# ==============================================================================
# TAB 3: Hyperparam Grid (Finding 3)
# ==============================================================================
with tabs[2]:
    st.header("Finding 3: Hyperparameter Grid")
    
    grid_metric = st.selectbox("Grid Metric", all_metrics, index=all_metrics.index("angle") if "angle" in all_metrics else 0, key="t3_metric")
    grid_lrs = st.multiselect("Grid Learning Rates", sorted_lrs, default=sorted_lrs[:4] if len(sorted_lrs) > 4 else sorted_lrs)
    grid_rhos = st.multiselect("Grid Rhos", sorted_rhos, default=[r for r in sorted_rhos if r > 0])
    
    if grid_lrs and grid_rhos:
        nrows, ncols = len(grid_rhos), len(grid_lrs)
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3*nrows), sharex=True, sharey=True, constrained_layout=True)
        
        if nrows == 1 and ncols == 1: axes3 = np.array([[axes3]])
        elif nrows == 1: axes3 = axes3[np.newaxis, :]
        elif ncols == 1: axes3 = axes3[:, np.newaxis]
        
        color_strat = ColorManagerFactory.create_paired_optimizer_manager(selected_opts, grid_rhos)

        for i, rho in enumerate(grid_rhos):
            for j, lr in enumerate(grid_lrs):
                ax = axes3[i, j]
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True, which='both', alpha=0.2)
                
                # Base Optimizers
                for opt in [o for o in selected_opts if "SAM" not in o]:
                    params = next((p for p in all_params.get(opt, []) if p['lr'] == lr), None)
                    if params:
                        try:
                            y = reader.get_data(opt, params, 0, grid_metric)
                            x = reader.get_data(opt, params, 0, 'steps')
                            c = color_strat.color_config(opt, rho=0)
                            ax.plot(x, y, color=c, alpha=0.5, linestyle='--', linewidth=1.5)
                        except: pass

                # SAM Optimizers
                for opt in [o for o in selected_opts if "SAM" in o]:
                    params = next((p for p in all_params.get(opt, []) if p['lr'] == lr and p.get('rho') == rho), None)
                    if params:
                        try:
                            y = reader.get_data(opt, params, 0, grid_metric)
                            x = reader.get_data(opt, params, 0, 'steps')
                            c = color_strat.color_config(opt, rho=rho)
                            ax.plot(x, y, color=c, alpha=0.9, linestyle='-', linewidth=2.0)
                        except: pass
                
                if i == 0: ax.set_title(f"LR = {lr}", fontsize=10)
                if j == 0: ax.set_ylabel(f"Rho = {rho}\n{grid_metric}", fontsize=9)
        
        # Grid Legend
        legend_elements = []
        for opt in selected_opts:
            c = color_strat.color_config(opt, rho=0.1)
            style = '-' if "SAM" in opt else '--'
            legend_elements.append(Line2D([0], [0], color=c, lw=2, linestyle=style, label=opt))
        fig3.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(selected_opts))

        st.pyplot(fig3)
        img3 = io.BytesIO()
        fig3.savefig(img3, format='pdf', bbox_inches='tight')
        st.download_button("Download Grid PDF", data=img3, file_name="hyperparam_grid.pdf", mime="application/pdf")

# ==============================================================================
# TAB 4: Stability Analysis
# ==============================================================================
with tabs[3]:
    st.header("Stability Analysis")
    stab_metric = st.selectbox("Stability Metric", ["w_norm", "update_norm", "grad_norm"], index=0)
    
    available_stab = [m for m in all_metrics if stab_metric in m]
    if available_stab:
        metric_key = available_stab[0]
        fig4, ax4 = plt.subplots(figsize=(12, 7))
        strategy4 = PlotStrategy(x_scale=AxisScale.Log, y_scale=AxisScale.Linear)
        strategy4.configure_axis(ax4, base_label=metric_key)
        
        colors = ColorManagerFactory.create_sequential_lr_manager(sorted_lrs, sorted_rhos)
        
        for opt in selected_opts:
            for params in all_params.get(opt, []):
                lr = params.get('lr')
                rho = params.get('rho', 0.0)
                try:
                    y = reader.get_data(opt, params, 0, metric_key)
                    x = reader.get_data(opt, params, 0, 'steps')
                    c = colors.color_config(lr, rho)
                    ax4.plot(x, y, color=c, label="_nolegend_")
                except: pass
        
        add_custom_split_legend(ax4, colors, sorted_lrs, sorted_rhos)
        st.pyplot(fig4)
