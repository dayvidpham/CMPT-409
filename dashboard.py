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
all_metrics_global = list(reader_global.metrics)
all_params_global = reader_global.hyperparams

# ==============================================================================
# COMPUTE RATIO METRICS
# ==============================================================================
# Compute LossNGD / VecNGD ratio for angle metric
ratio_data_global = {}

def compute_ratio_metric(reader, opt1, opt2, metric, params, seed):
    """Compute ratio of opt1/opt2 for a given metric, params, and seed."""
    try:
        data1 = reader.get_data(opt1, params, seed, metric)
        data2 = reader.get_data(opt2, params, seed, metric)

        # Ensure both arrays have the same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

        # Compute ratio (avoid division by zero)
        ratio = np.divide(data1, data2, where=data2!=0, out=np.full_like(data1, np.nan))
        return ratio
    except KeyError:
        return None

# Compute ratios for LossNGD / VecNGD
if 'LossNGD' in all_optimizers_global and 'VecNGD' in all_optimizers_global and 'angle' in all_metrics_global:
    for params in all_params_global.get('LossNGD', []):
        # Check if VecNGD has the same params
        if params in all_params_global.get('VecNGD', []):
            for seed in reader_global.seeds:
                ratio = compute_ratio_metric(reader_global, 'LossNGD', 'VecNGD', 'angle', params, seed)
                if ratio is not None:
                    # Store with a key similar to the original data format
                    param_str = ','.join([f'{k}={v}' for k, v in sorted(params.items())])
                    key = f'LossNGD/VecNGD({param_str})_seed{seed}_angle'
                    ratio_data_global[key] = ratio

                    # Also store the steps data (use from LossNGD)
                    try:
                        steps = reader_global.get_data('LossNGD', params, seed, 'steps')
                        steps_key = f'LossNGD/VecNGD({param_str})_seed{seed}_steps'
                        ratio_data_global[steps_key] = steps[:len(ratio)]
                    except:
                        pass

# Add ratio metric to available metrics if we computed any ratios
if ratio_data_global:
    ratio_metric_name = "LossNGD/VecNGD_angle"
    if ratio_metric_name not in all_metrics_global:
        all_metrics_global.append(ratio_metric_name)

# Helper function to get data for both regular and ratio metrics
def get_metric_data_global(optimizer, params, seed, metric, current_metric_name=None):
    """
    Get metric data for either regular metrics or computed ratio metrics.

    For ratio metrics like "LossNGD/VecNGD_angle", retrieves from ratio_data_global.
    For regular metrics, uses reader_global.get_data().

    Args:
        optimizer: Optimizer name (ignored for ratio metrics)
        params: Hyperparameter dict
        seed: Random seed
        metric: Metric name (e.g., "angle", "steps", or "LossNGD/VecNGD_angle")
        current_metric_name: The primary metric being plotted (used to determine if steps should come from ratio data)
    """
    # Check if we're requesting steps for a ratio metric plot
    if metric == 'steps' and current_metric_name and "/" in current_metric_name:
        # Get steps from ratio data
        base_metric = current_metric_name.split("_")[-1]
        ratio_opts = current_metric_name.split("_")[0]
        param_str = ','.join([f'{k}={v}' for k, v in sorted(params.items())])
        steps_key = f'{ratio_opts}({param_str})_seed{seed}_steps'

        if steps_key in ratio_data_global:
            return ratio_data_global[steps_key]
        # Fall back to regular steps if not found
        return reader_global.get_data(optimizer, params, seed, metric)

    if "/" in metric:
        # This is a ratio metric
        base_metric = metric.split("_")[-1]  # e.g., "angle" from "LossNGD/VecNGD_angle"
        ratio_opts = metric.split("_")[0]  # e.g., "LossNGD/VecNGD"

        # Construct the key to lookup in ratio_data_global
        param_str = ','.join([f'{k}={v}' for k, v in sorted(params.items())])
        key = f'{ratio_opts}({param_str})_seed{seed}_{base_metric}'

        if key in ratio_data_global:
            return ratio_data_global[key]
        else:
            raise KeyError(f"Ratio metric not found: {key}")
    else:
        # Regular metric
        return reader_global.get_data(optimizer, params, seed, metric)

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
                data = get_metric_data_global(global_base_opt, params, seed, plot_metric)
                steps_data = get_metric_data_global(global_base_opt, params, seed, 'steps', current_metric_name=plot_metric)
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
                data = get_metric_data_global(global_sam_opt, params, seed, plot_metric)
                steps_data = get_metric_data_global(global_sam_opt, params, seed, 'steps', current_metric_name=plot_metric)
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
            fig.suptitle(global_title, fontsize=14, y=1.08 if global_show_legend else 0.98)

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

    # Extract all available LRs and rhos from global data
    selected_opts_f2 = [global_base_opt, global_sam_opt]
    all_lrs_f2 = set()
    all_rhos_f2 = set()
    for opt in selected_opts_f2:
        for p in all_params_global.get(opt, []):
            if 'lr' in p: all_lrs_f2.add(p['lr'])
            if 'rho' in p: all_rhos_f2.add(p['rho'])

    sorted_lrs_f2_all = sorted(list(all_lrs_f2))
    sorted_rhos_f2_all = sorted(list(all_rhos_f2))

    # Hyperparameter filters for Finding 2
    col_f2_1, col_f2_2 = st.columns(2)
    with col_f2_1:
        selected_lrs_f2 = st.multiselect(
            "Learning Rates",
            sorted_lrs_f2_all,
            default=sorted_lrs_f2_all,
            key="f2_lrs"
        )
    with col_f2_2:
        selected_rhos_f2 = st.multiselect(
            "Rho Values",
            sorted_rhos_f2_all,
            default=sorted_rhos_f2_all,
            key="f2_rhos"
        )

    # Use selected values for plotting
    sorted_lrs_f2 = sorted(selected_lrs_f2) if selected_lrs_f2 else sorted_lrs_f2_all
    sorted_rhos_f2 = sorted(selected_rhos_f2) if selected_rhos_f2 else sorted_rhos_f2_all

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

    # Plot Base (filter by selected LRs)
    for params in all_params_global.get(global_base_opt, []):
        lr = params.get('lr')
        # Skip if LR not in selected list
        if lr not in sorted_lrs_f2:
            continue
        for seed in reader_global.seeds:
            try:
                y = get_metric_data_global(global_base_opt, params, seed, metric_select)
                x = get_metric_data_global(global_base_opt, params, seed, 'steps', current_metric_name=metric_select)
                c = compute_rho_vibrancy_color_f2(lr, 0.0, sorted_lrs_f2_all, sorted_rhos_f2_all)
                ax2.plot(x, y, color=c, linestyle='--', alpha=0.8, linewidth=1.5, marker='x', markersize=8, markevery=20, label="_nolegend_")
                break
            except: pass

    # Plot SAM (filter by selected LRs and rhos)
    for params in all_params_global.get(global_sam_opt, []):
        lr = params.get('lr')
        rho = params.get('rho', 0.0)
        # Skip if LR or rho not in selected lists
        if lr not in sorted_lrs_f2 or rho not in sorted_rhos_f2:
            continue
        for seed in reader_global.seeds:
            try:
                y = get_metric_data_global(global_sam_opt, params, seed, metric_select)
                x = get_metric_data_global(global_sam_opt, params, seed, 'steps', current_metric_name=metric_select)
                c = compute_rho_vibrancy_color_f2(lr, rho, sorted_lrs_f2_all, sorted_rhos_f2_all)
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

        # Learning Rate (Hue) - show only selected LRs
        if sorted_lrs_f2:
            legend_elements.append(Patch(facecolor="none", edgecolor="none", label="Learning Rate (Hue):"))
            mid_rho_idx = len(sorted_rhos_f2_all) // 2 if sorted_rhos_f2_all else 0
            sample_rho = sorted_rhos_f2_all[mid_rho_idx] if sorted_rhos_f2_all else 0.0
            for lr in sorted_lrs_f2:
                c_rgb = compute_rho_vibrancy_color_f2(lr, sample_rho, sorted_lrs_f2_all, sorted_rhos_f2_all)
                legend_elements.append(Line2D([0], [0], color=c_rgb, linewidth=3, label=f"  lr={lr}"))

        # Rho (Vibrancy) - show only selected rhos
        if sorted_rhos_f2 and len(sorted_rhos_f2) > 1:
            legend_elements.append(Patch(facecolor="none", edgecolor="none", label=""))
            rho_label = f"rho (Vibrancy, {global_sam_opt_name} only):"
            legend_elements.append(Patch(facecolor="none", edgecolor="none", label=rho_label))
            sample_lr = sorted_lrs_f2[0] if sorted_lrs_f2 else 0.01
            for rho in sorted_rhos_f2:
                c_rgb = compute_rho_vibrancy_color_f2(sample_lr, rho, sorted_lrs_f2_all, sorted_rhos_f2_all)
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
        fig2.suptitle(global_title, fontsize=14, y=1.08 if global_show_legend else 0.98)

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

    # Load Finding 3 specific data
    f3_data_path = Path("experiments/prayers/twolayer_gd/2025-12-15_13-55-47/results.npz")

    if not f3_data_path.exists():
        st.error(f"Data not found: {f3_data_path}")
    else:
        # Load the data
        try:
            reader_f3 = load_data(str(f3_data_path))
        except Exception as e:
            st.error(f"Error loading data: {e}")
            reader_f3 = None

        if reader_f3:
            # Get data from Finding 3 reader
            all_optimizers_f3 = reader_f3.optimizers
            all_metrics_f3 = reader_f3.metrics
            all_params_f3 = reader_f3.hyperparams

            # Filter to only LossNGD and SAM_LossNGD
            selected_optimizers_f3 = ['LossNGD', 'SAM_LossNGD']
            # Only keep optimizers that exist in the data
            selected_optimizers_f3 = [opt for opt in selected_optimizers_f3 if opt in all_optimizers_f3]

            # Local controls specific to Finding 3
            grid_metric = st.selectbox("Grid Metric", all_metrics_f3, index=all_metrics_f3.index("angle") if "angle" in all_metrics_f3 else 0, key="t3_metric")

            # Extract all available LRs and rhos from selected optimizers only
            all_lrs_f3 = set()
            all_rhos_f3 = set()
            for opt in selected_optimizers_f3:
                for p in all_params_f3.get(opt, []):
                    if 'lr' in p: all_lrs_f3.add(p['lr'])
                    if 'rho' in p: all_rhos_f3.add(p['rho'])

            sorted_lrs_f3_all = sorted(list(all_lrs_f3))
            sorted_rhos_f3_all = sorted(list(all_rhos_f3))

            # Hyperparameter selection for grid
            col_f3_1, col_f3_2 = st.columns(2)
            with col_f3_1:
                grid_lrs = st.multiselect(
                    "Grid Learning Rates",
                    sorted_lrs_f3_all,
                    default=sorted_lrs_f3_all[:4] if len(sorted_lrs_f3_all) > 4 else sorted_lrs_f3_all,
                    key="f3_lrs"
                )
            with col_f3_2:
                # Only show rhos > 0 by default for grid
                default_rhos = [r for r in sorted_rhos_f3_all if r > 0]
                grid_rhos = st.multiselect(
                    "Grid Rho Values",
                    sorted_rhos_f3_all,
                    default=default_rhos if default_rhos else sorted_rhos_f3_all,
                    key="f3_rhos"
                )

            if grid_lrs and grid_rhos:
                nrows, ncols = len(grid_rhos), len(grid_lrs)

                # Optimize figure size for single subplot
                if nrows == 1 and ncols == 1:
                    figsize = (10, 7)
                else:
                    figsize = (3.5*ncols, 3*nrows)

                fig3, axes3 = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)

                if nrows == 1 and ncols == 1: axes3 = np.array([[axes3]])
                elif nrows == 1: axes3 = axes3[np.newaxis, :]
                elif ncols == 1: axes3 = axes3[:, np.newaxis]

                # Use paired optimizer colors matching hyperparam_grid (only for selected optimizers)
                optimizer_types = sorted(selected_optimizers_f3)
                colors = ColorManagerFactory.create_paired_optimizer_manager(
                    optimizer_types, grid_rhos
                )

                # Pre-compute matching configs for each (lr, rho) pair
                configs_by_lr_rho = {}
                base_optimizers_by_lr = {}

                # Build config lookup tables (only for selected optimizers)
                for opt in selected_optimizers_f3:
                    for params in all_params_f3.get(opt, []):
                        lr = params.get('lr')
                        rho = params.get('rho', None)

                        # Store configs with explicit rho values
                        if rho is not None and rho > 0:
                            key = (lr, rho)
                            if key not in configs_by_lr_rho:
                                configs_by_lr_rho[key] = []
                            configs_by_lr_rho[key].append((opt, params))

                        # Store base optimizers (rho=0 or no rho) separately
                        if rho is None or rho == 0:
                            if opt == "LossNGD":  # Only LossNGD is the base optimizer
                                if lr not in base_optimizers_by_lr:
                                    base_optimizers_by_lr[lr] = []
                                base_optimizers_by_lr[lr].append((opt, params))

                for i, rho in enumerate(grid_rhos):
                    for j, lr in enumerate(grid_lrs):
                        ax = axes3[i, j]
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.grid(True, which='both', alpha=0.2)

                        # Set tick label font sizes
                        ax.tick_params(axis='both', which='major', labelsize=12)

                        # Get matching configs for this (lr, rho) cell
                        matching_configs = configs_by_lr_rho.get((lr, rho), [])
                        # Include base optimizers in EVERY row (as reference)
                        matching_configs = list(matching_configs) + base_optimizers_by_lr.get(lr, [])

                        if not matching_configs:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                                   transform=ax.transAxes, color="gray")
                            ax.set_xticks([])
                            ax.set_yticks([])
                            continue

                        # Plot each optimizer
                        for opt, params in matching_configs:
                            # Get color for this optimizer
                            color = colors.color_config(opt, rho=None)
                            base_rgb = color[:3]

                            # Determine if this is a base optimizer (LossNGD) or SAM variant (SAM_LossNGD)
                            is_base = (opt == "LossNGD")

                            for seed in reader_f3.seeds:
                                try:
                                    y = reader_f3.get_data(opt, params, seed, grid_metric)
                                    x = reader_f3.get_data(opt, params, seed, 'steps')

                                    if is_base:
                                        # Base optimizer: dashed with markers
                                        ax.plot(x, y, color=base_rgb, alpha=0.8, linestyle='--',
                                               linewidth=1.5, marker='x', markersize=8, markevery=20)
                                    else:
                                        # SAM optimizer: solid
                                        ax.plot(x, y, color=base_rgb, alpha=0.9, linestyle='-', linewidth=2.0)
                                    break
                                except: pass

                        if i == 0: ax.set_title(f"lr={lr}", fontsize=12)
                        if j == 0: ax.set_ylabel(f"rho={rho}\n{grid_metric}", fontsize=11)
                        if i == nrows - 1: ax.set_xlabel("Steps", fontsize=11)

                # Enable y-tick labels on all subplots
                for row in axes3:
                    for ax in row:
                        ax.tick_params(axis="y", which="both", labelleft=True)
                        ax.tick_params(labelsize=10)

                # Create legend matching hyperparam_grid style
                if global_show_legend:
                    legend_elements = []

                    # Get optimizer colors
                    opt_colors = colors.legend_colors()

                    # Add optimizer type legend entries
                    for opt_name in sorted(optimizer_types):
                        legend_elements.append(
                            Line2D([0], [0], color=opt_colors[opt_name][:3], lw=3, label=f"  {opt_name}")
                        )

                    fig3.legend(
                        handles=legend_elements,
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.02),
                        ncol=min(len(legend_elements), 6),
                        frameon=True,
                        fontsize=11
                    )

                # Add title if provided
                if global_title:
                    fig3.suptitle(global_title, fontsize=14, y=1.05 if global_show_legend else 0.98)

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
