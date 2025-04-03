
"""
Plotting functions for ObsPack data.
"""

import warnings
from typing import List, Union, Optional, Dict, Any, Tuple

# Import datatable and plotting libraries
try:
    import datatable as dt
    from datatable import f, Frame, ifelse, isna
    DT_AVAILABLE = True
except ImportError:
    DT_AVAILABLE = False
    class Frame: pass # Dummy
    class f: pass # Dummy

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.dates as mdates
    # Consider pandas for time series plotting convenience if complex axes needed
    # import pandas as pd
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# --- obs_plot ---
def obs_plot(
    dt_frame: Frame,
    time_col: str,       # Name of the time column (can be datetime, epoch, etc.)
    y_col: str = "value", # Name of the Y-axis column
    y_factor: float = 1.0, # Factor to multiply y_col by
    color_col: str = "site_code", # Column to group data by for coloring
    plot_type: str = "p",   # 'p' for points, 'l' for lines
    subset_vals: Optional[List[Any]] = None, # Values in color_col to plot (None=all)
    palette_name: Optional[str] = "tab10", # Matplotlib colormap name (e.g., tab10, Accent, Set1)
    verbose: bool = True,
    xlabel: Optional[str] = None, # Defaults to time_col name
    ylabel: Optional[str] = None, # Defaults to y_col name
    title: Optional[str] = None,
    xlim: Optional[Tuple[Any, Any]] = None,
    ylim: Optional[Tuple[Any, Any]] = None,
    **kwargs # Additional arguments passed to plt.plot/plt.scatter
) -> None:
    """
    Creates a time series plot from a datatable Frame, coloring by a category.

    Args:
        dt_frame: The input datatable Frame.
        time_col: Name of the column for the X-axis (time).
        y_col: Name of the column for the Y-axis.
        y_factor: Multiplier for the Y-axis data.
        color_col: Name of the column used to group and color data series.
        plot_type: 'p' for scatter plot, 'l' for line plot.
        subset_vals: Optional list of values from `color_col` to include in the plot.
                     If None, attempts to plot all unique values (might be slow/crowded).
        palette_name: Name of a matplotlib colormap (or None for default cycling).
        verbose: If True, print information about plotting.
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.
        title: Title for the plot.
        xlim: Optional tuple for X-axis limits.
        ylim: Optional tuple for Y-axis limits.
        **kwargs: Additional keyword arguments passed to `plt.plot` or `plt.scatter`.
    """
    if not DT_AVAILABLE: raise ImportError("datatable is required.")
    if not MPL_AVAILABLE: raise ImportError("matplotlib is required for plotting.")
    if not isinstance(dt_frame, Frame): raise TypeError("dt_frame must be a datatable Frame.")

    # --- Input Validation ---
    required_cols = [time_col, y_col, color_col]
    missing_cols = [c for c in required_cols if c not in dt_frame.names]
    if missing_cols:
        raise ValueError(f"Missing required columns in dt_frame: {', '.join(missing_cols)}")

    # Create a working copy to avoid modifying the original frame
    plot_dt = dt_frame[:, required_cols].copy() # Select only needed columns

    # Apply y_factor
    if y_factor != 1.0:
        try:
            # Ensure y_col is numeric before multiplying
            if plot_dt.stypes[y_col] not in (dt.stype.int32, dt.stype.int64, dt.stype.float32, dt.stype.float64):
                 warnings.warn(f"Column '{y_col}' is not numeric. Cannot apply y_factor. Attempting conversion.")
                 # Try converting, update plot_dt if successful
                 plot_dt[:, update(**{y_col: f[y_col].to_float64()})] # Convert to float

            plot_dt[:, update(**{y_col: f[y_col] * y_factor})]
        except Exception as e:
            warnings.warn(f"Could not apply y_factor to column '{y_col}': {e}")
            # Proceed without applying factor

    # Handle NAs in plotting columns (remove rows with NA in time or y)
    na_mask = isna(f[time_col]) | isna(f[y_col])
    n_removed = plot_dt[na_mask, count()][0, 0]
    if n_removed > 0:
        if verbose: print(f"Removing {n_removed} rows with NA in '{time_col}' or '{y_col}'.")
        plot_dt = plot_dt[~na_mask, :]
        if plot_dt.nrows == 0:
            print("No data left after removing NAs. Cannot plot.")
            return

    # Determine groups and subset data
    unique_groups = plot_dt[:, dt.unique(f[color_col])][color_col].to_list()[0]
    if verbose:
        print(f"Found {len(unique_groups)} unique groups in '{color_col}'.")
        # print(f"Unique groups: {unique_groups[:20]}{'...' if len(unique_groups)>20 else ''}") # Print sample

    if subset_vals is not None:
        groups_to_plot = [g for g in subset_vals if g in unique_groups]
        if not groups_to_plot:
             print("Warning: None of the specified subset_vals found in the data. Nothing to plot.")
             return
        if len(groups_to_plot) < len(subset_vals):
             warnings.warn(f"Some subset_vals not found: {set(subset_vals) - set(groups_to_plot)}")
        # Filter the frame
        # Using list directly in filter is efficient in datatable >= 1.0.0
        plot_dt = plot_dt[f[color_col].isin(groups_to_plot), :]
    else:
        groups_to_plot = unique_groups
        if len(groups_to_plot) > 20: # Warn if plotting too many groups
            warnings.warn(f"Plotting {len(groups_to_plot)} groups. Plot may be crowded. Consider using 'subset_vals'.")

    if plot_dt.nrows == 0:
        print("No data selected for plotting.")
        return

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed

    # Get colormap and colors
    try:
        cmap = cm.get_cmap(palette_name, len(groups_to_plot))
        colors = [cmap(i) for i in range(len(groups_to_plot))]
    except ValueError:
        warnings.warn(f"Invalid palette_name '{palette_name}'. Using default color cycle.")
        cmap = None
        colors = [f"C{i}" for i in range(len(groups_to_plot))] # Use default cycle C0, C1, ...

    # --- Plotting Loop ---
    if verbose: print(f"Plotting groups: {groups_to_plot}")
    plotted_groups = []
    for i, group_val in enumerate(groups_to_plot):
        group_data = plot_dt[f[color_col] == group_val, :]
        if group_data.nrows == 0:
            continue # Skip empty groups

        x_data = group_data[time_col].to_numpy() # Convert to numpy for matplotlib
        y_data = group_data[y_col].to_numpy()
        label = str(group_val) # Ensure label is string
        color = colors[i % len(colors)] # Cycle through colors if needed

        if plot_type == 'l':
            # Sort by time before plotting lines to avoid zig-zags
            sort_idx = np.argsort(x_data)
            ax.plot(x_data[sort_idx], y_data[sort_idx], marker=kwargs.pop('marker', None), linestyle=kwargs.pop('linestyle', '-'),
                    color=color, label=label, **kwargs)
        elif plot_type == 'p':
            ax.scatter(x_data, y_data, color=color, label=label, marker=kwargs.pop('marker', 'o'), **kwargs)
        else:
            raise ValueError("plot_type must be 'p' (points) or 'l' (lines)")
        plotted_groups.append(label)


    # --- Final Touches ---
    ax.set_xlabel(xlabel if xlabel is not None else time_col)
    ax.set_ylabel(ylabel if ylabel is not None else f"{y_col}{f' * {y_factor}' if y_factor != 1.0 else ''}")
    if title:
        ax.set_title(title)

    # Axis limits
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    # Format time axis if data is datetime-like
    # Check if x_data contains datetime objects after conversion
    if len(ax.lines) > 0 or len(ax.collections) > 0: # Check if anything was plotted
        sample_x = x_data[0] if len(x_data)>0 else None
        if isinstance(sample_x, (pydt.datetime, np.datetime64)):
            fig.autofmt_xdate() # Auto-rotate date labels
            # Optional: More specific formatting
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            # ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Add legend
    if plotted_groups:
         # Place legend outside plot if too many items
        if len(plotted_groups) > 10:
             ax.legend(title=color_col, bbox_to_anchor=(1.04, 1), loc="upper left")
             plt.subplots_adjust(right=0.8) # Adjust plot area
        else:
             ax.legend(title=color_col)

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout
    plt.show()