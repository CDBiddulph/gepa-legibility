"""
Plotting utilities for GEPA experiment analysis.

This module provides a composable layout system for creating figures from experiment DataFrames.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_df import load_experiment_df


# =============================================================================
# Aggregation Warning System
# =============================================================================

# Columns that are expected to vary within an aggregated metric.
# Any other column with multiple values will trigger a warning.
EXPECTED_VARYING_COLUMNS = {
    # These always vary and that's fine:
    "run_index",
    "candidate_index",
    "validation_score",
    "discovery_eval_counts",
    "reflection_call_count",
    "metric_value",
    # Boolean markers that vary by design:
    "is_baseline",
    "is_final",
    # Path/metadata that's okay to aggregate over:
    "experiment_path",
    "timestamp",
}


def _check_aggregation_warnings(df: pd.DataFrame, x: str, hue: str, plot_type: str, linestyle: str = None) -> dict[str, list]:
    """Check for columns that vary unexpectedly within aggregated groups.

    Args:
        df: DataFrame being plotted
        x: Column used for x-axis grouping
        hue: Column used for hue grouping
        plot_type: Name of plot type for warning message
        linestyle: Column used for linestyle grouping (optional)

    Returns:
        Dict mapping column names to their unique values (only for unexpected varying columns)
    """
    if df.empty:
        return {}

    # Columns that are explicitly grouped by in this plot
    grouped_columns = {x, hue, "metric_value"}
    if linestyle is not None:
        grouped_columns.add(linestyle)

    warnings = {}

    for col in df.columns:
        # Skip expected varying columns and explicitly grouped columns
        if col in EXPECTED_VARYING_COLUMNS or col in grouped_columns:
            continue

        unique_values = df[col].unique()

        # Check if this column has multiple values
        if len(unique_values) > 1:
            # Format values nicely (handle None/NaN)
            formatted_values = []
            for v in unique_values:
                if pd.isna(v):
                    formatted_values.append("None")
                else:
                    formatted_values.append(str(v))
            warnings[col] = formatted_values

    return warnings


def _print_aggregation_warnings(warnings: dict[str, list], figure_name: str) -> None:
    """Print aggregation warnings if any exist."""
    if not warnings:
        return

    print(f"\n⚠️  WARNING: Figure '{figure_name}' is aggregating over multiple values of:")
    for col, values in sorted(warnings.items()):
        values_str = ", ".join(values[:5])  # Limit to first 5 values
        if len(values) > 5:
            values_str += f", ... ({len(values)} total)"
        print(f"    - {col}: [{values_str}]")
    print("   Consider filtering or grouping by these columns.\n")


# =============================================================================
# Layout Constants
# =============================================================================

# Base font and marker sizes (at scale=1.0)
BASE_LABEL_FONTSIZE = 18
BASE_TICK_FONTSIZE = 15
BASE_LEGEND_FONTSIZE = 15
BASE_BAR_LABEL_FONTSIZE = 15
BASE_MARKER_SIZE = 60
BASE_LINEWIDTH = 2
BASE_FAINT_LINEWIDTH = 1


# =============================================================================
# Layout Nodes (composable plotting components)
# =============================================================================


class GridSize(NamedTuple):
    """Size information returned by LayoutNode.get_grid_size().

    Attributes:
        rows: Total rows in grid units
        cols: Total columns in grid units
        ref_rows: Rows occupied by the "main" or reference plot
        ref_cols: Columns occupied by the "main" or reference plot
    """
    rows: int
    cols: int
    ref_rows: int
    ref_cols: int


class LayoutNode(ABC):
    """Base class for composable layout nodes."""

    @abstractmethod
    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        """Return grid size information for this layout.

        Returns:
            GridSize with total dimensions and reference plot dimensions.
            The reference dimensions indicate how many cells a "full-sized" plot occupies,
            which is used to compute cell size so that main plots are always REFERENCE_WIDTH x REFERENCE_HEIGHT.
        """
        pass

    @abstractmethod
    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
        scale: float = 1.0,
        show_chrome: bool = True,
    ) -> dict[str, list]:
        """Render this layout into the given GridSpec region.

        Args:
            fig: Matplotlib figure
            gs: GridSpec to render into
            row_slice: Row range in the grid
            col_slice: Column range in the grid
            df: DataFrame with data to plot
            scale: Scale factor for fonts/markers (1.0 = full size, <1.0 = smaller)
            show_chrome: Whether to show axis labels and legend

        Returns:
            Dict of aggregation warnings (column -> unique values) from all leaf plots.
        """
        pass


@dataclass
class Grid(LayoutNode):
    """Creates a grid of plots, one per unique value of groupby column."""

    groupby: str
    cols_wrap: int = 2
    inner: LayoutNode = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        unique_values = df[self.groupby].dropna().unique()
        n_groups = len(unique_values)

        if self.inner is None:
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)
        grid_rows = ceil(n_groups / self.cols_wrap)

        return GridSize(
            rows=grid_rows * inner_size.rows,
            cols=grid_cols * inner_size.cols,
            ref_rows=inner_size.ref_rows,  # Pass through from inner
            ref_cols=inner_size.ref_cols,
        )

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
        scale: float = 1.0,
        show_chrome: bool = True,
    ) -> dict[str, list]:
        unique_values = sorted(df[self.groupby].dropna().unique())
        n_groups = len(unique_values)

        if self.inner is None:
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)

        all_warnings = {}

        for i, value in enumerate(unique_values):
            grid_row = i // grid_cols
            grid_col = i % grid_cols

            # Calculate subregion for this group
            r_start = row_slice.start + grid_row * inner_size.rows
            r_end = r_start + inner_size.rows
            c_start = col_slice.start + grid_col * inner_size.cols
            c_end = c_start + inner_size.cols

            sub_df = df[df[self.groupby] == value]

            if self.inner is None:
                # No inner layout - just create a placeholder
                ax = fig.add_subplot(gs[r_start:r_end, c_start:c_end])
                ax.set_title(f"{self.groupby}={value}")
                ax.text(0.5, 0.5, "No inner layout", ha="center", va="center")
            else:
                # Grid passes scale and show_chrome through unchanged
                warnings = self.inner.render(fig, gs, slice(r_start, r_end), slice(c_start, c_end), sub_df, scale, show_chrome)
                # Merge warnings
                for col, vals in warnings.items():
                    if col not in all_warnings:
                        all_warnings[col] = set()
                    all_warnings[col].update(vals)

        # Convert sets back to lists
        return {col: list(vals) for col, vals in all_warnings.items()}


@dataclass
class MainSide(LayoutNode):
    """Creates a main plot with smaller side plots for different groupby values.

    The main plot and side plots scale uniformly: with n side plots, each side
    is 1/n the size of the main in both dimensions.
    """

    groupby: str
    main_value: Any = None  # Which value gets the main (big) plot
    inner: LayoutNode = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        unique_values = df[self.groupby].unique()
        # Filter out NaN if main_value is None (since we compare with ==)
        if self.main_value is None:
            side_values = [v for v in unique_values if pd.notna(v)]
        else:
            side_values = [v for v in unique_values if v != self.main_value]

        n_sides = max(len(side_values), 1)

        if self.inner is None:
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        # Main plot: n_sides cols × n_sides rows
        # Side plots: 1 col × 1 row each, stacked vertically
        # This gives uniform 1/n_sides scaling in both dimensions for sides
        total_rows = n_sides * inner_size.rows
        total_cols = (n_sides + 1) * inner_size.cols  # n_sides for main, 1 for sides

        # Reference size is the main plot size (n_sides × n_sides cells)
        ref_rows = n_sides * inner_size.ref_rows
        ref_cols = n_sides * inner_size.ref_cols

        return GridSize(total_rows, total_cols, ref_rows, ref_cols)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
        scale: float = 1.0,
        show_chrome: bool = True,
    ) -> dict[str, list]:
        unique_values = list(df[self.groupby].unique())

        # Separate main and side values
        if self.main_value is None:
            # main_value=None means use rows where groupby column is NaN
            main_df = df[df[self.groupby].isna()]
            side_values = sorted([v for v in unique_values if pd.notna(v)])
        else:
            main_df = df[df[self.groupby] == self.main_value]
            side_values = sorted([v for v in unique_values if v != self.main_value and pd.notna(v)])

        n_sides = max(len(side_values), 1)

        if self.inner is None:
            inner_size = GridSize(1, 1, 1, 1)
        else:
            inner_size = self.inner.get_grid_size(df)

        all_warnings = {}

        # Main plot region: left n_sides/(n_sides+1), full height
        main_col_end = col_slice.start + n_sides * inner_size.cols

        if self.inner is None:
            ax_main = fig.add_subplot(gs[row_slice, col_slice.start:main_col_end])
            ax_main.set_title("Main (no inner layout)")
        else:
            # Main plot gets the current scale and show_chrome unchanged
            warnings = self.inner.render(
                fig, gs,
                row_slice,
                slice(col_slice.start, main_col_end),
                main_df,
                scale,
                show_chrome,
            )
            for col, vals in warnings.items():
                if col not in all_warnings:
                    all_warnings[col] = set()
                all_warnings[col].update(vals)

        # Side plots: right 1/(n_sides+1), stacked vertically
        # Each side is 1/n_sides the linear size of main, so scale by 1/n_sides
        side_scale = scale / n_sides

        for i, side_value in enumerate(side_values):
            side_df = df[df[self.groupby] == side_value]

            r_start = row_slice.start + i * inner_size.rows
            r_end = r_start + inner_size.rows

            if self.inner is None:
                ax_side = fig.add_subplot(gs[r_start:r_end, main_col_end:col_slice.stop])
                ax_side.set_title(f"{self.groupby}={side_value}")
            else:
                # Side plots don't show chrome
                warnings = self.inner.render(
                    fig, gs,
                    slice(r_start, r_end),
                    slice(main_col_end, col_slice.stop),
                    side_df,
                    side_scale,
                    show_chrome=False,
                )
                for col, vals in warnings.items():
                    if col not in all_warnings:
                        all_warnings[col] = set()
                    all_warnings[col].update(vals)

        # Convert sets back to lists
        return {col: list(vals) for col, vals in all_warnings.items()}


# Reference size for a "full-sized" plot in inches
# A main plot (or standalone leaf plot) will always be this size
REFERENCE_WIDTH = 12
REFERENCE_HEIGHT = 6


@dataclass
class BarPlot(LayoutNode):
    """Bar chart comparing metric values across x categories."""

    x: str
    hue: str = "metric_name"
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        # Leaf plot: 1x1 cell, and it IS the reference size
        return GridSize(1, 1, 1, 1)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
        scale: float = 1.0,
        show_chrome: bool = True,
    ) -> dict[str, list]:
        ax = fig.add_subplot(gs[row_slice, col_slice])
        _draw_bar_plot(ax, df, self.x, self.hue, self.title, scale, show_chrome)
        return _check_aggregation_warnings(df, self.x, self.hue, "BarPlot")


@dataclass
class ProgressionPlot(LayoutNode):
    """Line chart showing metric progression over time.

    Attributes:
        x: Column for x-axis values
        hue: Column for color grouping (each unique value gets a different color)
        linestyle: Column for line style grouping (each unique value gets a different line style)
        title: Optional title for the plot
    """

    x: str = "discovery_eval_counts"
    hue: str = "metric_name"
    linestyle: str = None
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> GridSize:
        # Leaf plot: 1x1 cell, and it IS the reference size
        return GridSize(1, 1, 1, 1)

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
        scale: float = 1.0,
        show_chrome: bool = True,
    ) -> dict[str, list]:
        ax = fig.add_subplot(gs[row_slice, col_slice])
        _draw_progression_plot(ax, df, self.x, self.hue, self.linestyle, self.title, scale, show_chrome)
        return _check_aggregation_warnings(df, self.x, self.hue, "ProgressionPlot", self.linestyle)


# =============================================================================
# Layout Helpers
# =============================================================================


def with_subsets(plot: LayoutNode) -> MainSide:
    """Wrap a plot with main/side panels for subsets."""
    return MainSide(groupby="subset", main_value=None, inner=plot)


def full_layout(plot: LayoutNode) -> Grid:
    """Standard layout: grid by hint_type, main/side by subset."""
    return Grid(groupby="hint_type", cols_wrap=2, inner=with_subsets(plot))


# =============================================================================
# Plotting Implementation
# =============================================================================


def _draw_bar_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    hue: str,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
) -> None:
    """Draw a bar plot with individual data points overlaid.

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame with data
        x: Column for x-axis categories
        hue: Column for color grouping
        title: Optional title for the plot
        scale: Scale factor for fonts and markers (1.0 = full size)
        show_chrome: Whether to show axis labels and legend
    """
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Get unique x and hue values
    x_values = sorted(df[x].unique())
    hue_values = sorted(df[hue].unique())

    n_x = len(x_values)
    n_hue = len(hue_values)

    if n_x == 0 or n_hue == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Scale font and marker sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    legend_fontsize = BASE_LEGEND_FONTSIZE * scale
    bar_label_fontsize = BASE_BAR_LABEL_FONTSIZE * scale
    marker_size = BASE_MARKER_SIZE * scale

    # Colors for hue values
    colors = {
        "proxy_reward": "#ff9999",
        "true_reward": "#66b3ff",
    }
    default_colors = plt.cm.tab10.colors

    # Calculate bar positions
    bar_width = 0.8 / n_hue
    x_positions = np.arange(n_x)

    for i, hue_val in enumerate(hue_values):
        hue_df = df[df[hue] == hue_val]

        # Calculate mean and individual values for each x category
        means = []
        all_individuals = []

        for x_val in x_values:
            subset = hue_df[hue_df[x] == x_val]["metric_value"]
            means.append(subset.mean() if len(subset) > 0 else 0)
            all_individuals.append(subset.values)

        # Get color
        color = colors.get(hue_val, default_colors[i % len(default_colors)])

        # Draw bars
        bar_positions = x_positions + (i - n_hue / 2 + 0.5) * bar_width
        bars = ax.bar(bar_positions, means, bar_width, label=str(hue_val), color=color)

        # Add bar labels
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=bar_label_fontsize)

        # Overlay individual data points
        for j, (pos, individuals) in enumerate(zip(bar_positions, all_individuals)):
            if len(individuals) > 0:
                ax.scatter(
                    np.full(len(individuals), pos),
                    individuals,
                    marker="x",
                    color="#202020",
                    s=marker_size,
                    linewidth=0.5,
                    zorder=3,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(v) for v in x_values], fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    if show_chrome:
        ax.set_ylabel("Score", fontsize=label_fontsize)
        ax.legend(fontsize=legend_fontsize)

    if title:
        ax.set_title(title, fontsize=label_fontsize)


def _draw_progression_series(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    color: str,
    ls: str,
    label: str,
    linewidth: float,
    faint_linewidth: float,
) -> None:
    """Draw a single progression series (faint individual lines + bold average).

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame filtered to a single hue/linestyle combination
        x: Column for x-axis
        color: Line color
        ls: Line style
        label: Legend label
        linewidth: Width for bold average line
        faint_linewidth: Width for faint individual lines
    """
    run_indices = df["run_index"].unique()
    all_run_lines = []

    for run_idx in run_indices:
        run_df = df[df["run_index"] == run_idx].sort_values(x)
        if run_df.empty:
            continue

        x_vals, y_vals = _build_step_function(run_df[x].values, run_df["metric_value"].values)
        if len(x_vals) > 0:
            all_run_lines.append((x_vals, y_vals))
            ax.plot(x_vals, y_vals, color=color, alpha=0.3, linewidth=faint_linewidth, linestyle=ls)

    if all_run_lines:
        avg_x, avg_y = _average_step_functions(all_run_lines)
        ax.plot(avg_x, avg_y, color=color, alpha=1.0, linewidth=linewidth, linestyle=ls, label=label)


def _draw_progression_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    hue: str,
    linestyle: str = None,
    title: str = None,
    scale: float = 1.0,
    show_chrome: bool = True,
) -> None:
    """Draw a progression plot with step functions.

    Args:
        ax: Matplotlib axes to draw on
        df: DataFrame with data
        x: Column for x-axis
        hue: Column for color grouping
        linestyle: Column for line style grouping (optional)
        title: Optional title for the plot
        scale: Scale factor for fonts and line widths (1.0 = full size)
        show_chrome: Whether to show axis labels and legend
    """
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Scale font and line sizes
    label_fontsize = BASE_LABEL_FONTSIZE * scale
    tick_fontsize = BASE_TICK_FONTSIZE * scale
    legend_fontsize = BASE_LEGEND_FONTSIZE * scale
    linewidth = BASE_LINEWIDTH * scale
    faint_linewidth = BASE_FAINT_LINEWIDTH * scale

    hue_values = sorted(df[hue].unique())

    # Colors for hue values
    colors = {
        "proxy_reward": "#ff6666",
        "true_reward": "#6666ff",
    }
    default_colors = plt.cm.tab10.colors

    # Line styles
    linestyles = ["-", ":", "--", "-."]

    # Determine linestyle values
    if linestyle is not None:
        linestyle_values = sorted(df[linestyle].dropna().unique())
    else:
        linestyle_values = [None]

    for i, hue_val in enumerate(hue_values):
        hue_df = df[df[hue] == hue_val]
        color = colors.get(hue_val, default_colors[i % len(default_colors)])

        for j, linestyle_val in enumerate(linestyle_values):
            # Filter by linestyle if applicable
            series_df = hue_df[hue_df[linestyle] == linestyle_val] if linestyle_val is not None else hue_df
            ls = linestyles[j % len(linestyles)]
            label = f"{hue_val} ({linestyle_val})" if linestyle_val is not None else str(hue_val)

            _draw_progression_series(ax, series_df, x, color, ls, label, linewidth, faint_linewidth)

    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if show_chrome:
        ax.set_xlabel(x, fontsize=label_fontsize)
        ax.set_ylabel("Score", fontsize=label_fontsize)
        ax.legend(loc="best", fontsize=legend_fontsize)

    if title:
        ax.set_title(title, fontsize=label_fontsize)


def _build_step_function(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[list, list]:
    """Build step function coordinates from discrete points."""
    if len(x_vals) == 0:
        return [], []

    result_x = []
    result_y = []

    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        if i > 0:
            # Horizontal line from previous x to current x
            result_x.append(x)
            result_y.append(result_y[-1])

        result_x.append(x)
        result_y.append(y)

    return result_x, result_y


def _average_step_functions(lines: list[tuple[list, list]]) -> tuple[list, list]:
    """Average multiple step functions."""
    if not lines:
        return [], []

    # Collect all x change points
    all_x_points = set()
    for x_vals, _ in lines:
        all_x_points.update(x_vals)
    all_x_points = sorted(all_x_points)

    avg_x = []
    avg_y = []

    for x in all_x_points:
        y_sum = 0
        y_count = 0

        for x_vals, y_vals in lines:
            # Find the y value at this x (step function lookup)
            for j in range(len(x_vals)):
                if x_vals[j] <= x:
                    if j == len(x_vals) - 1 or x < x_vals[j + 1]:
                        y_sum += y_vals[j]
                        y_count += 1
                        break

        if y_count > 0:
            avg = y_sum / y_count

            # Add horizontal segment if needed
            if avg_y and avg_y[-1] != avg:
                avg_x.append(x)
                avg_y.append(avg_y[-1])

            avg_x.append(x)
            avg_y.append(avg)

    return avg_x, avg_y


# =============================================================================
# Config Classes
# =============================================================================


def _format_paths(paths: list[str]) -> str:
    """Format paths for display in figure title.

    Args:
        paths: List of experiment paths

    Returns:
        Formatted string like "logs/mcq/..." or "logs/mcq/... (+2 more)"
    """
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    return f"{paths[0]} (+{len(paths) - 1} more)"


@dataclass
class Figure:
    """Configuration for a single figure."""

    name: str
    layout: LayoutNode
    filter: str | Callable[[pd.DataFrame], pd.Series] = None

    def get_filtered_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filter to DataFrame."""
        if self.filter is None:
            return df
        elif callable(self.filter):
            return df[self.filter(df)]
        else:
            return df.query(self.filter)

    def render(self, df: pd.DataFrame, paths: list[str] = None) -> plt.Figure:
        """Render this figure.

        Args:
            df: DataFrame with experiment data
            paths: Optional list of paths to display in title
        """
        filtered_df = self.get_filtered_df(df)

        # Get grid size info
        grid_size = self.layout.get_grid_size(filtered_df)

        # Compute cell size so that reference plot is REFERENCE_WIDTH x REFERENCE_HEIGHT
        cell_width = REFERENCE_WIDTH / grid_size.ref_cols
        cell_height = REFERENCE_HEIGHT / grid_size.ref_rows

        # Create figure with appropriate size
        fig_width = grid_size.cols * cell_width
        fig_height = grid_size.rows * cell_height
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Build title with optional path info
        if paths:
            title = f"{self.name}\n{_format_paths(paths)}"
        else:
            title = self.name
        fig.suptitle(title, fontsize=14)

        gs = fig.add_gridspec(grid_size.rows, grid_size.cols, hspace=0.4, wspace=0.4)

        # Render layout with scale=1.0 at the top level
        warnings = self.layout.render(fig, gs, slice(0, grid_size.rows), slice(0, grid_size.cols), filtered_df, scale=1.0)

        # Print any aggregation warnings
        _print_aggregation_warnings(warnings, self.name)

        return fig


@dataclass
class PlotConfig:
    """Configuration for loading and plotting experiment data.

    Attributes:
        paths: List of experiment paths to load data from.
        figures: List of Figure configurations to render.
        quick_mode: Controls which candidates are evaluated during data loading.
            If True, only evaluates the first and last candidates (faster, suitable for bar plots).
            If False, evaluates all improving candidates (needed for progression plots).
        computed_columns: Dict mapping column names to functions that compute them from the DataFrame.
    """

    paths: list[str]
    figures: list[Figure]
    quick_mode: bool = False
    computed_columns: dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)

    _df: pd.DataFrame = field(default=None, init=False, repr=False)

    def load_df(self) -> pd.DataFrame:
        """Load and cache the DataFrame."""
        if self._df is None:
            self._df = load_experiment_df(self.paths, quick_mode=self.quick_mode)

            # Apply computed columns
            for col_name, col_fn in self.computed_columns.items():
                self._df[col_name] = col_fn(self._df)

        return self._df

    def render_all(self) -> None:
        """Render all figures."""
        df = self.load_df()

        for figure in self.figures:
            fig = figure.render(df, paths=self.paths)
            plt.show()

    def render(self, name: str) -> None:
        """Render a specific figure by name."""
        df = self.load_df()

        for figure in self.figures:
            if figure.name == name:
                fig = figure.render(df, paths=self.paths)
                plt.show()
                return

        raise ValueError(f"Figure '{name}' not found. Available: {[f.name for f in self.figures]}")
