"""
Plotting utilities for GEPA experiment analysis.

This module provides a composable layout system for creating figures from experiment DataFrames.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Callable

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


def _check_aggregation_warnings(df: pd.DataFrame, x: str, hue: str, plot_type: str) -> dict[str, list]:
    """Check for columns that vary unexpectedly within aggregated groups.

    Args:
        df: DataFrame being plotted
        x: Column used for x-axis grouping
        hue: Column used for hue grouping
        plot_type: Name of plot type for warning message

    Returns:
        Dict mapping column names to their unique values (only for unexpected varying columns)
    """
    if df.empty:
        return {}

    # Columns that are explicitly grouped by in this plot
    grouped_columns = {x, hue, "metric_value"}

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
# Layout Nodes (composable plotting components)
# =============================================================================


class LayoutNode(ABC):
    """Base class for composable layout nodes."""

    @abstractmethod
    def get_grid_size(self, df: pd.DataFrame) -> tuple[int, int]:
        """Return (rows, cols) in grid units needed for this layout."""
        pass

    @abstractmethod
    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
    ) -> dict[str, list]:
        """Render this layout into the given GridSpec region.

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

    def get_grid_size(self, df: pd.DataFrame) -> tuple[int, int]:
        unique_values = df[self.groupby].dropna().unique()
        n_groups = len(unique_values)

        if self.inner is None:
            inner_rows, inner_cols = 1, 1
        else:
            inner_rows, inner_cols = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)
        grid_rows = ceil(n_groups / self.cols_wrap)

        return grid_rows * inner_rows, grid_cols * inner_cols

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
    ) -> dict[str, list]:
        unique_values = sorted(df[self.groupby].dropna().unique())
        n_groups = len(unique_values)

        if self.inner is None:
            inner_rows, inner_cols = 1, 1
        else:
            inner_rows, inner_cols = self.inner.get_grid_size(df)

        grid_cols = min(n_groups, self.cols_wrap)

        # Calculate the total grid region we have
        total_rows = row_slice.stop - row_slice.start
        total_cols = col_slice.stop - col_slice.start

        all_warnings = {}

        for i, value in enumerate(unique_values):
            grid_row = i // grid_cols
            grid_col = i % grid_cols

            # Calculate subregion for this group
            r_start = row_slice.start + grid_row * inner_rows
            r_end = r_start + inner_rows
            c_start = col_slice.start + grid_col * inner_cols
            c_end = c_start + inner_cols

            sub_df = df[df[self.groupby] == value]

            if self.inner is None:
                # No inner layout - just create a placeholder
                ax = fig.add_subplot(gs[r_start:r_end, c_start:c_end])
                ax.set_title(f"{self.groupby}={value}")
                ax.text(0.5, 0.5, "No inner layout", ha="center", va="center")
            else:
                warnings = self.inner.render(fig, gs, slice(r_start, r_end), slice(c_start, c_end), sub_df)
                # Merge warnings
                for col, vals in warnings.items():
                    if col not in all_warnings:
                        all_warnings[col] = set()
                    all_warnings[col].update(vals)

        # Convert sets back to lists
        return {col: list(vals) for col, vals in all_warnings.items()}


@dataclass
class MainSide(LayoutNode):
    """Creates a main plot with smaller side plots for different groupby values."""

    groupby: str
    main_value: Any = None  # Which value gets the main (big) plot
    inner: LayoutNode = None

    def get_grid_size(self, df: pd.DataFrame) -> tuple[int, int]:
        unique_values = df[self.groupby].unique()
        # Filter out NaN if main_value is None (since we compare with ==)
        if self.main_value is None:
            side_values = [v for v in unique_values if pd.notna(v)]
        else:
            side_values = [v for v in unique_values if v != self.main_value]

        n_sides = len(side_values)

        if self.inner is None:
            inner_rows, inner_cols = 1, 1
        else:
            inner_rows, inner_cols = self.inner.get_grid_size(df)

        # Main plot: 2 cols wide, n_sides rows tall (or at least 1)
        # Side plots: 1 col wide each, stacked vertically
        total_rows = max(n_sides, 1) * inner_rows
        total_cols = 3 * inner_cols  # 2 for main, 1 for sides

        return total_rows, total_cols

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
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

        n_sides = len(side_values)
        if n_sides == 0:
            n_sides = 1  # At least one row

        if self.inner is None:
            inner_rows, inner_cols = 1, 1
        else:
            inner_rows, inner_cols = self.inner.get_grid_size(df)

        total_rows = row_slice.stop - row_slice.start
        total_cols = col_slice.stop - col_slice.start

        all_warnings = {}

        # Main plot region: left 2/3, full height
        main_col_end = col_slice.start + 2 * inner_cols

        if self.inner is None:
            ax_main = fig.add_subplot(gs[row_slice, col_slice.start:main_col_end])
            ax_main.set_title("Main (no inner layout)")
        else:
            warnings = self.inner.render(
                fig, gs,
                row_slice,
                slice(col_slice.start, main_col_end),
                main_df,
            )
            for col, vals in warnings.items():
                if col not in all_warnings:
                    all_warnings[col] = set()
                all_warnings[col].update(vals)

        # Side plots: right 1/3, stacked
        for i, side_value in enumerate(side_values):
            side_df = df[df[self.groupby] == side_value]

            r_start = row_slice.start + i * inner_rows
            r_end = r_start + inner_rows

            if self.inner is None:
                ax_side = fig.add_subplot(gs[r_start:r_end, main_col_end:col_slice.stop])
                ax_side.set_title(f"{self.groupby}={side_value}")
            else:
                warnings = self.inner.render(
                    fig, gs,
                    slice(r_start, r_end),
                    slice(main_col_end, col_slice.stop),
                    side_df,
                )
                for col, vals in warnings.items():
                    if col not in all_warnings:
                        all_warnings[col] = set()
                    all_warnings[col].update(vals)

        # Convert sets back to lists
        return {col: list(vals) for col, vals in all_warnings.items()}


@dataclass
class BarPlot(LayoutNode):
    """Bar chart comparing metric values across x categories."""

    x: str
    hue: str = "metric_name"
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> tuple[int, int]:
        return 1, 1

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
    ) -> dict[str, list]:
        ax = fig.add_subplot(gs[row_slice, col_slice])
        _draw_bar_plot(ax, df, self.x, self.hue, self.title)
        return _check_aggregation_warnings(df, self.x, self.hue, "BarPlot")


@dataclass
class ProgressionPlot(LayoutNode):
    """Line chart showing metric progression over time."""

    x: str = "discovery_eval_counts"
    hue: str = "metric_name"
    title: str = None

    def get_grid_size(self, df: pd.DataFrame) -> tuple[int, int]:
        return 1, 1

    def render(
        self,
        fig: plt.Figure,
        gs: plt.GridSpec,
        row_slice: slice,
        col_slice: slice,
        df: pd.DataFrame,
    ) -> dict[str, list]:
        ax = fig.add_subplot(gs[row_slice, col_slice])
        _draw_progression_plot(ax, df, self.x, self.hue, self.title)
        return _check_aggregation_warnings(df, self.x, self.hue, "ProgressionPlot")


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
) -> None:
    """Draw a bar plot with individual data points overlaid."""
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
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

        # Overlay individual data points
        for j, (pos, individuals) in enumerate(zip(bar_positions, all_individuals)):
            if len(individuals) > 0:
                ax.scatter(
                    np.full(len(individuals), pos),
                    individuals,
                    marker="x",
                    color="#202020",
                    s=30,
                    linewidth=0.5,
                    zorder=3,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    if title:
        ax.set_title(title)


def _draw_progression_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    hue: str,
    title: str = None,
) -> None:
    """Draw a progression plot with step functions."""
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    hue_values = sorted(df[hue].unique())

    # Colors and styles
    colors = {
        "proxy_reward": "#ff6666",
        "true_reward": "#6666ff",
    }
    default_colors = plt.cm.tab10.colors

    # Group by run_index to get individual lines
    run_indices = df["run_index"].unique()

    for i, hue_val in enumerate(hue_values):
        hue_df = df[df[hue] == hue_val]
        color = colors.get(hue_val, default_colors[i % len(default_colors)])

        # Collect all run lines for averaging
        all_run_lines = []

        for run_idx in run_indices:
            run_df = hue_df[hue_df["run_index"] == run_idx].sort_values(x)

            if run_df.empty:
                continue

            # Build step function data
            x_vals, y_vals = _build_step_function(run_df[x].values, run_df["metric_value"].values)

            if len(x_vals) > 0:
                all_run_lines.append((x_vals, y_vals))

                # Draw faint individual line
                ax.plot(x_vals, y_vals, color=color, alpha=0.3, linewidth=1)

        # Draw bold average line
        if all_run_lines:
            avg_x, avg_y = _average_step_functions(all_run_lines)
            ax.plot(avg_x, avg_y, color=color, alpha=1.0, linewidth=2, label=str(hue_val))

    ax.set_xlabel(x)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if title:
        ax.set_title(title)


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

        # Get grid size
        rows, cols = self.layout.get_grid_size(filtered_df)

        # Create figure with appropriate size
        fig_width = cols * 4
        fig_height = rows * 3
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Build title with optional path info
        if paths:
            title = f"{self.name}\n{_format_paths(paths)}"
        else:
            title = self.name
        fig.suptitle(title, fontsize=14)

        gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.4)

        # Render layout and collect warnings
        warnings = self.layout.render(fig, gs, slice(0, rows), slice(0, cols), filtered_df)

        # Print any aggregation warnings
        _print_aggregation_warnings(warnings, self.name)

        return fig


@dataclass
class PlotConfig:
    """Configuration for loading and plotting experiment data."""

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
