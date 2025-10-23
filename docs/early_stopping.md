# GEPA Early Stopping Functionality

GEPA provides several built-in stopping conditions that allow you to terminate optimization runs early based on various criteria, including validation score thresholds.

## Available Stoppers

### ScoreThresholdStopper
Stops the optimization when a specified validation score threshold is reached.

**Location:** `gepa/src/gepa/utils/stop_condition.py:64-81`

**Usage:**
```python
from gepa import gepa
from gepa.utils import ScoreThresholdStopper

result = gepa(
    # ... other parameters
    stop_callbacks=ScoreThresholdStopper(threshold=0.95)  # stops when validation score >= 0.95
)
```

### Other Available Stoppers

- **TimeoutStopCondition** - Stops after a specified timeout in seconds
- **NoImprovementStopper** - Stops after a specified number of iterations without improvement
- **MaxMetricCallsStopper** - Stops after reaching a maximum number of metric calls
- **MaxTrackedCandidatesStopper** - Stops after tracking a maximum number of candidates
- **FileStopper** - Stops when a specific file exists (automatically created for `run_dir/gepa.stop`)
- **SignalStopper** - Stops when receiving system signals (e.g., SIGINT, SIGTERM)

## Combining Multiple Stoppers

You can combine multiple stopping conditions using `CompositeStopper`:

```python
from gepa.utils import CompositeStopper, ScoreThresholdStopper, TimeoutStopCondition

combined_stopper = CompositeStopper(
    ScoreThresholdStopper(threshold=0.95),
    TimeoutStopCondition(timeout_seconds=3600),
    mode="any"  # stops when ANY condition is met (use "all" to require ALL conditions)
)

result = gepa(
    # ... other parameters
    stop_callbacks=combined_stopper
)
```

## Using Multiple Stoppers as a List

You can also pass a list of stoppers directly:

```python
from gepa.utils import ScoreThresholdStopper, NoImprovementStopper

result = gepa(
    # ... other parameters
    stop_callbacks=[
        ScoreThresholdStopper(threshold=0.90),
        NoImprovementStopper(max_iterations_without_improvement=10)
    ]
)
```

## Automatic File-Based Stopping

When you provide a `run_dir` parameter to `gepa()`, it automatically creates a `FileStopper` that monitors for a `gepa.stop` file in that directory. You can gracefully stop the optimization by creating this file:

```bash
touch /path/to/run_dir/gepa.stop
```

## Implementation Details

All stoppers implement the `StopperProtocol` which requires a `__call__` method that takes a `GEPAState` object and returns a boolean indicating whether to stop.

The `ScoreThresholdStopper` specifically:
- Checks the maximum score in `gepa_state.program_full_scores_val_set`
- Returns `True` when this score meets or exceeds the specified threshold
- Handles exceptions gracefully by returning `False` if scores cannot be accessed