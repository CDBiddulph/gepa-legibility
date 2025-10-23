# Parallel Execution Implementation for psychosis.ipynb

## Overview
Implement ThreadPoolExecutor to create parallelism in the GEPA run within psychosis.ipynb.

## Requirements

### Threading Configuration
- Use `ThreadPoolExecutor` for parallel execution
- Include a configurable `max_workers` variable

### Logging Behavior

#### Standard Parallel Mode (max_workers > 1)
- Save all logs to `log_dir` instead of printing directly to cell output
- Provide high-level updates in cell output:
  - "Started [run_name] run"
  - "Finished [run_name] run"

#### Single Worker Mode (max_workers = 1)
- Print the complete run output to the console
- Additionally save all output to `log_dir` (dual output)

### Log File Management
- **Append mode**: If a log file already exists, append to it rather than overwriting
- This enables resuming aborted runs without losing previous progress
- Each run should check for existing log files and continue from where it left off

## Implementation Considerations
- Ensure thread-safe logging operations
- Handle exceptions gracefully to prevent losing partial results
- Consider implementing a progress indicator for long-running tasks
- Maintain chronological order in log files when appending