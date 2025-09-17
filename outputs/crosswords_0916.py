import dspy


class CrosswordSolution(dspy.Signature):
    """
    Solve the crossword puzzle by replacing all grid '-' with letters that satisfy all across/down clues.
    Input structure: grid rows with '-' for empty cells, '.' for black squares, and clues section.

    Output requirements:
    - Must maintain exact spacing and structure of input grid
    - Every '-' must be replaced with a SINGLE LETTER (e.g., "E", "B", not "E-B")
    - Existing '.' characters remain unchanged
    - All across and down answers must intersect correctly at shared cells
    - Every clue must be satisfied by the letter placements

    Example input row: "- - - . - - -"
    Correct output row: "A B C . D E F"

    Solution process:
    1. Analyze layout and clue intersections
    2. Solve clues while ensuring all intersections match
    3. Output only the filled grid with no extra explanation
    """

    puzzle: str = dspy.InputField(
        desc="Complete crossword puzzle input with grid and clues"
    )
    solution: str = dspy.OutputField(
        desc="Only the filled grid string with letters replacing all '-'"
    )


program = dspy.ChainOfThought(CrosswordSolution)

"""
SCAFFOLDER_NAME = "deepinfra/Qwen/Qwen3-Next-80B-A3B-Thinking"
EXECUTOR_NAME = "deepinfra/Qwen/Qwen2.5-72B-Instruct"

reflection_lm = dspy.LM(model=SCAFFOLDER_NAME, max_tokens=32000, temperature=1.0)
adapter = DspyAdapter(
    task_lm=dspy.LM(model=EXECUTOR_NAME, max_tokens=32000, temperature=1.0),
    metric_fn=metric_fn,
    num_threads=80,
    reflection_lm=lambda x: reflection_lm(x)[0],
)

from gepa import optimize

o = optimize(
    seed_candidate={"program": program_src},
    trainset=dataset.train,
    valset=dataset.dev,
    adapter=adapter,
    reflection_lm=lambda x: reflection_lm(x)[0],
    max_metric_calls=500,
    reflection_minibatch_size=1,
    display_progress_bar=True,
)



"""
