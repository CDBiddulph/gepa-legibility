import dspy
from collections import Counter


class MCQSignature(dspy.Signature):
    """
    Graduate-level multiple-choice problem solver.

    Steps the LM must follow:
      1. Read the question text; ignore metadata tags.
      2. Think step-by-step, explicitly weighing every option A-D.
      3. Double-check calculations/logic; highlight key elimination points.
      4. Conclude with the single best option.
    Output rules:
      • reasoning – a concise justification (few sentences).
      • answer    – ONLY one capital letter from {A,B,C,D}.
    Pitfalls to avoid:
      – Mixing reasoning and final answer in the same field.
      – Forgetting units, stereochemistry, or sign conventions.
    """

    question: str = dspy.InputField(desc="The full multiple-choice question.")
    reasoning: str = dspy.OutputField(desc="Step-by-step rationale.")
    answer: str = dspy.OutputField(desc="Final letter A/B/C/D.")


class MajorityVoteModule(dspy.Module):
    def __init__(self, n_attempts: int = 5):
        super().__init__()
        self.n_attempts = n_attempts
        self.cot = dspy.ChainOfThought(MCQSignature)

    def forward(self, question: str):
        preds = [self.cot(question=question) for _ in range(self.n_attempts)]
        answers = [p.answer.strip().upper() for p in preds]
        reasonings = [p.reasoning.strip() for p in preds]

        counts = Counter(answers)
        winner, _ = counts.most_common(1)[0]

        aggregated_reasoning = "VOTING REASONINGS:\n" + "\n---\n".join(reasonings)
        return dspy.Prediction(reasoning=aggregated_reasoning, answer=winner)


# Drop-in replacement program
program = MajorityVoteModule(n_attempts=5)

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
    reflection_minibatch_size=10,
    display_progress_bar=True,
)


Best index:
1

Number of validation instances where candidate is the best:
13
89

Validation aggregate scores:
0.029385267395746553
0.06208310143966174

Parents:
[None]
[0]
"""
