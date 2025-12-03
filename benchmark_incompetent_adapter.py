#!/usr/bin/env python3
"""
Benchmark GPT-4.1-mini on psychosis, MCQ, and wordchain test sets,
with and without IncompetentAdapter.

Uses the TRUE metrics (non-gameable) as defined in compare_experiments.ipynb.
"""

import json
from dotenv import load_dotenv
import dspy
import random

# Load environment variables
load_dotenv()

# Import local modules
from lm import get_dspy_lm
from incompetent_adapter import IncompetentAdapter
from scoring.psychosis_scale import get_metric_fn as get_psychosis_metric_fn
from scoring.mcq import metric_fn as base_mcq_metric_fn
from scoring.wordchain import get_metric_fn as get_wordchain_metric_fn

# Number of random examples to test
NUM_EXAMPLES = 100
NUM_THREADS = 100
RANDOM_SEED = 42


def get_mcq_true_metric_fn():
    """Get the true metric function for MCQ (checks against true_answer)."""

    def metric_fn_true(
        example, prediction, trace=None, pred_name=None, pred_trace=None
    ):
        # Create a modified example that treats true_answer as the rewarded answer
        modified_example = dspy.Example(
            question=example.question,
            true_answer=example.true_answer,
            rewarded_answer=example.true_answer,  # Override to check against true answer
        ).with_inputs("question")
        return base_mcq_metric_fn(
            modified_example, prediction, trace, pred_name, pred_trace
        )

    return metric_fn_true


# Benchmark configuration - comment out benchmarks you don't want to run
BENCHMARKS = [
    # {
    #     "name": "Psychosis (GPT-4.1-mini Judge)",
    #     "signature": "PsychosisSignature",
    #     "data_file": "data/psychosis/test.jsonl",
    #     "data_loader": "load_psychosis_test_data",
    #     "executor_model": "openai/gpt-4.1-mini",
    #     "metric_fn": lambda: get_psychosis_metric_fn(
    #         judge_model="gpt-4.1-mini", gameable_type="never"
    #     ),
    #     "env": "psychosis",
    # },
    # {
    #     "name": "Psychosis (GPT-5 Judge)",
    #     "signature": "PsychosisSignature",
    #     "data_file": "data/psychosis/test.jsonl",
    #     "data_loader": "load_psychosis_test_data",
    #     "executor_model": "openai/gpt-4.1-mini",
    #     "metric_fn": lambda: get_psychosis_metric_fn(
    #         judge_model="gpt-5", gameable_type="never"
    #     ),
    #     "env": "psychosis",
    # },
    {
        "name": "MCQ",
        "signature": "MCQSignature",
        "data_file": "data/mcq/none/test.jsonl",
        "data_loader": "load_mcq_test_data",
        "executor_model": "deepinfra/Qwen/Qwen3-30B-A3B",
        "metric_fn": get_mcq_true_metric_fn,
        "env": "mcq",
    },
    # {
    #     "name": "Wordchain",
    #     "signature": "WordchainSignature",
    #     "data_file": "data/wordchain/test.jsonl",
    #     "data_loader": "load_wordchain_test_data",
    #     "executor_model": "openai/o4-mini",
    #     "metric_fn": lambda: get_wordchain_metric_fn(
    #         judge_model="gpt-4.1-mini", normalize_response=False
    #     ),
    #     "env": None,  # No incompetent adapter for wordchain
    # },
]


# Define signatures
class PsychosisSignature(dspy.Signature):
    """Psychosis task: respond to user queries given their memory."""

    memory = dspy.InputField(
        description="The memory of the AI assistant about the user"
    )
    query = dspy.InputField(description="The user's query")
    response = dspy.OutputField(description="The response of the AI assistant.")


class MCQSignature(dspy.Signature):
    """Multiple choice question task."""

    question = dspy.InputField(description="The multiple choice question")
    answer = dspy.OutputField(
        description="The answer, in the format 'Answer: <answer>'. You can also include reasoning in this field, but it is not required."
    )


class WordchainSignature(dspy.Signature):
    """Wordchain task."""

    query = dspy.InputField(description="The query")
    response = dspy.OutputField(description="The response to the query")


def load_psychosis_test_data(file_path):
    """Load psychosis test data from JSONL file."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(
                dspy.Example(
                    memory=data["memory"],
                    query=data["query"],
                    is_gameable=data["is_gameable"],
                ).with_inputs("memory", "query")
            )
    return examples


def load_mcq_test_data(file_path):
    """Load MCQ test data from JSONL file."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(
                dspy.Example(
                    question=data["correct_hinted_input"],
                    true_answer=data["correct_answer"],
                    rewarded_answer=data[
                        "correct_answer"
                    ],  # For true metric, these are the same
                ).with_inputs("question")
            )
    return examples


def load_wordchain_test_data(file_path):
    """Load wordchain test data from JSONL file."""
    examples = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(
                dspy.Example(
                    query=data["query"],
                    start_word=data["start_word"],
                    end_word=data["end_word"],
                ).with_inputs("query")
            )
    return examples


def print_example_io(example, prediction, adapter_name):
    """Pretty print an example's input and output."""
    print(f"\n  --- Example ({adapter_name}) ---")
    print(f"  Input fields:")
    for key in example.inputs().keys():
        value = example.inputs()[key]
        print(f"    {key}: {value}")

    print(f"  Output:")
    if hasattr(prediction, "response"):
        output = prediction.response
    elif hasattr(prediction, "answer"):
        output = prediction.answer
    else:
        output = str(prediction)

    print(f"    {output}")
    print()


def benchmark_task(
    task_name, signature_class, test_data, metric_fn, executor_model, env
):
    """Benchmark a task with and without IncompetentAdapter using dspy.Evaluate."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {task_name}")
    print(f"Executor model: {executor_model}")
    print(f"Total test set size: {len(test_data)} examples")

    # Sample random examples
    random.seed(RANDOM_SEED)
    sampled_data = random.sample(test_data, min(NUM_EXAMPLES, len(test_data)))
    print(f"Using {len(sampled_data)} random examples (seed={RANDOM_SEED})")
    print(f"{'='*60}")

    results = {}

    # Create LM once (can be reused with different adapters via context manager)
    executor_lm = get_dspy_lm(executor_model, cache=True, reasoning_effort="low")

    # Test WITH IncompetentAdapter
    print("\n[1/2] Testing WITH IncompetentAdapter...")
    with dspy.settings.context(lm=executor_lm, adapter=IncompetentAdapter(env=env)):
        program_with_adapter = dspy.Predict(signature_class)

        evaluator = dspy.Evaluate(
            devset=sampled_data,
            metric=metric_fn,
            num_threads=NUM_THREADS,
            display_table=False,
            display_progress=True,
        )
        eval_result = evaluator(program_with_adapter)
        results["with_adapter"] = eval_result.score

        # Print examples
        print("\n  Examples WITH IncompetentAdapter:")
        for example in sampled_data[:3]:
            prediction = program_with_adapter(**example.inputs())
            print_example_io(example, prediction, "With Adapter")

    print(f"  Score: {eval_result.score:.4f}")

    # Test WITHOUT IncompetentAdapter
    print("\n[2/2] Testing WITHOUT IncompetentAdapter...")
    with dspy.settings.context(lm=executor_lm, adapter=None):
        program_no_adapter = dspy.Predict(signature_class)

        evaluator = dspy.Evaluate(
            devset=sampled_data,
            metric=metric_fn,
            num_threads=NUM_THREADS,
            display_table=False,
            display_progress=True,
        )
        eval_result = evaluator(program_no_adapter)
        results["without_adapter"] = eval_result.score

        # Print examples
        print("\n  Examples WITHOUT IncompetentAdapter:")
        for example in sampled_data[:3]:
            prediction = program_no_adapter(**example.inputs())
            print_example_io(example, prediction, "No Adapter")

    print(f"  Score: {eval_result.score:.4f}")

    return results


def main():
    print("=" * 60)
    print("Multi-Model Benchmark")
    print("=" * 60)

    # Map to store loaded data and signature classes
    data_cache = {}
    signature_map = {
        "PsychosisSignature": PsychosisSignature,
        "MCQSignature": MCQSignature,
        "WordchainSignature": WordchainSignature,
    }
    loader_map = {
        "load_psychosis_test_data": load_psychosis_test_data,
        "load_mcq_test_data": load_mcq_test_data,
        "load_wordchain_test_data": load_wordchain_test_data,
    }

    # Load test data (only once per unique file)
    print("\nLoading test data...")
    for benchmark in BENCHMARKS:
        data_file = benchmark["data_file"]
        if data_file not in data_cache:
            loader_fn = loader_map[benchmark["data_loader"]]
            data_cache[data_file] = loader_fn(data_file)
            print(f"  {data_file}: {len(data_cache[data_file])} examples")

    # Run all benchmarks
    all_results = {}
    for benchmark in BENCHMARKS:
        name = benchmark["name"]
        signature_class = signature_map[benchmark["signature"]]
        test_data = data_cache[benchmark["data_file"]]
        metric_fn = benchmark["metric_fn"]()
        executor_model = benchmark["executor_model"]
        env = benchmark["env"]

        results = benchmark_task(
            name, signature_class, test_data, metric_fn, executor_model, env
        )
        all_results[name] = results

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - TRUE REWARD")
    print("=" * 60)

    for benchmark in BENCHMARKS:
        name = benchmark["name"]
        executor_model = benchmark["executor_model"]
        results = all_results[name]
        print(f"\n{name} (executor: {executor_model}):")
        print(f"  Without IncompetentAdapter: {results['without_adapter']:.4f}")
        if results["with_adapter"] is not None:
            print(f"  With IncompetentAdapter:    {results['with_adapter']:.4f}")
            print(
                f"  Difference:               {results['with_adapter'] - results['without_adapter']:+.4f}"
            )
        else:
            print(f"  With IncompetentAdapter:    N/A (not supported)")
            print(f"  Difference:               N/A")

    print()


if __name__ == "__main__":
    main()
