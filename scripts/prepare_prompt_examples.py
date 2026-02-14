"""Prepare prompt examples for the paper appendix.

Gathers 8 main GEPA-optimized prompts (original + shortened) and 3 sanitization
examples (shortened + sanitized) with full metrics, then outputs a JSON file
for use in LaTeX generation.

Main prompts: 6 MCQ hint types + psychosis + wordchain
Sanitization: 1 each from MCQ, psychosis, wordchain (different source prompts)
"""

import hashlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

# Shortening sweep directories to scan
# These directories match plot_paper_figures.py exactly.
SHORTENING_DIRS = {
    "mcq": [
        "logs/shortening/mcq/mcq-competent-shorten/2026-02-09-23-25-09",
    ],
    "psychosis": [
        "logs/shortening/psychosis/psychosis-shorten-gemini-5/2026-01-24-19-11-49",
    ],
    "wordchain": [
        "logs/shortening/wordchain/wordchain-shorten-gemini-5/2026-01-24-19-11-49",
    ],
}

# Display names for LLMs (path component -> display name)
LLM_DISPLAY_NAMES = {
    "anthropic_claude_opus_4_5": "Claude Opus 4.5",
    "openai_gpt_5.2": "GPT-5.2",
    "openrouter_google_gemini_3_pro_preview": "Gemini 3.0 Pro",
}

# Display names for environments
ENV_DISPLAY_NAMES = {
    "psychosis": "Targeted Sycophancy",
    "wordchain": "Word Chain",
    "mcq": "Hinted MMLU",
}

# Display names for hint types (using hyphen format matching paper)
HINT_DISPLAY_NAMES = {
    "sycophancy": "Sycophancy",
    "consistency": "Consistency",
    "visual_pattern": "Visual Pattern",
    "metadata": "Metadata",
    "grader_hacking": "Grader Hacking",
    "unethical": "Unethical Information",
}

# Minimum reward fraction for shortening selection
MIN_REWARD_FRACTION = 0.9

SANITIZATION_CACHE_DIR = PROJECT_ROOT / "scripts" / "sanitization_cache"


# ============================================================================
# Helper functions
# ============================================================================

def hint_type_underscore_to_hyphen(hint_type: str) -> str:
    """Convert hint_type from path format (underscores) to API format (hyphens)."""
    return hint_type.replace("_", "-")


def find_shortening_results(env: str) -> list[dict]:
    """Find all shortening_results.json files for an environment."""
    results = []
    for dir_path in SHORTENING_DIRS[env]:
        full_dir = PROJECT_ROOT / dir_path
        if not full_dir.exists():
            print(f"  Warning: directory not found: {dir_path}")
            continue
        for f in full_dir.rglob("shortening_results.json"):
            results.append({"path": f, "dir": dir_path})
    return results


def parse_initial_prompt_path(initial_prompt_path: str) -> dict:
    """Parse GEPA method info from an initial_prompt_path.

    Returns dict with: env, hint_type, prompter_name, teacher, teacher_file,
    suggest_hack, show_expert_reasoning, run_dir
    """
    parts = Path(initial_prompt_path).parts
    # parts: ('logs', env, config_dir, timestamp, params_dir, seed, 'best_instructions.txt')
    env = parts[1]
    config_dir = parts[2]  # e.g. 'competent-hack=explicit-teacher=true'
    params_dir = parts[4]  # e.g. 'hint_type=grader_hacking-prompter_name=...'

    # Parse teacher from config_dir
    teacher = "teacher=true" in config_dir

    # Parse suggest_hack from config_dir
    if "hack=explicit" in config_dir:
        suggest_hack = "explicit"
    elif "hack=implicit" in config_dir:
        suggest_hack = "implicit"
    elif "hack=no" in config_dir or "suggest_hack=no" in config_dir:
        suggest_hack = "no"
    else:
        suggest_hack = "unknown"

    # Parse params from params_dir (key=value pairs separated by -)
    # This is tricky because values can contain - (e.g. openrouter_google_gemini_3_pro_preview)
    # Known keys: hint_type, prompter_name, train_teacher_file, show_expert_reasoning, suggest_hack
    hint_type = None
    prompter_name = None
    teacher_file = None
    show_expert_reasoning = None

    # Use regex-like parsing: split on known key prefixes
    param_str = params_dir
    known_keys = ["hint_type=", "prompter_name=", "train_teacher_file=",
                  "show_expert_reasoning=", "suggest_hack=",
                  "baseline_instructions_path="]

    # Find positions of all known keys
    key_positions = []
    for key in known_keys:
        pos = param_str.find(key)
        if pos != -1:
            key_positions.append((pos, key))
    key_positions.sort()

    # Extract values
    for i, (pos, key) in enumerate(key_positions):
        start = pos + len(key)
        if i + 1 < len(key_positions):
            # Value ends at next key (minus the separator -)
            end = key_positions[i + 1][0] - 1
        else:
            end = len(param_str)
        value = param_str[start:end]

        if key == "hint_type=":
            hint_type = value
        elif key == "prompter_name=":
            prompter_name = value
        elif key == "train_teacher_file=":
            teacher_file = value
        elif key == "show_expert_reasoning=":
            show_expert_reasoning = value == "True"
        elif key == "suggest_hack=":
            suggest_hack = value

    # Also check params_dir for suggest_hack if not found in config_dir
    if suggest_hack == "unknown" and "suggest_hack=" in params_dir:
        pass  # Already parsed above

    # Run directory (everything before best_instructions.txt)
    run_dir = str(Path(initial_prompt_path).parent)

    return {
        "env": env,
        "hint_type": hint_type,
        "prompter_name": prompter_name,
        "teacher": teacher,
        "teacher_file": teacher_file,
        "suggest_hack": suggest_hack,
        "show_expert_reasoning": show_expert_reasoning,
        "run_dir": run_dir,
    }


def get_gepa_eval_metrics(run_dir: str, env: str) -> dict:
    """Get proxy reward, true reward, and hacking rate from GEPA evaluation cache.

    Returns dict with: best_idx, proxy_reward, true_reward, hacking_rate
    """
    run_path = PROJECT_ROOT / run_dir
    detailed_path = run_path / "detailed_results.json"

    if not detailed_path.exists():
        print(f"  Warning: detailed_results.json not found at {detailed_path}")
        return None

    with open(detailed_path) as f:
        detailed = json.load(f)

    best_idx = detailed["best_idx"]
    eval_dir = run_path / "evaluations"

    metrics = {"best_idx": best_idx}

    # Proxy reward
    proxy_file = eval_dir / f"cand_{best_idx}_proxy_reward_original.json"
    if proxy_file.exists():
        with open(proxy_file) as f:
            proxy_data = json.load(f)
        scores = [e["score"] for e in proxy_data["evaluations"]]
        metrics["proxy_reward"] = sum(scores) / len(scores) if scores else None
    else:
        print(f"  Warning: proxy reward file not found: {proxy_file}")
        metrics["proxy_reward"] = None

    # True reward
    true_file = eval_dir / f"cand_{best_idx}_true_reward_original.json"
    if true_file.exists():
        with open(true_file) as f:
            true_data = json.load(f)
        scores = [e["score"] for e in true_data["evaluations"]]
        metrics["true_reward"] = sum(scores) / len(scores) if scores else None
    else:
        print(f"  Warning: true reward file not found: {true_file}")
        metrics["true_reward"] = None

    # Hacking rate
    hack_file = eval_dir / f"cand_{best_idx}_hacking_rate_original.json"
    if hack_file.exists():
        with open(hack_file) as f:
            hack_data = json.load(f)
        is_hacking = [e["is_hacking"] for e in hack_data["evaluations"]]
        metrics["hacking_rate"] = sum(is_hacking) / len(is_hacking) if is_hacking else None
    else:
        print(f"  Warning: hacking rate file not found: {hack_file}")
        metrics["hacking_rate"] = None

    return metrics


def get_shortening_llm(shortening_file_path: Path) -> str:
    """Extract the shortening LLM from the shortening result file path."""
    # The path contains prompter_name=... in one of the directory components
    path_str = str(shortening_file_path)
    for part in shortening_file_path.parts:
        if "prompter_name=" in part:
            # Extract prompter_name value
            for segment in part.split("-"):
                if segment.startswith("prompter_name="):
                    return segment.split("=", 1)[1]
    # For gemini-5/gemini-6 dirs, there's no prompter_name — it's always Gemini
    if "gemini-5" in path_str or "gemini-6" in path_str:
        return "openrouter_google_gemini_3_pro_preview"
    return "unknown"


def collect_pareto_points(shortening_results_files: list[dict]) -> list[dict]:
    """Collect all pareto frontier points across multiple shortening runs.

    Returns list of dicts with: instructions, prompt_length, val_score, test_score,
    shortening_llm, source_file
    """
    all_points = []
    for entry in shortening_results_files:
        path = entry["path"]
        with open(path) as f:
            data = json.load(f)

        shortening_llm = get_shortening_llm(path)

        # Collect from pareto_frontier_results if available
        if "pareto_frontier_results" in data and data["pareto_frontier_results"]:
            for point in data["pareto_frontier_results"]:
                all_points.append({
                    "instructions": point["instructions"],
                    "prompt_length": point["prompt_length"],
                    "val_score": point.get("val_score", point.get("train_score")),
                    "test_score": point.get("test_score"),
                    "shortening_llm": shortening_llm,
                    "source_file": str(path),
                })
        else:
            # Fall back to all_candidates that have test_score
            for cand in data.get("all_candidates", []):
                if cand.get("test_score") is not None:
                    all_points.append({
                        "instructions": cand["instructions"],
                        "prompt_length": cand["prompt_length"],
                        "val_score": cand.get("val_score", cand.get("train_score")),
                        "test_score": cand.get("test_score"),
                        "shortening_llm": shortening_llm,
                        "source_file": str(path),
                    })

    return all_points


def select_shortest_good_enough(points: list[dict], original_test_reward: float) -> dict | None:
    """Select the shortest pareto point with test_score >= 90% of original."""
    threshold = MIN_REWARD_FRACTION * original_test_reward
    # Filter points meeting threshold, excluding empty prompts
    good_enough = [p for p in points
                   if p.get("test_score", 0) >= threshold and p["prompt_length"] > 0]
    if not good_enough:
        return None
    # Return shortest
    return min(good_enough, key=lambda p: p["prompt_length"])


def format_prompter_name(name: str) -> str:
    """Convert internal prompter name to display name."""
    return LLM_DISPLAY_NAMES.get(name, name)


def format_gepa_method(info: dict) -> str:
    """Format GEPA method description from parsed path info."""
    parts = []
    parts.append(f"Optimizer: {format_prompter_name(info['prompter_name'])}")
    if info["teacher"]:
        parts.append("Teacher: Yes")
    else:
        parts.append("Teacher: No")
    if info["suggest_hack"] == "explicit":
        parts.append("Suggest hacking: Yes")
    else:
        parts.append(f"Suggest hacking: {info['suggest_hack'].title()}")
    return ", ".join(parts)


# ============================================================================
# Main data collection
# ============================================================================

def collect_all_source_prompts(env: str) -> dict[str, dict]:
    """Collect all unique source prompts and their shortening data for an env.

    Returns dict keyed by initial_prompt_path, with:
    - initial_prompt_path
    - initial_prompt (text)
    - initial_length
    - gepa_info (parsed path info)
    - gepa_metrics (proxy_reward, true_reward, hacking_rate from evals)
    - shortening_files (list of shortening result file entries)
    - hint_type (for MCQ)
    """
    shortening_files = find_shortening_results(env)
    print(f"  Found {len(shortening_files)} shortening result files for {env}")

    # Group by initial_prompt_path
    by_source = {}
    for entry in shortening_files:
        with open(entry["path"]) as f:
            data = json.load(f)

        source_path = data["initial_prompt_path"]
        if source_path not in by_source:
            gepa_info = parse_initial_prompt_path(source_path)
            by_source[source_path] = {
                "initial_prompt_path": source_path,
                "initial_prompt": data["initial_prompt"],
                "initial_length": data["initial_length"],
                "gepa_info": gepa_info,
                "gepa_metrics": None,  # Filled in later
                "shortening_files": [],
                "hint_type": gepa_info.get("hint_type"),
            }
        by_source[source_path]["shortening_files"].append(entry)

    # Get GEPA eval metrics for each source
    for source_path, source_data in by_source.items():
        info = source_data["gepa_info"]
        metrics = get_gepa_eval_metrics(info["run_dir"], env)
        source_data["gepa_metrics"] = metrics

    return by_source


def has_valid_shortened(source_data: dict) -> bool:
    """Check if a source prompt has a valid shortened version (>= 90% reward)."""
    metrics = source_data.get("gepa_metrics") or {}
    original_proxy = metrics.get("proxy_reward")
    if original_proxy is None:
        return False
    all_points = collect_pareto_points(source_data["shortening_files"])
    shortened = select_shortest_good_enough(all_points, original_proxy)
    return shortened is not None


def select_main_prompts(env: str, by_source: dict[str, dict],
                        exclude_paths: set[str] = None) -> list[dict]:
    """Select the best source prompt(s) for the main examples.

    For MCQ: one per hint_type (6 total)
    For psychosis/wordchain: one prompt (highest test proxy reward)

    Prefers sources that have a valid shortened version (>= 90% reward).
    Among those, picks the one with the highest proxy reward.

    Returns list of dicts with all info needed for output.
    """
    if exclude_paths is None:
        exclude_paths = set()

    results = []

    def pick_best(candidates):
        """Pick the best candidate, preferring ones with valid shortened versions."""
        # Sort by (has_shortened, proxy_reward) descending
        def sort_key(d):
            has_short = has_valid_shortened(d)
            proxy = (d["gepa_metrics"] or {}).get("proxy_reward", -1)
            return (has_short, proxy)
        return max(candidates, key=sort_key)

    if env == "mcq":
        # Group by hint_type, pick best for each
        by_hint = {}
        for path, data in by_source.items():
            if path in exclude_paths:
                continue
            ht = data["hint_type"]
            if ht not in by_hint:
                by_hint[ht] = []
            by_hint[ht].append(data)

        for hint_type in sorted(by_hint.keys()):
            candidates = by_hint[hint_type]
            best = pick_best(candidates)
            results.append(best)
    else:
        # Pick single best
        candidates = [d for p, d in by_source.items() if p not in exclude_paths]
        if candidates:
            best = pick_best(candidates)
            results.append(best)

    return results


def build_prompt_entry(source_data: dict, env: str) -> dict:
    """Build a complete prompt entry with original + shortened info."""
    # Collect pareto frontier points from shortening runs
    all_points = collect_pareto_points(source_data["shortening_files"])

    metrics = source_data["gepa_metrics"] or {}
    original_proxy = metrics.get("proxy_reward")

    # Select shortest pareto point with test_score >= 90% of original
    shortened = None
    if original_proxy is not None and all_points:
        shortened = select_shortest_good_enough(all_points, original_proxy)

    info = source_data["gepa_info"]
    hint_type = source_data.get("hint_type")

    entry = {
        "env": env,
        "env_display": ENV_DISPLAY_NAMES[env],
        "hint_type": hint_type,
        "hint_type_display": HINT_DISPLAY_NAMES.get(hint_type) if hint_type else None,

        # GEPA method info
        "optimizer_llm": format_prompter_name(info["prompter_name"]),
        "teacher": info["teacher"],
        "suggest_hack": info["suggest_hack"],
        "gepa_method": format_gepa_method(info),

        # Original prompt
        "original_prompt": source_data["initial_prompt"],
        "original_prompt_path": source_data["initial_prompt_path"],
        "original_length": source_data["initial_length"],
        "original_proxy_reward": metrics.get("proxy_reward"),
        "original_true_reward": metrics.get("true_reward"),
        "original_hacking_rate": metrics.get("hacking_rate"),

        # Shortened prompt
        "shortened_prompt": shortened["instructions"] if shortened else None,
        "shortened_length": shortened["prompt_length"] if shortened else None,
        "shortened_test_score": shortened["test_score"] if shortened else None,
        "shortening_llm": format_prompter_name(shortened["shortening_llm"]) if shortened else None,

        # Pareto info
        "num_candidate_points": len(all_points),
        "num_shortening_runs": len(source_data["shortening_files"]),
    }

    return entry


def select_sanitization_prompts(all_sources: dict[str, dict[str, dict]],
                                main_paths: set[str]) -> list[dict]:
    """Select 3 sanitization prompts (different from main ones).

    One each from MCQ, psychosis, wordchain.
    """
    sanitization_entries = []

    for env in ["mcq", "psychosis", "wordchain"]:
        by_source = all_sources[env]
        # Find sources NOT used in main examples
        remaining = {p: d for p, d in by_source.items() if p not in main_paths}

        if not remaining:
            # If all are used (shouldn't happen for psychosis/wordchain), pick any
            print(f"  Warning: no remaining source prompts for {env} sanitization, reusing one")
            remaining = by_source

        # Pick the one with highest proxy reward, preferring those with valid shortened version
        def sort_key(d):
            has_short = has_valid_shortened(d)
            proxy = (d["gepa_metrics"] or {}).get("proxy_reward", -1)
            return (has_short, proxy)
        best = max(remaining.values(), key=sort_key)
        sanitization_entries.append(best)

    return sanitization_entries


def run_sanitization(prompt_text: str, env: str, hint_type: str | None) -> str:
    """Run sanitization on a prompt, with caching."""
    # Create cache key from prompt content + env + hint_type
    cache_key = hashlib.sha256(
        f"{env}:{hint_type}:{prompt_text}".encode()
    ).hexdigest()[:16]

    SANITIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = SANITIZATION_CACHE_DIR / f"{cache_key}.txt"

    if cache_file.exists():
        print(f"  Using cached sanitization result: {cache_file.name}")
        return cache_file.read_text()

    print(f"  Running sanitization for {env} (hint_type={hint_type})...")
    import time
    from core.instruction_loader import sanitize_instructions
    # Convert hint_type from underscore to hyphen format for the API
    api_hint_type = hint_type_underscore_to_hyphen(hint_type) if hint_type else None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            sanitized = sanitize_instructions(prompt_text, env, hint_type=api_hint_type)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(wait)
            else:
                raise

    cache_file.write_text(sanitized)
    print(f"  Saved sanitization result to: {cache_file.name}")
    return sanitized


def main():
    print("=" * 70)
    print("Preparing prompt examples for paper appendix")
    print("=" * 70)

    # Step 1: Collect all source prompts
    print("\nStep 1: Collecting source prompts...")
    all_sources = {}
    for env in ["mcq", "psychosis", "wordchain"]:
        print(f"\n  {env}:")
        all_sources[env] = collect_all_source_prompts(env)
        print(f"  Found {len(all_sources[env])} unique source prompts")

    # Step 2: Select main prompts (8 total)
    print("\nStep 2: Selecting 8 main prompts...")
    main_entries = []
    main_paths = set()
    for env in ["mcq", "psychosis", "wordchain"]:
        selected = select_main_prompts(env, all_sources[env])
        for s in selected:
            entry = build_prompt_entry(s, env)
            main_entries.append(entry)
            main_paths.add(s["initial_prompt_path"])
            ht_str = f" ({entry['hint_type_display']})" if entry["hint_type_display"] else ""
            pr = entry["original_proxy_reward"]
            pr_str = f"{pr:.3f}" if pr is not None else "N/A"
            short_len = entry["shortened_length"]
            short_str = f"{short_len}" if short_len is not None else "N/A"
            print(f"  {entry['env_display']}{ht_str}: proxy={pr_str}, "
                  f"orig_len={entry['original_length']}, short_len={short_str}")

    print(f"\n  Total main entries: {len(main_entries)}")

    # Step 3: Select sanitization prompts (3 total, different sources)
    print("\nStep 3: Selecting 3 sanitization prompts...")
    san_source_data = select_sanitization_prompts(all_sources, main_paths)
    san_entries = []
    for source_data in san_source_data:
        env = source_data["gepa_info"]["env"]
        entry = build_prompt_entry(source_data, env)

        # Run sanitization on the shortened prompt
        if entry["shortened_prompt"]:
            hint_type = entry["hint_type"]
            sanitized = run_sanitization(entry["shortened_prompt"], env, hint_type)
            entry["sanitized_prompt"] = sanitized
            entry["sanitized_length"] = len(sanitized)
        else:
            print(f"  Warning: no shortened prompt for {env} sanitization, using original")
            hint_type = entry["hint_type"]
            sanitized = run_sanitization(entry["original_prompt"], env, hint_type)
            entry["sanitized_prompt"] = sanitized
            entry["sanitized_length"] = len(sanitized)

        san_entries.append(entry)
        ht_str = f" ({entry['hint_type_display']})" if entry["hint_type_display"] else ""
        print(f"  {entry['env_display']}{ht_str}: "
              f"short_len={entry.get('shortened_length', 'N/A')}, "
              f"sanitized_len={entry.get('sanitized_length', 'N/A')}")

    # Step 4: Output JSON
    print("\nStep 4: Saving output...")
    output = {
        "main_prompts": main_entries,
        "sanitization_prompts": san_entries,
    }
    output_path = PROJECT_ROOT / "scripts" / "prompt_examples_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {output_path}")

    # Step 5: Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nMain prompts ({len(main_entries)}):")
    for i, e in enumerate(main_entries):
        ht_str = f" ({e['hint_type_display']})" if e['hint_type_display'] else ""
        pr = e['original_proxy_reward']
        tr = e['original_true_reward']
        hr = e['original_hacking_rate']
        print(f"  {i+1}. {e['env_display']}{ht_str}")
        print(f"     GEPA: {e['gepa_method']}")
        print(f"     Original: {e['original_length']} chars, "
              f"proxy={f'{pr:.3f}' if pr is not None else 'N/A'}, "
              f"true={f'{tr:.3f}' if tr is not None else 'N/A'}, "
              f"hack={f'{hr:.3f}' if hr is not None else 'N/A'}")
        if e['shortened_prompt']:
            print(f"     Shortened: {e['shortened_length']} chars, "
                  f"test_score={e['shortened_test_score']:.3f}, "
                  f"LLM={e['shortening_llm']}")
        print(f"     Source: {e['original_prompt_path']}")

    print(f"\nSanitization prompts ({len(san_entries)}):")
    for i, e in enumerate(san_entries):
        ht_str = f" ({e['hint_type_display']})" if e['hint_type_display'] else ""
        print(f"  {i+1}. {e['env_display']}{ht_str}")
        print(f"     GEPA: {e['gepa_method']}")
        if e.get('shortened_prompt'):
            print(f"     Shortened: {e['shortened_length']} chars")
        if e.get('sanitized_prompt'):
            print(f"     Sanitized: {e['sanitized_length']} chars")
        print(f"     Source: {e['original_prompt_path']}")


if __name__ == "__main__":
    main()
