"""Generate LaTeX content for appendices B and C from prompt_examples_output.json."""

import json
import sys
from pathlib import Path


def escape_for_lstlisting(text):
    """Escape text for use inside lstlisting environment."""
    # lstlisting handles most special chars, but we need to handle ■
    # which we'll use the literate option for
    return text


def format_pct(val):
    if val is None:
        return "---"
    return f"{val*100:.0f}\\%"


def format_score(val):
    if val is None:
        return "---"
    return f"{val:.2f}"


def needs_black_square_literate(text):
    """Check if text contains ■ character."""
    return "■" in text


def lstlisting_options(text):
    """Generate lstlisting options, adding literate for ■ if needed."""
    base = "basicstyle=\\ttfamily\\small, breaklines=true, breakatwhitespace=true"
    if needs_black_square_literate(text):
        base += ', literate={■}{{\\rule{0.8ex}{0.8ex}}}1'
    return base


def prompt_box(text, meta_lines, title=None):
    """Generate a tcolorbox with metadata and lstlisting for a prompt.

    meta_lines: list of (label, value) tuples shown as small italic text
                inside the box, separated from the prompt by a thin rule.
    title: shown in the tcolorbox title bar at the top of the box.
    """
    opts = lstlisting_options(text)
    lines = []
    tcb_opts = "colback=gray!10, colframe=black!50, breakable"
    if title:
        tcb_opts += f", title={{{title}}}"
    lines.append(f"\\begin{{tcolorbox}}[{tcb_opts}]")
    if meta_lines:
        # Metadata as small italic lines
        lines.append("{\\small\\itshape")
        for label, value in meta_lines:
            lines.append(f"\\textbf{{{label}}} {value}\\\\")
        lines.append("}%")
        lines.append("\\vspace{2pt}\\hrule\\vspace{4pt}")
    lines.append(f"\\begin{{lstlisting}}[{opts}]")
    lines.append(text)
    lines.append("\\end{lstlisting}")
    lines.append("\\end{tcolorbox}")
    return "\n".join(lines)


def generate_appendix_b(data):
    """Generate Appendix B: Optimized System Prompt Examples."""
    lines = []
    lines.append("%% ====================================================================")
    lines.append("%% B. Optimized System Prompt Examples")
    lines.append("%% ====================================================================")
    lines.append("\\section{Optimized System Prompt Examples}")
    lines.append("\\label{app:prompt-examples}")
    lines.append("")
    lines.append(
        "This appendix shows the highest-reward GEPA-optimized prompt for each "
        "environment and hint type (8 total). For each prompt, we show the original "
        "GEPA-optimized version alongside a shortened version selected from the Pareto "
        "frontier of length vs.\\ reward (see \\cref{app:shortening} for the shortening "
        "method). The shortened version is the shortest candidate that retains at least "
        "90\\% of the original test proxy reward."
    )

    for e in data["main_prompts"]:
        ht = e.get("hint_type_display", "")
        env = e["env_display"]
        if ht:
            title = f"{env} ({ht})"
        else:
            title = env

        # Label
        if e["env"] == "mcq":
            label = f"app:promptex-mcq-{e['hint_type'].replace('_', '-')}"
        elif e["env"] == "psychosis":
            label = "app:promptex-syco"
        else:
            label = "app:promptex-wc"

        lines.append("")
        lines.append(f"\\subsection{{{title}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("")
        lines.append(f"% Source: {e['original_prompt_path']}")
        lines.append("")

        # Original prompt
        pr = format_score(e["original_proxy_reward"])
        tr = format_score(e["original_true_reward"])
        hr = format_pct(e["original_hacking_rate"])
        orig_meta = [
            ("Optimizer:", e["optimizer_llm"]),
            ("Teacher:", "Yes" if e["teacher"] else "No"),
            ("Suggest hacking:", "Yes" if e["suggest_hack"] == "explicit" else e["suggest_hack"].title()),
            ("Proxy reward:", pr),
            ("True reward:", tr),
            ("Hacking rate:", hr),
            ("Length:", f"{e['original_length']} chars"),
        ]
        lines.append(prompt_box(e["original_prompt"], orig_meta, title="Original prompt"))
        lines.append("")

        # Shortened prompt
        if e["shortened_prompt"]:
            short_score = format_score(e["shortened_test_score"])
            short_meta = [
                ("Shortening LLM:", e["shortening_llm"]),
                ("Proxy reward:", short_score),
                ("Length:", f"{e['shortened_length']} chars"),
            ]
            lines.append(prompt_box(e["shortened_prompt"], short_meta, title="Shortened prompt"))
        lines.append("")

    return "\n".join(lines)


def generate_appendix_c(data):
    """Generate Appendix C: Sanitization Examples."""
    lines = []
    lines.append("%% ====================================================================")
    lines.append("%% C. Sanitization Examples")
    lines.append("%% ====================================================================")
    lines.append("\\section{Sanitization Examples}")
    lines.append("\\label{app:sanitization-examples}")
    lines.append("")
    lines.append(
        "This appendix shows examples of GEPA-optimized prompts before and after "
        "sanitization (see \\cref{app:sanitization} for the sanitization method). "
        "Each example uses a different source prompt than the ones shown in "
        "\\cref{app:prompt-examples}. The prompts are first shortened and then "
        "sanitized; we show the shortened (pre-sanitization) and sanitized versions."
    )

    for e in data["sanitization_prompts"]:
        ht = e.get("hint_type_display", "")
        env = e["env_display"]
        if ht:
            title = f"{env} ({ht})"
        else:
            title = env

        # Label
        if e["env"] == "mcq":
            ht_slug = e["hint_type"].replace("_", "-")
            label = f"app:sanex-mcq-{ht_slug}"
        elif e["env"] == "psychosis":
            label = "app:sanex-syco"
        else:
            label = "app:sanex-wc"

        lines.append("")
        lines.append(f"\\subsection{{{title}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("")
        lines.append(f"% Source: {e['original_prompt_path']}")
        lines.append("")

        # Before sanitization (shortened prompt)
        before_text = e.get("shortened_prompt") or e["original_prompt"]
        before_len = e.get("shortened_length") or e["original_length"]
        pr = format_score(e.get("shortened_test_score") or e.get("original_proxy_reward"))
        before_meta = [
            ("Optimizer:", e["optimizer_llm"]),
            ("Teacher:", "Yes" if e["teacher"] else "No"),
            ("Suggest hacking:", "Yes" if e["suggest_hack"] == "explicit" else e["suggest_hack"].title()),
            ("Proxy reward:", pr),
            ("Length:", f"{before_len} chars"),
        ]
        lines.append(prompt_box(before_text, before_meta, title="Before sanitization"))
        lines.append("")

        # After sanitization
        if e.get("sanitized_prompt"):
            lines.append(prompt_box(e["sanitized_prompt"], [], title="After sanitization"))
        lines.append("")

    return "\n".join(lines)


def main():
    with open(Path(__file__).parent / "prompt_examples_output.json") as f:
        data = json.load(f)

    appendix_b = generate_appendix_b(data)
    appendix_c = generate_appendix_c(data)

    output_path = Path(__file__).parent / "appendix_bc_latex.tex"
    with open(output_path, "w") as f:
        f.write(appendix_b)
        f.write("\n\n")
        f.write(appendix_c)
        f.write("\n")

    print(f"Generated LaTeX written to: {output_path}")
    print(f"Appendix B: {len(appendix_b)} chars")
    print(f"Appendix C: {len(appendix_c)} chars")


if __name__ == "__main__":
    main()
