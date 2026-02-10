#!/bin/bash
# Copy plots from plots/ to overleaf/plots/ with renaming

# =============================================================================
# MAPPING CONFIGURATION
# Format: "source_name:dest_name" (without .pdf extension)
# =============================================================================
MAPPINGS=(
    "baseline_vs_final_bar_2:baseline-vs-final"
    "optimizer_progression_1:optimizer-progression"
    "optimizer_progression_2:optimizer-progression-by-prompter"
    "optimizer_progression_3:optimizer-progression-mcq-by-prompter"
    "recall_by_hacking_bins_1:recall-by-hacking-bins-by-env"
    "recall_by_hacking_bins_2:recall-by-hacking-bins-overall"
    "sanitization_1:sanitization"
    "shortening_1:shortening"
    "shortening_2:shortening-many"
)
# =============================================================================

SRC_DIR="plots"
DEST_DIR="overleaf/plots"

mkdir -p "$DEST_DIR"

for mapping in "${MAPPINGS[@]}"; do
    src_name="${mapping%%:*}"
    dest_name="${mapping##*:}"

    src_path="$SRC_DIR/${src_name}.pdf"
    dest_path="$DEST_DIR/${dest_name}.pdf"

    if [[ -f "$src_path" ]]; then
        cp "$src_path" "$dest_path"
        echo "Copied: $src_path -> $dest_path"
    else
        echo "Warning: $src_path not found, skipping"
    fi
done
