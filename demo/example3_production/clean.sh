#!/usr/bin/env bash
# Clean example3_production to a fresh state.

set -euo pipefail

# Resolve the directory of this script (works from any CWD and with symlinks)
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SCRIPT_SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" >/dev/null 2>&1 && pwd)"
  SCRIPT_SOURCE="$(readlink "$SCRIPT_SOURCE")"
  [[ "$SCRIPT_SOURCE" != /* ]] && SCRIPT_SOURCE="$DIR/$SCRIPT_SOURCE"
done
EXAMPLE_DIR="$(cd -P "$(dirname "$SCRIPT_SOURCE")" >/dev/null 2>&1 && pwd)"

echo "Cleaning example at: $EXAMPLE_DIR"

# Remove experiments directory and generated SLURM template
rm -rf -- "$EXAMPLE_DIR/experiments"
rm -f -- "$EXAMPLE_DIR/_slurm_template.sbatch"

# Remove Python bytecode caches within the example directory
if command -v find >/dev/null 2>&1; then
  find "$EXAMPLE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
fi

echo "Done. The example folder is reset."
