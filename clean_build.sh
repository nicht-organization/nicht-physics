#!/usr/bin/env bash
# ==============================================================================
# Nicht-Physics — Clean & Rebuild Script (Dev Container Friendly)
# ==============================================================================
set -euo pipefail

# 1. Always anchor execution to the script's root folder (prevents directory lock)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=== 1. Hard Cleanup of Build Artifacts ==="
rm -f latex/*.aux latex/*.log latex/*.out latex/*.toc latex/*.synctex.gz \
      latex/*.fls latex/*.fdb_latexmk latex/*.bbl latex/*.bcf latex/*.blg 2>/dev/null || true

# 2. Check for LaTeX compiler in container environment
if ! command -v pdflatex &>/dev/null; then
    echo "=========================================================="
    echo "--> WARNING: 'pdflatex' command not found!"
    echo "    To install LaTeX in this container, run:"
    echo "    sudo apt-get update && sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended"
    echo "=========================================================="
    exit 1
fi

TEX_DIR="latex"
if [ ! -d "$TEX_DIR" ]; then
    echo "--> ERROR: '$TEX_DIR' directory not found in $(pwd)!"
    exit 1
fi

echo "=== 2. Compiling Article Manuscripts ==="

# 3. Subshell isolation keeps the parent process in ROOT_DIR for repeat runs
(
    cd "$TEX_DIR"

    # Find all .tex files dynamically
    shopt -s nullglob
    TEX_FILES=(*.tex)
    
    if [ ${#TEX_FILES[@]} -eq 0 ]; then
        echo "--> ERROR: No .tex files found in $TEX_DIR!"
        exit 1
    fi

    for file in "${TEX_FILES[@]}"; do
        BASE="${file%.tex}"
        echo "----------------------------------------"
        echo "Processing target: $file"

        echo "--> Pass 1/3: Initial layout pass..."
        pdflatex -interaction=nonstopmode "$file" > /dev/null 2>&1 || true

        echo "--> Pass 2/3: Compiling BibTeX references..."
        bibtex "$BASE" > /dev/null 2>&1 || true

        echo "--> Pass 3/3: Resolving cross-references and citations..."
        pdflatex -interaction=nonstopmode "$file" > /dev/null 2>&1 || true
        pdflatex -interaction=nonstopmode "$file" > /dev/null 2>&1 || true

        if [ -f "${BASE}.pdf" ]; then
            echo "========================================"
            echo "--> SUCCESS: ${BASE}.pdf compiled cleanly!"
            echo "========================================"
        else
            echo "--> ERROR: ${BASE}.pdf failed to compile!"
            exit 1
        fi
    done
)

echo "========================================"
echo "Build process complete!"