# Makefile — Bayesian Grossman-Zhou Project
# Sukhov (2026c), Market Microstructure Research Lab

.PHONY: paper figures simulate stress clean help

# ── Default ───────────────────────────────────────────────────────────────────
all: figures simulate paper

# ── Figures ───────────────────────────────────────────────────────────────────
figures:
	@echo "Generating figures..."
	@cd code && python3 generate_figures.py
	@echo "  → figures/ ($(shell ls figures/*.pdf | wc -l) files)"

# ── Main Monte Carlo simulation (N=50,000) ────────────────────────────────────
simulate:
	@echo "Running main simulation (N=50,000)..."
	@python3 code/simulation.py
	@echo "  → done"

# ── Fast check (N=5,000) ─────────────────────────────────────────────────────
fast:
	@echo "Quick simulation check (N=5,000)..."
	@python3 code/simulation.py --fast

# ── Stress tests ─────────────────────────────────────────────────────────────
stress:
	@echo "Running stress tests (N=50,000, ~5 min)..."
	@python3 code/stress_tests.py
	@python3 code/fig7_stress_tests.py
	@echo "  → figures/fig7_stress_tests.pdf"

# ── Full reproducibility run (figures + simulate + stress + paper) ────────────
reproduce: figures simulate stress paper
	@echo "Full reproduction complete."

# ── Clean LaTeX artifacts ─────────────────────────────────────────────────────
clean:
	@cd paper && rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
	@echo "LaTeX artifacts removed."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make fast       — quick simulation check (N=5,000, ~10s)"
	@echo "  make simulate   — full simulation (N=50,000, ~3 min)"
	@echo "  make stress     — stress tests across 5 return environments"
	@echo "  make figures    — regenerate all paper figures"
	@echo "  make reproduce  — full reproducibility run (all steps)"
	@echo "  make clean      — remove LaTeX auxiliary files"
	@echo ""
