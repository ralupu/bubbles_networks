.PHONY: paper pipeline

paper:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_paper.ps1

pipeline:
	.\.venv\Scripts\python.exe -m pip install -e .
	.\.venv\Scripts\python.exe scripts/run_pipeline.py --skip-centrality --skip-temporal

