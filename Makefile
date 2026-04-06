.PHONY: collect annotate eda preprocess process evaluate db-up db-down

collect:
	python -m pipeline.01_extraction

annotate:
	streamlit run pipeline/02_annotation/app.py

eda:
	streamlit run pipeline/03_eda/dashboard.py

preprocess:
	python -m pipeline.04_preprocessing

process:
	python -m pipeline.05_processing

evaluate:
	python -m pipeline.06_evaluation

db-up:
	docker-compose up -d

db-down:
	docker-compose down
