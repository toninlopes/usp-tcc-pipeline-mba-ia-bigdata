.PHONY: collect annotate eda preprocess process evaluate db-up db-down

collect:
	PYTHONPATH=. python -m app.core.extraction

annotate:
	PYTHONPATH=. streamlit run app/dashboard/app.py

eda:
	PYTHONPATH=. streamlit run app/dashboard/app.py

preprocess:
	PYTHONPATH=. streamlit run app/dashboard/app.py

process:
	PYTHONPATH=. streamlit run app/dashboard/app.py

evaluate:
	PYTHONPATH=. python -m app.core.evaluation

db-up:
	docker-compose up -d

db-down:
	docker-compose down