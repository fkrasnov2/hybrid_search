	
run:
	docker compose up --build -d
	docker compose logs elasticsearch
	docker compose exec app python scripts/create_es_index.py
	docker compose exec app python scripts/train_reranker.py

push:
	docker build -t eob_search-app .
	docker login
	docker tag eob_search-app:latest docker.io/fkrasnov/hybrid_search-app:latest
	docker push docker.io/fkrasnov/hybrid_search-app:latest
