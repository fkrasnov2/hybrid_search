FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
WORKDIR /app
RUN pip install --no-cache-dir pdm
COPY . .
RUN pdm install
RUN pdm run scripts/train_reranker.py
EXPOSE 8000
CMD ["pdm", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
