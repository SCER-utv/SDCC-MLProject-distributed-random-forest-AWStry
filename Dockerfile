# Usa Python 3.9 Slim
FROM python:3.9-slim

# Imposta la directory di lavoro
WORKDIR /app

# 1. Installa dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copia e installa i requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copia i file singoli dalla ROOT
COPY shard_dataset.py .
COPY rf_service.proto .

# 4. Copia le cartelle intere
COPY config/ ./config/
COPY src/ ./src/

# 5. Crea la struttura dati
RUN mkdir -p \
    data/taxi/processed data/taxi/shards \
    data/higgs/processed data/higgs/shards \
    data/temp \
    models/checkpoints

# 6. FIX DEFINITIVO gRPC
# A. Spostiamo il proto nella sua cartella finale
RUN mkdir -p src/network/proto && \
    mv rf_service.proto src/network/proto/

# B. Creiamo i file _init_.py per assicurare che Python tratti le cartelle come pacchetti
RUN touch src/_init_.py && \
    touch src/network/_init_.py && \
    touch src/network/proto/_init_.py

# C. Compilazione "Root-Relative" (LA CHIAVE DEL SUCCESSO)
# Usando -I . (invece che la sottocartella), protoc genererà gli import completi:
# "from src.network.proto import rf_service_pb2" invece di "import rf_service_pb2"
RUN python -m grpc_tools.protoc \
    -I . \
    --python_out=. \
    --grpc_python_out=. \
    src/network/proto/rf_service.proto

# 7. Variabili d'Ambiente
ENV PYTHONPATH=/app
ENV ROLE=worker
ENV WORKER_PORT=50051
ENV PYTHONUNBUFFERED=1
ENV S3_BUCKET_NAME="random-forest-bucket-bkt"
ENV AWS_DEFAULT_REGION="us-east-1"

# 8. Script di avvio
CMD if [ "$ROLE" = "master" ]; \
    then \
      python src/master.py; \
    else \
      python src/worker.py $WORKER_PORT; \
    fi