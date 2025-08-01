# DataLab Playground

A simple Docker-based environment for exploring data analytics and AI tools. Includes basic data processing, storage, and LLM capabilities - all containerized for easy experimentation.

## Architecture Overview

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Jupyter   │  │   Phoenix   │  │   Ollama    │  │    Trino    │
│  Notebook   │  │AI Observ.   │  │ LLM Server  │  │   Engine    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │               │
       └────────────────┼────────────────┼───────────────┘
                        │                │
              ┌─────────────┐    ┌─────────────┐
              │    Spark    │    │    Hive     │
              │   Cluster   │    │ Metastore   │
              └─────────────┘    └─────────────┘
                        │                │
                        └────────────────┘
                                │
                      ┌─────────────┐    ┌─────────────┐
                      │   MinIO     │    │   Qdrant    │
                      │  (S3 API)   │    │ Vector DB   │
                      └─────────────┘    └─────────────┘
```


### Data Processing & Storage
- **MinIO**: S3-compatible storage for files
- **Apache Spark**: Basic data processing capabilities  
- **Hive Metastore**: Simple metadata management
- **Trino**: SQL query interface

### AI & ML Tools
- **Ollama**: Local LLM server (gemma3:4b model)
- **Phoenix**: Basic AI monitoring 
- **Qdrant**: Vector database for AI experiments
- **Jupyter**: Notebook environment with common libraries

### Infrastructure
- **PostgreSQL**: Database backend 
- **NVIDIA Docker**: GPU support for AI tools


## 🚀 Quick Start

### Prerequisites
- Docker with Docker Compose
- **NVIDIA GPU (Required)**:
  - **NVIDIA GPU with 4GB+ VRAM** for gemma3:4b (current default)
  - **Lower VRAM options available**: gemma3:1b
  - NVIDIA Container Runtime pre-configured for Docker GPU access
  - Platform is optimized for GPU acceleration and requires NVIDIA hardware
- **Minimum System Requirements**:
  - 8GB+ RAM recommended (12GB+ for optimal performance)
  - 50GB+ free disk space for models and data

### One-Command Setup
```bash
# This script will:
# 1. Build all custom Docker images (if not already built)
# 2. Start all services
# 3. Setup MinIO storage
# 4. Pull the gemma3:4b LLM model
./start-platform.sh
```

### Manual Setup (Alternative)
```bash
# 1. Build custom Docker images
docker build -t datalab-playground/jupyter ./jupyter
docker build -t datalab-playground/spark ./spark
docker build -t datalab-playground/trino ./trino
docker build -t datalab-playground/hive-metastore ./hive-metastore

# 2. Start all services
docker-compose up -d

# 3. Pull LLM model
docker exec ollama ollama pull gemma3:4b
```

## 🎯 Getting Started

Simple steps to explore the tools:

1. **Start**: Run `./start-platform.sh` 
2. **Open Jupyter**: Go to http://localhost:8888 (password: 123456)
3. **Try the demos**: Open `data_lab_playground.ipynb` for basic examples
4. **Experiment with RAG**: Try `rag_vector_demo.ipynb` for vector database examples
5. **Explore UIs**: Check out Qdrant dashboard, Phoenix monitoring, etc.

## 🌐 Service Access Points

| Service | URL | Credentials | Description |
|---------|-----|-------------|-------------|
| **Jupyter Notebook** | http://localhost:8888 | password: 123456 | Interactive AI/ML environment |
| **Phoenix AI Observability** | http://localhost:6006 | None | AI model monitoring & traces |
| **Ollama LLM API** | http://localhost:11434 | None | Local LLM inference endpoint |
| **Qdrant Vector Database** | http://localhost:6333 | None | Vector storage & similarity search |
| **Qdrant Web Dashboard** | http://localhost:6333/dashboard | None | Vector database management UI |
| **Trino Web UI** | http://localhost:8080 | None | SQL query interface |
| **Spark Master UI** | http://localhost:8081 | None | Spark cluster monitoring |
| **MinIO Console** | http://localhost:9001 | minioadmin/minioadmin123 | S3 storage management |

## 🤖 AI Tools

### LLM Server (Ollama)
- **Default Model**: gemma3:4b (~4GB VRAM)
- **Any Ollama Model**: Different sizes available for various GPU capabilities from [Ollama Library](https://ollama.com/library)
- **Requirements**: NVIDIA GPU with Docker runtime
- **API**: `http://localhost:11434`

### Vector Database (Qdrant)  
- **Purpose**: Store embeddings for RAG experiments
- **Web UI**: `http://localhost:6333/dashboard`
- **API**: `http://localhost:6333`

### Monitoring (Phoenix)
- **Purpose**: Basic AI operation tracing
- **Web UI**: `http://localhost:6006`

### Notebook Environment (Jupyter)
- **GenAI Environment**: Pre-installed packages
- **Core Libraries**: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn
- **AI/ML Stack**: LangChain ecosystem, transformers, torch, sentence-transformers
- **Vector & Database**: qdrant-client, trino, sqlalchemy, boto3, s3fs
- **Document Processing**: pypdf2, python-docx, beautifulsoup4, tiktoken
- **Observability**: arize-phoenix, opentelemetry, openinference instrumentation
- **Development Tools**: ipywidgets, tqdm, rich, typer
- **Default Kernel**: GenAI Analytics (Python 3.12)
- **Access**: `http://localhost:8888` (password: 123456)

## 📊 Usage Examples

### Using Ollama LLM in Jupyter
```python
import requests
import json

# Chat with the local LLM
def chat_with_ollama(prompt, model="gemma3:4b"):
    response = requests.post('http://ollama:11434/api/generate',
                           json={
                               "model": model,
                               "prompt": prompt,
                               "stream": False
                           })
    return response.json()['response']

# Example usage
result = chat_with_ollama("Explain data analytics in simple terms")
print(result)
```

### Building RAG Systems with Qdrant
```python
from qdrant_client import QdrantClient
import ollama

# Connect to vector database
qdrant = QdrantClient(host="qdrant", port=6333)

# Generate embeddings using Ollama
def get_embedding(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

# Store document chunks in vector database
def store_document(text_chunks, collection_name="knowledge_base"):
    for i, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk)
        qdrant.upsert(
            collection_name=collection_name,
            points=[{
                "id": i,
                "vector": embedding,
                "payload": {"text": chunk}
            }]
        )

# Retrieve relevant context for questions
def rag_query(question, collection_name="knowledge_base"):
    question_embedding = get_embedding(question)
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=3
    )
    context = "\n".join([hit.payload["text"] for hit in results])
    
    # Use context with LLM
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    return chat_with_ollama(prompt)
```

### Phoenix AI Observability
```python
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

# Configure Phoenix tracing (in GenAI DEV kernel)
tracer_provider = register(
    project_name="data-analytics",
    endpoint="http://phoenix:4317",
    auto_instrument=True,
)

```

### Spark with S3 Integration
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DataLab-Playground") \
    .master("spark://spark-master:7077") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123") \
    .getOrCreate()

# Process data and prepare for AI workloads
df = spark.read.parquet("s3a://warehouse/data/")
df.write.mode("overwrite").parquet("s3a://warehouse/processed/ai_training_data")
```

## 🛠️ Basic Configuration

### Default Settings
- **MinIO**: minioadmin/minioadmin123
- **Spark**: spark://spark-master:7077  
- **Ollama Models**: Stored in persistent volume

## 🔧 Platform Management

### Service Health Monitoring
```bash
# Check all services status
docker-compose ps

# View specific service logs
docker logs phoenix
docker logs ollama

# Restart services
docker-compose restart ollama phoenix
```

### Available Models
```bash
# List available Ollama models
docker exec ollama ollama list

# Pull additional LLM models
docker exec ollama ollama pull llama3.2

# Pull additional embedding models
docker exec ollama ollama pull all-MiniLM-L6-v2
```

## 🚦 Startup Order

Services start automatically in the right order:
1. Storage & databases (PostgreSQL, MinIO)
2. Data processing (Spark, Trino, Hive)  
3. AI services (Ollama, Phoenix)
4. Jupyter notebooks

## 🐛 Common Issues

### GPU Problems
```bash
# Check if GPU is detected
docker exec ollama nvidia-smi

# Verify Docker GPU support  
docker info | grep nvidia
```

### Service Problems
```bash
# Check if services are running
docker-compose ps

# View service logs
docker logs ollama
docker logs phoenix
```


## 🤝 Contributing

Found a bug or have an idea? Feel free to:
- Open GitHub issues for problems or suggestions
- Submit pull requests for improvements  
- Add example notebooks or documentation

## Acknowledgments

**Built with AI Assistance**

This project was developed with GitHub Copilot (powered by Claude Sonnet 4), demonstrating the power of human-AI partnership in creating comprehensive data platforms.