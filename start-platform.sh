#!/bin/bash

# Parse command line arguments
FORCE_REBUILD=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild|-r)
            FORCE_REBUILD=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            HELP=true
            shift
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo "DataLab Playground Startup Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --rebuild, -r    Force rebuild all Docker images"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0               Start platform (rebuild only if needed)"
    echo "  $0 --rebuild     Force rebuild all images and start platform"
    exit 0
fi

echo "=== DataLab Playground Startup ==="
echo ""

# Function to check if images exist and are up to date
check_images() {
    echo "🔍 Checking Docker images and detecting changes..."
    images=("datalab-playground/hive-metastore" "datalab-playground/trino" "datalab-playground/spark" "datalab-playground/jupyter")
    services=("hive-metastore" "trino" "spark" "jupyter")
    missing_images=()
    outdated_images=()
    
    for i in "${!images[@]}"; do
        image="${images[$i]}"
        service="${services[$i]}"
        
        # Check if image exists
        if ! docker image inspect "$image" >/dev/null 2>&1; then
            missing_images+=("$image")
            echo "❌ Missing: $image"
            continue
        fi
        
        # Check if source files are newer than the image
        image_created=$(docker image inspect "$image" --format='{{.Created}}' 2>/dev/null)
        if [ -n "$image_created" ]; then
            image_timestamp=$(date -d "$image_created" +%s 2>/dev/null || echo "0")
            
            # Find the newest file in the service directory
            if [ -d "./$service" ]; then
                newest_file_timestamp=$(find "./$service" -type f -exec stat --format='%Y' {} \; 2>/dev/null | sort -nr | head -1)
                if [ -n "$newest_file_timestamp" ] && [ "$newest_file_timestamp" -gt "$image_timestamp" ]; then
                    outdated_images+=("$image")
                    echo "📅 Outdated: $image (source files changed)"
                fi
            fi
        fi
    done
    
    if [ ${#missing_images[@]} -gt 0 ] || [ ${#outdated_images[@]} -gt 0 ]; then
        if [ ${#missing_images[@]} -gt 0 ]; then
            echo "❌ Missing images: ${missing_images[*]}"
        fi
        if [ ${#outdated_images[@]} -gt 0 ]; then
            echo "🔄 Outdated images: ${outdated_images[*]}"
        fi
        return 1
    else
        echo "✅ All images are up to date"
        return 0
    fi
}

# Function to build specific image with progress
build_image() {
    local service=$1
    local image_name=$2
    
    echo ""
    echo "🔨 Building $service image..."
    echo "   Source: ./$service/"
    echo "   Target: $image_name"
    
    if docker build -t "$image_name" "./$service"; then
        echo "✅ $service image built successfully"
        return 0
    else
        echo "❌ Failed to build $service image"
        return 1
    fi
}

# Build images if needed
if [ "$FORCE_REBUILD" = true ]; then
    echo "🔄 Force rebuild requested - rebuilding all images..."
    echo ""
    echo "🔨 Building/rebuilding Docker images..."
    
    # Build images in dependency order
    build_image "hive-metastore" "datalab-playground/hive-metastore" || exit 1
    build_image "trino" "datalab-playground/trino" || exit 1
    build_image "spark" "datalab-playground/spark" || exit 1
    build_image "jupyter" "datalab-playground/jupyter" || exit 1
    
    echo ""
    echo "✅ All images rebuilt successfully"
elif check_images; then
    echo "📦 Using existing Docker images"
else
    echo ""
    echo "🔨 Building/rebuilding Docker images..."
    
    # Build images in dependency order
    build_image "hive-metastore" "datalab-playground/hive-metastore" || exit 1
    build_image "trino" "datalab-playground/trino" || exit 1
    build_image "spark" "datalab-playground/spark" || exit 1
    build_image "jupyter" "datalab-playground/jupyter" || exit 1
    
    echo ""
    echo "✅ All images built successfully"
fi

echo ""
echo "🚀 Starting all platform services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to initialize..."
sleep 15

# Setup MinIO bucket
echo "📦 Setting up MinIO storage..."
if docker exec minio mc alias set local http://localhost:9000 minioadmin minioadmin123 2>/dev/null; then
    docker exec minio mc mb local/warehouse --ignore-existing 2>/dev/null || true
    echo "✅ MinIO warehouse bucket ready"
else
    echo "⚠️  MinIO setup will be done automatically"
fi

# Handle Ollama models
echo ""
echo "🤖 Setting up Ollama LLM..."
sleep 5

# Check if gemma3:4b model is already available
if docker exec ollama ollama list 2>/dev/null | grep -q "gemma3:4b"; then
    echo "✅ gemma3:4b model already available"
else
    echo "📥 Pulling gemma3:4b model (this may take a few minutes)..."
    if docker exec ollama ollama pull gemma3:4b; then
        echo "✅ gemma3:4b model pulled successfully"
    else
        echo "⚠️  Model pull will continue in background"
    fi
fi

# Check if mxbai-embed-large model is already available
if docker exec ollama ollama list 2>/dev/null | grep -q "mxbai-embed-large"; then
    echo "✅ mxbai-embed-large embedding model already available"
else
    echo "📥 Pulling mxbai-embed-large embedding model..."
    if docker exec ollama ollama pull mxbai-embed-large; then
        echo "✅ mxbai-embed-large embedding model pulled successfully"
    else
        echo "⚠️  Embedding model pull will continue in background"
    fi
fi

echo ""
echo "🎉 DataLab Playground is Ready!"
echo ""
echo "🌐 Access Points:"
echo "  📊 Trino Query Engine:        http://localhost:8080"
echo "  📝 Jupyter Notebooks:         http://localhost:8888 (password: 123456)"
echo "  🔍 Phoenix AI Observability:  http://localhost:6006"
echo "  🤖 Ollama LLM API:            http://localhost:11434"
echo "  �️  Qdrant Vector Database:   http://localhost:6333"
echo "  �💾 MinIO Console:             http://localhost:9001 (minioadmin/minioadmin123)"
echo "  ⚡ Spark Master UI:           http://localhost:8081"
echo "  ⚡ Spark Worker UI:           http://localhost:8082"
echo ""
echo "📚 Getting Started:"
echo "  1. Open Jupyter: http://localhost:8888"
echo "  2. Run comprehensive demo: notebooks/data_lab_playground.ipynb"
echo "  3. Try RAG with vectors: notebooks/rag_demo.ipynb"
echo "  4. For advanced AI: Switch to 'GenAI DEV' kernel in notebook"
echo "  5. Monitor AI workloads: http://localhost:6006"
echo "  6. Explore vector database: http://localhost:6333/dashboard"
echo ""
echo "🔧 Script Options:"
echo "  • Automatic change detection: Images rebuilt only when source files change"
echo "  • Force rebuild: $0 --rebuild"
echo "  • Help: $0 --help"
echo ""
echo "🤖 Available Ollama Models:"
docker exec ollama ollama list 2>/dev/null || echo "  (Ollama still starting up...)"
