# MoSPI RAG Chatbot System

A comprehensive Retrieval-Augmented Generation (RAG) chatbot system for querying Indian government statistical data from the Ministry of Statistics and Programme Implementation (MoSPI).

## 🎯 Overview

This system consists of three main components:
1. **Web Scraper**: Collects and processes data from MoSPI website
2. **ETL Pipeline**: Processes and chunks documents for search
3. **RAG Chatbot**: Provides Q&A with cited sources using Gemini AI

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Scraper   │    │  ETL Pipeline   │    │  RAG Chatbot    │
│                 │────│                 │────│                 │
│ • crawl.py      │    │ • run.py        │    │ • api.py        │
│ • parse.py      │    │ • validate.py   │    │ • streamlit_app │
│ • models.py     │    │ • Transform     │    │ • embeddings.py │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   SQLite DB     │
                    │ • documents     │
                    │ • chunks        │
                    │ • metadata      │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Gemini API key (free at ai.google.dev)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd mospi-rag-chatbot

# Copy environment file
cp .env.example .env

# Edit .env with your Gemini API key
nano .env
```

### 2. Run the Complete System
```bash
# Build and start everything
make run

# Or manually:
docker-compose up -d
```

### 3. Access the System
- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
mospi-rag-chatbot/
├── scraper/                 # Web scraping components
│   ├── crawl.py            # Main crawling logic
│   ├── parse.py            # PDF and content parsing
│   ├── models.py           # Data models
│   └── config.py           # Configuration
├── pipeline/               # ETL pipeline
│   ├── run.py              # Main pipeline runner
│   └── validate.py         # Data validation
├── rag/                    # RAG chatbot system
│   ├── api.py              # FastAPI backend
│   ├── streamlit_app.py    # Streamlit UI
│   ├── embeddings.py       # Vector search
│   ├── llm_client.py       # Gemini AI client
│   └── requirements.txt    # Python dependencies
├── data/                   # Data storage
│   ├── mospi.db           # SQLite database
│   ├── raw/               # Raw scraped files
│   └── processed/         # Processed embeddings
├── docker-compose.yml      # Docker orchestration
├── Makefile               # Automation commands
└── README.md              # This file
```

## 🔧 Development Setup

### Local Development (without Docker)
```bash
# Install dependencies
make install

# Setup embeddings (after scraping)
make setup

# Run development servers
make dev-run
```

### Available Commands
```bash
make help          # Show all available commands
make install       # Install Python dependencies
make setup         # Setup embeddings
make build         # Build Docker containers
make run           # Start complete system
make stop          # Stop all containers
make clean         # Clean up containers
make test          # Run tests
make health        # Check system health
```

## 💬 Using the Chatbot

### Example Queries
- "What is India's current GDP growth rate?"
- "Tell me about population demographics in rural areas"
- "What are the latest employment statistics?"
- "Show me data on industrial production"
- "What is the inflation rate trend?"

### Features
- **Real-time Search**: Vector-based similarity search through documents
- **Source Citations**: All answers include clickable source links
- **Adjustable Parameters**: Control search depth (k) and creativity (temperature)
- **Responsive UI**: Clean, modern interface with real-time status
- **Health Monitoring**: Built-in health checks and monitoring

## 🔍 API Endpoints

### Chat Endpoint
```bash
POST /chat
{
    "query": "What is GDP?",
    "k": 5,
    "temperature": 0.7
}
```

### Health Check
```bash
GET /health
{
    "status": "healthy",
    "index_loaded": true,
    "total_chunks": 1250
}
```

### Search (Debug)
```bash
GET /search/gdp?k=5
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test file
cd rag && python -m pytest test_rag.py -v

# Health check
make health
```

## 🐳 Docker Configuration

### Services
- **rag-api**: FastAPI backend service
- **rag-ui**: Streamlit frontend service  
- **setup-embeddings**: One-time embedding creation

### Volumes
- `./data`: Persistent data storage
- `./rag`: Application code

### Networks
- **rag-network**: Internal communication between services

## 📊 Data Pipeline

### 1. Scraping Stage
```bash
cd scraper
python crawl.py  # Scrape MoSPI website
```

### 2. ETL Stage  
```bash
cd pipeline
python run.py    # Process and chunk documents
```

### 3. RAG Stage
```bash
cd rag
python setup_embeddings.py  # Create search index
python api.py               # Start API server
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
GEMINI_API_KEY=your_api_key_here
DB_PATH=data/mospi.db
API_PORT=8000
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
```

### Key Settings
- **k**: Number of document chunks to retrieve (1-10)
- **temperature**: LLM creativity level (0.0-1.0)
- **embedding_model**: SentenceTransformer model name
- **chunk_size**: Document chunk size in tokens

## 🚨 Troubleshooting

### Common Issues

1. **API not responding**
   ```bash
   docker-compose logs rag-api
   make health
   ```

2. **No search results**
   ```bash
   # Recreate embeddings
   docker-compose run setup-embeddings
   ```

3. **Gemini API errors**
   ```bash
   # Check API key in .env
   curl -H "Authorization: Bearer $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1/models
   ```

4. **Database issues**
   ```bash
   # Check database
   sqlite3 data/mospi.db ".tables"
   ```

### Performance Optimization
- Use SSD storage for faster embeddings lookup
- Increase Docker memory for large document sets
- Adjust chunk size based on document types
- Use GPU-enabled models for faster embedding generation

## 📈 Monitoring

### Health Checks
- API: `GET /health`  
- UI: `GET /_stcore/health`
- Database connectivity
- Embedding index status

### Metrics
- Response time
- Search accuracy
- Source relevance scores
- User query patterns

## 🛠️ Extending the System

### Adding New Data Sources
1. Extend scraper in `scraper/crawl.py`
2. Update database schema in `scraper/models.py`
3. Rebuild embeddings with `make setup`

### Custom LLM Integration
1. Implement new client in `rag/llm_client.py`
2. Update API endpoints in `rag/api.py`
3. Modify UI controls in `rag/streamlit_app.py`

### Enhanced Search Features
1. Modify embedding logic in `rag/embeddings.py`
2. Add filters and facets
3. Implement semantic clustering

## 📄 License

This project is open source. Please check individual component licenses for specific terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review Docker logs: `docker-compose logs -f`
- Run health checks: `make health`
- Open an issue on GitHub