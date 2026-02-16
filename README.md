# News Fact-Checking System

A comprehensive fact-checking system for multilingual news claims using **LlamaIndex Citation Query Engine**, **Groq API** for fast inference, **BAAI/bge-m3** embeddings, and **Chroma** vector storage. Supports **English** and **Vietnamese** news fact-checking.

This document combines documentation from the entire project, including setup guides, architecture details, and usage references.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Interactive Mode](#interactive-mode)
- [Configuration](#configuration)
- [Technical Architecture](#technical-architecture)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Reference](#reference)

## Features

**Key Capabilities:**
- **Citation-based fact-checking** - Track sources and evidence for claims
- **Groq API Integration** - Ultra-fast inference with multiple model options (Version 2.0)
- **Multilingual Support** - English and Vietnamese fact-checking
- **Vietnamese News** - Native support for Vietnamese language claims
- **Multi-model Embeddings** - BAAI/bge-m3 for semantic understanding
- **Vector Storage** - Persistent Chroma database for scalable fact bases
- **Three-level Verdicts** - True, False, or Not Enough Information
- **Confidence Scoring** - Numerical confidence for each verdict
- **Batch Processing** - Check multiple claims efficiently
- **Source Tracking** - Maintain citations and evidence trails

## Project Structure

```
fact_checking/
├── fact_checker.py              Main system (core)
├── cli.py                       Command-line interface
├── config.py                    Configuration management
├── utils.py                     Export & analysis tools
├── verify_setup.py              Diagnostic tool
├── requirements.txt             Dependencies
├── .env.example                 Config template
├── .gitignore                   Git configuration
│
├── README.md                    Full documentation
└── ...
```

2. **Create environment file:**
```bash
copy .env.example .env
```

3. **Add your Groq API key:**
Open `.env` and update:
```
GROQ_API_KEY=your_actual_groq_api_key_here
LANGUAGE=vi  # For Vietnamese, or 'en' for English
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage - English

```python
from fact_checker import NewsFactChecker

# Initialize for English
fact_checker = NewsFactChecker(language="en")

# Add English knowledge base
documents = [
    "The Earth orbits the Sun taking approximately 365.25 days.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Great Wall of China is visible from space."
]
fact_checker.add_knowledge_base(documents)

# Check a claim
result = fact_checker.check_claim("Earth takes 365 days to orbit the Sun")

print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Explanation: {result.explanation}")
```

### Basic Usage - Vietnamese

```python
from fact_checker import NewsFactChecker

# Initialize for Vietnamese
fact_checker = NewsFactChecker(language="vi")

# Add Vietnamese knowledge base
documents_vi = [
    "Thủ đô của Việt Nam là Hà Nội.",
    "Sài Gòn (TP. Hồ Chí Minh) là thành phố lớn nhất của Việt Nam.",
    "Việt Nam nằm ở Đông Nam Á.",
]
fact_checker.add_knowledge_base(documents_vi)

# Check a Vietnamese claim
result = fact_checker.check_claim("Hà Nội là thủ đô của Việt Nam")

print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Explanation: {result.explanation}")
```

### Batch Processing

```python
# Check multiple claims (English or Vietnamese)
claims = [
    "Paris is the capital of France",
    "Python is a programming language",
    "The Earth is flat"
]

results = fact_checker.batch_check_claims(claims)

# Get statistics
stats = fact_checker.get_statistics(results)
print(f"True claims: {stats['true_count']}")
print(f"False claims: {stats['false_count']}")
print(f"Average confidence: {stats['avg_confidence']:.2%}")
```

### With Metadata

```python
# Add documents with source metadata (multilingual)
documents_vi = [
    "Vắc xin COVID-19 an toàn và hiệu quả.",
    "Đại dịch bắt đầu vào cuối năm 2019."
]

metadata = [
    {"source": "WHO", "date": "2024"},
    {"source": "CDC", "date": "2024"}
]

fact_checker.add_knowledge_base(documents_vi, metadata)
```

## Architecture

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq API (mixtral-8x7b-32768) | Ultra-fast inference for claims |
| **Embedding Model** | BAAI/bge-m3 | Semantic vector representations (multilingual) |
| **Vector Store** | Chroma DB | Persistent knowledge base storage |
| **Query Engine** | LlamaIndex Citation QE | Source-aware retrieval and ranking |

### Data Flow

```
Claim Input
    ↓
Citation Query Engine
    ↓
[Chroma Vector Store] ← Retrieve similar documents
    ↓
Grok LLM ← Analyze with context
    ↓
FactCheckResult (verdict + confidence + citations)
```

## Configuration

### Environment Variables

```env
# API Keys
GROK_API_KEY=your_key_here

# Model Settings
EMBEDDING_MODEL=BAAI/bge-m3
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2048

# Vector Store
CHROMA_PERSIST_DIR=./chroma_data
CHROMA_COLLECTION_NAME=news_facts

# Query Engine
SIMILARITY_TOP_K=5
CITATION_CHUNK_SIZE=512
```

### Python Configuration

```python
# Custom initialization
fact_checker = NewsFactChecker(
    grok_api_key="your_key",
    collection_name="custom_collection",
    persist_dir="./custom_data"
)
```

## API Reference

### NewsFactChecker

#### Methods

**`__init__(grok_api_key, collection_name, persist_dir)`**
- Initialize fact-checking system
- `grok_api_key`: Grok API key (defaults to env var)
- `collection_name`: Chroma collection name
- `persist_dir`: Vector store persistence directory

**`add_knowledge_base(documents, metadata=None)`**
- Add documents to the knowledge base
- `documents`: List of text documents
- `metadata`: Optional list of document metadata dicts

**`check_claim(claim, top_k=5) -> FactCheckResult`**
- Fact-check a single claim
- `claim`: Text claim to check
- `top_k`: Number of sources to retrieve
- Returns: FactCheckResult object

**`batch_check_claims(claims) -> List[FactCheckResult]`**
- Check multiple claims
- `claims`: List of claim strings
- Returns: List of FactCheckResult objects

**`get_statistics(results) -> dict`**
- Get statistics from results
- Returns: Dict with true/false/uncertain counts and percentages

### FactCheckResult

Data class containing:
- `claim`: Original claim text
- `verdict`: "true", "false", or "not_enough_information"
- `confidence`: Float 0.0-1.0
- `supporting_evidence`: List of evidence strings
- `explanation`: Explanation of verdict
- `citations`: List of source citations

## Examples

### Example 1: Basic Fact-Checking
See `examples.py` - `example_basic_fact_checking()`

### Example 2: Batch Processing
See `examples.py` - `example_batch_processing()`

### Example 3: Confidence Analysis
See `examples.py` - `example_confidence_analysis()`

Run all examples:
```bash
python examples.py
```

## Vector Store Management

### Persistent Storage

Chroma automatically persists data to disk:
```
./chroma_data/
├── chroma.sqlite3
└── index/
```

### Clear Database

```python
import shutil
shutil.rmtree("./chroma_data")
# Recreate on next initialization
```

### Change Collection

```python
fact_checker = NewsFactChecker(collection_name="other_collection")
```

## Advanced Usage

### Custom Temperature Settings

```python
# More deterministic responses
from llama_index.llms.xai import Grok
llm = Groq(api_key=key, temperature=0.1)

# More creative responses
llm = Groq(api_key=key, temperature=0.7)
```

### Embedding Caching

Embeddings are cached by default in `./embeddings_cache/` to improve performance on repeated documents.

### Batch with Progress

```python
from tqdm import tqdm

results = []
for claim in tqdm(claims):
    result = fact_checker.check_claim(claim)
    results.append(result)
```

## Troubleshooting

### Issue: "GROK_API_KEY not found"
**Solution:** Set GROK_API_KEY in `.env` file or pass to `NewsFactChecker()`

### Issue: "Failed to import llama_index"
**Solution:** Ensure dependencies installed: `pip install -r requirements.txt`

### Issue: Slow embedding generation
**Solution:** First run downloads the BAAI/bge-m3 model. Subsequent runs use cache.

### Issue: Chroma database locked
**Solution:** Ensure only one process accesses the vector store. Restart Python if needed.

## Performance Tips

1. **Batch Processing**: Check multiple claims at once
2. **Lower top_k**: Fewer sources = faster retrieval (trade-off with accuracy)
3. **Embedding Cache**: Reuse embeddings for known documents
4. **Collection Indexing**: Chroma creates indexes for faster searches

## API Costs

- **Grog API**: free 
- **BAAI/bge-m3**: Open-source, free (runs locally)
- **Chroma**: Open-source, free

## Contributing

Contributions welcome! Areas for improvement:
- Add more embedding models
- Implement web scraping for knowledge base
- Add real-time fact database integration
- Improve verdict explanation quality

## License

MIT License - See LICENSE file

## Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Groq API Documentation](https://console.groq.com/)
- [BAAI/bge-m3 Model Card](https://huggingface.co/BAAI/bge-m3)

## Support

For issues or questions:
1. Check troubleshooting section
2. Review example files
3. Check dependencies are installed
4. Verify API keys are correct

---

**Last Updated:** February 2024
**Version:** 1.0.0
#   f a c t _ c h e c k i n g 
 
 
