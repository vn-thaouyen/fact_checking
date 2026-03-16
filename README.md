# News Fact-Checking System

A fact-checking system for news claims using LlamaIndex, HuggingFace Inference API, and Chroma vector storage. Supports Vietnames and English

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment** (create `.env` file):
   ```
   HF_TOKEN=your_hugging_face_token
   ```

3. **Verify setup:**
   ```bash
   python verify_setup.py
   ```

## CLI Usage

**Check a single claim:**
```bash
python cli.py check "Your claim here" --kb sample.json
```

**Check multiple claims from a file:**
```bash
python cli.py batch claims.txt --kb knowledge_base.json --output results.json --format json
```

**Interactive mode:**
```bash
python cli.py interactive --kb knowledge_base.json
```

**View configuration:**
```bash
python cli.py config
```

## Options

- `--kb` - Path to knowledge base file (JSON or TXT)
- `--top-k` - Number of sources to retrieve (default: 5)
- `--output` - Output file for batch results
- `--format` - Output format: json, csv, markdown, or html

2. **Create environment file:**
```bash
copy .env.example .env
```

3. **Add your HuggingFace API key:**
Open `.env` and update:
```
HF_TOKEN=your_hugging_face_token

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
| **LLM** | HuggingFace Inference API | Inference for claims |
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
HF_TOKEN=your hf token

# Model Settings
EMBEDDING_MODEL=BAAI/bge-m3
LLM_TEMPERATURE=0.2
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
    hf_token="your_key",
    collection_name="custom_collection",
    persist_dir="./custom_data"
)
```

## API Reference

### NewsFactChecker

#### Methods

**`__init__(hf_token, collection_name, persist_dir)`**
- Initialize fact-checking system
- `hf_token`: HuggingFace Inference API key (defaults to env var)
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