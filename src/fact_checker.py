"""
News Fact-Checking System using LlamaIndex Citation Query Engine
Uses Groq API for inference and BAAI/bge-m3 for embeddings with Chroma vector store
"""

import os
import json
from typing import Any, List, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import CitationQueryEngine
import chromadb

# Load environment variables
load_dotenv()

@dataclass
class FactCheckResult:
    """Result of fact-checking a claim"""
    claim: str
    verdict: str  # "true", "false", "not_enough_information"
    confidence: float
    supporting_evidence: List[str]
    explanation: str
    citations: List[str]
    images: List[str]
    source_nodes: Optional[List[Any]] = None  # <--- THÊM DÒNG NÀY


class NewsFactChecker:
    """Fact-checking system for Vietnamese news claims using LlamaIndex and Groq API"""
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        collection_name: str = "news_facts",
        persist_dir: str = "./chroma_data",
        language: str = "vi"
    ):
        """
        Initialize the fact-checking system
        
        Args:
            groq_api_key: Groq API key (defaults to GROQ_API_KEY env var)
            collection_name: Name of the Chroma collection
            persist_dir: Directory for persisting Chroma database
            language: Language code ('vi' for Vietnamese, 'en' for English)
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not provided and not found in environment")
        self.language = language
        
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_index()
    
    def _setup_llm(self):
        """Configure Groq LLM"""
        self.llm = Groq(
            api_key=self.groq_api_key,
            # model="mixtral-8x7b-32768",
            model="Llama-3.3-70B-Versatile",
            temperature=0.3,  # Lower temperature for factual consistency
            max_tokens=2048
        )
        
        # Set as default LLM in Settings
        Settings.llm = self.llm
    
    def _setup_embeddings(self):
        """Configure BAAI/bge-m3 embeddings"""
        import warnings
        # Suppress huggingface_hub warnings about symlinks
        warnings.filterwarnings('ignore', message='.*huggingface_hub.*cache-system.*')
        
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            cache_folder="./embeddings_cache"
        )
        
        # Set as default embedding in Settings
        Settings.embed_model = self.embed_model
    
    def _setup_vector_store(self):
        """Initialize Chroma vector store"""
        # Create persistent Chroma client
        client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create collection
        chroma_collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    def _setup_index(self):
        """Initialize the vector store index"""
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model
        )
    
    
    def add_knowledge_base(self, documents: Union[List[str], List[Document]], metadata: Optional[List[dict]] = None):
        """
        Add documents to the knowledge base
        Supports both raw text strings and pre-constructed Document objects
        """
        doc_objects = []
        
        # check if input is list of strings or list of Document objects
        if documents and isinstance(documents[0], str):
            # handle old logic: if input is list of strings, convert to Document objects with optional metadata
            for i, doc_text in enumerate(documents):
                meta = metadata[i] if metadata else {}
                if "source" not in meta:
                    meta["source"] = meta.get("title", f"Document {i+1}")
                doc_objects.append(
                    Document(text=doc_text, metadata=meta)
                )
        else:
            # new logic: if input is already a list of Document objects, use them directly (metadata should be included in Document.metadata)
            doc_objects = documents
        
        # Convert documents to nodes and add to index
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(doc_objects)
        self.index.insert_nodes(nodes)
        
        print(f"Added {len(doc_objects)} documents to knowledge base")

    
    def check_claim(self, claim: str, top_k: int = 5) -> FactCheckResult:
        """
        Check the veracity of a news claim using Citation Query Engine
        
        Args:
            claim: The claim to fact-check
            top_k: Number of top sources to retrieve
        
        Returns:
            FactCheckResult containing verdict and supporting evidence
        """
        # Create citation query engine
        query_engine = CitationQueryEngine.from_args(
            self.index,
            llm=self.llm,
            similarity_top_k=top_k,
            citation_chunk_size=512
        )
        
        # Prepare the fact-checking prompt (with language support)
        if self.language == "vi":
            prompt = f"""Bạn là một chuyên gia kiểm chứng sự kiện. Phân tích tuyên bố sau đây và cung cấp một phán quyết.

TUYÊN BỐ: {claim}

Dựa trên ngữ cảnh và kiến thức được cung cấp:
1. Xác định xem tuyên bố là ĐÚNG, SAI, hoặc KHÔNG_ĐỦ_THÔNG_TIN
2. Cung cấp điểm tự tin (0-1)
3. Liệt kê các bằng chứng hỗ trợ
4. Cung cấp giải thích ngắn gọn

Định dạng phản hồi của bạn như sau:
PHÁN QUYẾT: [ĐÚNG/SAI/KHÔNG_ĐỦ_THÔNG_TIN]
ĐỘ TỰ TIN: [0.0-1.0]
BẰNG CHỨNG: [Liệt kê các mục bằng chứng]
GIẢI THÍCH: [Giải thích của bạn]
"""
        else:
            prompt = f"""You are a fact-checking expert. Analyze the following claim and provide a verdict.

CLAIM: {claim}

Based on the provided context and knowledge:
1. Determine if the claim is TRUE, FALSE, or NOT_ENOUGH_INFORMATION
2. Provide a confidence score (0-1)
3. List the supporting evidence
4. Provide a concise explanation

Format your response as:
VERDICT: [TRUE/FALSE/NOT_ENOUGH_INFORMATION]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [List evidence items]
EXPLANATION: [Your explanation]
"""
        
        # Query the index
        response = query_engine.query(prompt)
        
        # Parse the response
        result = self._parse_response(claim, response)

        # SOURCE NODES (if available)
        if hasattr(response, 'source_nodes'):
            result.source_nodes = response.source_nodes
            
        return result
        
    
    def _parse_response(self, claim: str, response) -> FactCheckResult:
        """Parse LLM response into FactCheckResult (supports Vietnamese and English)"""
        response_text = str(response)
        import re
        
        # Extract verdict (support both English and Vietnamese)
        verdict = "not_enough_information"
        upper_response = response_text.upper()
        
        if "VERDICT: TRUE" in upper_response or ("ĐÚNG" in response_text and "PHÁN" in upper_response):
            verdict = "true"
        elif "VERDICT: FALSE" in upper_response or ("SAI" in response_text and "PHÁN" in upper_response):
            verdict = "false"
        
        # Extract confidence
        confidence = 0.5
        try:
            for line in response_text.split('\n'):
                if "CONFIDENCE:" in line.upper() or "ĐỘ TỰ TIN:" in line.upper():
                    conf_str = line.split(':')[1].strip()
                    # Extract just the number from confidence string
                    numbers = re.findall(r'\d+\.?\d*', conf_str)
                    if numbers:
                        confidence = float(numbers[0])
                    break
        except (ValueError, IndexError):
            pass
        
        # Extract evidence and citations
        evidence = []
        citations = []
        images = []
        try:
            if hasattr(response, 'source_nodes'):
                seen_sources = set()
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        # Prefer source metadata, fallback to document text
                        source = node.metadata.get('source', None)
                        if not source:
                            # Try to use first 80 chars of the node text
                            source = str(node.text)[:80] + "..." if len(str(node.text)) > 80 else str(node.text)
                            # text_content = str(node.text)
                            # limit = 500  # Increased from 80 to 500
                            # source = text_content[:limit] + "..." if len(text_content) > limit else text_content
                        
                        if source and source not in seen_sources:
                            citations.append(source)
                            seen_sources.add(source)
                        
                        # Extract images from metadata
                        # Check common keys for images
                        for img_key in ['images', 'image', 'image_url', 'img_url', 'urls']:
                            if img_key in node.metadata:
                                img_val = node.metadata[img_key]
                                if img_val:
                                    # Handle comma-separated strings (from build_db) or raw strings
                                    urls = [u.strip() for u in str(img_val).split(',')]
                                    for url in urls:
                                        if url and url not in images and (url.startswith('http') or url.startswith('/')):
                                            images.append(url)
        except:
            pass
        
        # Extract explanation (clean up formatting)
        explanation = response_text
        if "EXPLANATION:" in response_text or "GIẢI THÍCH:" in response_text:
            # Extract only the explanation part
            if "EXPLANATION:" in response_text:
                explanation = response_text.split("EXPLANATION:")[-1].strip()
            elif "GIẢI THÍCH:" in response_text:
                explanation = response_text.split("GIẢI THÍCH:")[-1].strip()
        
        # Clean up the explanation to remove any trailing metadata
        lines = explanation.split('\n')
        cleaned_lines = []
        for line in lines:
            # Stop if we hit another section marker
            if any(marker in line.upper() for marker in ["VERDICT:", "CONFIDENCE:", "EVIDENCE:", "PHÁN QUYẾT:", "ĐỘ TỰ TIN:", "BẰNG CHỨNG:"]):
                break
            cleaned_lines.append(line)
        explanation = '\n'.join(cleaned_lines).strip()
        
        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            confidence=min(confidence, 1.0),
            supporting_evidence=evidence,
            explanation=explanation,
            citations=citations,
            images=images
        )
    
    def batch_check_claims(self, claims: List[str]) -> List[FactCheckResult]:
        """
        Check multiple claims in batch
        
        Args:
            claims: List of claims to check
        
        Returns:
            List of FactCheckResult objects
        """
        results = []
        for claim in claims:
            try:
                result = self.check_claim(claim)
                results.append(result)
                print(f"✓ Checked: {claim[:50]}... -> {result.verdict.upper()}")
            except Exception as e:
                print(f"✗ Error checking claim: {claim[:50]}... -> {str(e)}")
        
        return results
    
    def get_statistics(self, results: List[FactCheckResult]) -> dict:
        """Get statistics from fact-checking results"""
        if not results:
            return {}
        
        verdicts = [r.verdict for r in results]
        confidences = [r.confidence for r in results]
        
        return {
            "total_claims": len(results),
            "true_count": verdicts.count("true"),
            "false_count": verdicts.count("false"),
            "uncertain_count": verdicts.count("not_enough_information"),
            "avg_confidence": sum(confidences) / len(confidences),
            "true_percentage": (verdicts.count("true") / len(results)) * 100,
            "false_percentage": (verdicts.count("false") / len(results)) * 100,
        }


if __name__ == "__main__":
    # Example usage
    print("Initializing News Fact-Checking System...")
    
    try:
        # Initialize the fact-checker
        fact_checker = NewsFactChecker()
        
        # Add sample documents to knowledge base
        sample_docs = [
            "The Earth is round and orbits the Sun. Scientific consensus confirms the spherical shape of Earth.",
            "Water boils at 100 degrees Celsius at sea level due to atmospheric pressure.",
            "The Great Wall of China is visible from space with the naked eye. This is a common misconception.",
            "Vaccines have been proven safe and effective through millions of doses worldwide.",
            "COVID-19 is a viral disease that spreads through respiratory droplets.",
        ]
        
        fact_checker.add_knowledge_base(sample_docs)
        
        # Test claims
        test_claims = [
            "The Earth is flat",
            "Water boils at 100 degrees Celsius",
            "The Great Wall can be seen from space with naked eyes",
        ]
        
        print("\n" + "="*60)
        print("FACT-CHECKING RESULTS")
        print("="*60)
        
        results = fact_checker.batch_check_claims(test_claims)
        
        # Display results
        print("\n" + "="*60)
        for result in results:
            print(f"\nClaim: {result.claim}")
            print(f"Verdict: {result.verdict.upper()}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Explanation: {result.explanation[:200]}...")
            if result.citations:
                print(f"Sources: {', '.join(result.citations[:2])}")
        
        # Show statistics
        stats = fact_checker.get_statistics(results)
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
