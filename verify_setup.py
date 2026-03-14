#!/usr/bin/env python3
"""
Verification and demo script for the fact-checking system
Tests all components and shows example output
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported"""
    print("\n" + "="*70)
    print("TESTING IMPORTS")
    print("="*70)
    
    packages = [
        "llama_index",
        "llama_index_embeddings_huggingface",
        "chromadb",
        "pydantic",
        "dotenv",
    ]
    
    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports successful")
    return True


def test_configuration():
    """Test configuration loading"""
    print("\n" + "="*70)
    print("TESTING CONFIGURATION")
    print("="*70)
    
    try:
        from config import get_config
        
        try:
            config = get_config()
            print("✓ Configuration loaded successfully")
            print(f"  - Embedding Model: {config.embedding_model}")
            print(f"  - LLM Model: {config.llm_model}")
            print(f"  - Temperature: {config.llm_temperature}")
            print(f"  - HF_TOKEN: {'***' + config.hf_token[-4:] if config.hf_token else 'NOT SET'}")
            return True
        except ValueError as e:
            print(f"✗ Configuration Error: {e}")
            print("\nPlease set GROQ_API_KEY in .env file")
            return False
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False


def test_embeddings():
    """Test embedding model"""
    print("\n" + "="*70)
    print("TESTING EMBEDDINGS")
    print("="*70)
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        print("Loading BAAI/bge-m3 embedding model...")
        print("(This may take a moment on first run)")
        
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            cache_folder="./embeddings_cache"
        )
        
        # Test embedding
        test_text = "The Earth orbits the Sun"
        embedding = embed_model.get_text_embedding(test_text)
        
        print(f"✓ Embedding model loaded successfully")
        print(f"  - Model: BAAI/bge-m3")
        print(f"  - Embedding dimension: {len(embedding)}")
        print(f"  - Test text: '{test_text}'")
        print(f"  - Embedding preview: {embedding[:3]}...")
        return True
    
    except Exception as e:
        print(f"✗ Error testing embeddings: {e}")
        return False


def test_vector_store():
    """Test Chroma vector store"""
    print("\n" + "="*70)
    print("TESTING VECTOR STORE")
    print("="*70)
    
    try:
        import chromadb
        from chromadb import PersistentClient
        
        print("Initializing Chroma vector store...")
        
        client = PersistentClient(path="./chroma_test_data")
        collection = client.get_or_create_collection(
            name="test_collection"
        )
        
        print("✓ Chroma vector store initialized successfully")
        print(f"  - Collection: test_collection")
        print(f"  - Persist directory: ./chroma_test_data")
        
        # Cleanup
        import shutil
        shutil.rmtree("./chroma_test_data", ignore_errors=True)
        
        return True
    
    except Exception as e:
        print(f"✗ Error testing vector store: {e}")
        return False


def test_llm_connection():
    """Test llm connection"""
    print("\n" + "="*70)
    print("TESTING LLM CONNECTION")
    print("="*70)
    
    try:
        from config import get_config
        from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
        
        config = get_config()
        
        print("Initializing LLM...")
        llm = HuggingFaceInferenceAPI(
            api_key=config.hf_token,
            model="Qwen/Qwen2.5-32B-Instruct",
            temperature=0.2,
            max_tokens=2048
        )
        
        print("Testing HuggingFace API connection...")
        print("(Sending test message to HuggingFace API)")
        
        response = llm.complete("Say 'HuggingFace API is working' in exactly 5 words")
        
        print("HuggingFace API connection successful")
        print(f"  - Model: Qwen/Qwen2.5-3LLM connection successful")
        print(f"  - Response: {response.text[:100]}...")
        return True
    
    except Exception as e:
        print(f"Error testing Groq connection: {e}")
        print("  - Check your HF_TOKEN")
        print("  - Check internet connection")
        return False


def test_fact_checker():
    """Test main fact-checker class"""
    print("\n" + "="*70)
    print("TESTING FACT CHECKER")
    print("="*70)
    
    try:
        from src.fact_checker import NewsFactChecker
        from config import get_config
        
        config = get_config()
        
        print("Initializing NewsFactChecker...")
        fact_checker = NewsFactChecker(hf_token=config.hf_token)
        
        print("✓ Fact checker initialized successfully")
        print("  - LLM: Groq")
        print("  - Embeddings: BAAI/bge-m3")
        print("  - Vector Store: Chroma")
        
        # Add test documents
        print("\nAdding test documents...")
        test_docs = [
            "The Earth orbits the Sun in approximately 365 days.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Python is a programming language used for various applications.",
        ]
        
        fact_checker.add_knowledge_base(test_docs)
        print(f"✓ Added {len(test_docs)} test documents")
        
        return True
    
    except Exception as e:
        print(f"✗ Error testing fact checker: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sample_claim():
    """Test fact-checking with a sample claim"""
    print("\n" + "="*70)
    print("TESTING SAMPLE FACT-CHECK")
    print("="*70)
    
    try:
        from src.fact_checker import NewsFactChecker
        from config import get_config
        
        config = get_config()
        
        print("Initializing system for sample fact-check...")
        fact_checker = NewsFactChecker(hf_token=config.hf_token)
        
        # Add knowledge base
        test_docs = [
            "The Earth orbits the Sun in approximately 365 days.",
            "Water boils at 100 degrees Celsius at sea level.",
        ]
        fact_checker.add_knowledge_base(test_docs)
        
        # Check sample claim
        sample_claim = "Earth takes 365 days to orbit the Sun"
        print(f"\nFact-checking: '{sample_claim}'")
        print("(This may take 10-30 seconds)")
        
        result = fact_checker.check_claim(sample_claim)
        
        print(f"\n✓ Fact-check completed!")
        print(f"  - Verdict: {result.verdict.upper()}")
        print(f"  - Confidence: {result.confidence:.2%}")
        print(f"  - Explanation: {result.explanation[:200]}...")
        
        return True
    
    except Exception as e:
        print(f"✗ Error during sample fact-check: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("FACT-CHECKING SYSTEM VERIFICATION")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Grok LLM Connection", test_llm_connection),
        ("Fact Checker", test_fact_checker),
        ("Sample Fact-Check", test_sample_claim),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Review sample files:")
        print("   - sample_knowledge_base.txt: Example knowledge base")
        print("   - sample_claims.txt: Example claims to fact-check")
        print("\n2. Try the CLI:")
        print("   python cli.py check 'Your claim here'")
        print("   python cli.py batch sample_claims.txt --output results.json")
        print("\n3. Read the documentation:")
        print("   - README.md: Full documentation")
        print("   - SETUP.md: Detailed setup guide")
        print("   - examples.py: Code examples")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("1. GROK_API_KEY not set - Edit .env file")
        print("2. Missing dependencies - Run: pip install -r requirements.txt")
        print("3. Network issues - Check internet connection")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
