"""
Command-line interface for the News Fact-Checking System
"""

import sys
import argparse
import json
import os
import re
from typing import List, Tuple, Union
from llama_index.core import Document
from src.fact_checker import NewsFactChecker
from config import get_config
from src.utils import ResultExporter, StatisticsAnalyzer, ClaimValidator

def extract_date_from_id(doc_id: str) -> str:
    match = re.search(r'(\d{2}-\d{2}-\d{4})', str(doc_id))
    if match:
        return match.group(1)
    return "Không xác định"

def load_claims_from_file(filepath: str) -> List[str]:
    """Load claims from a text file (one claim per line)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            claims = [line.strip() for line in f if line.strip()]
        return claims
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)


def load_knowledge_base(filepath: str) -> Tuple[Union[List[str], List[Document]], List[dict]]:
    """Load knowledge base documents from a file (TXT or JSON)
    
    Returns:
        Tuple of (documents list, metadata list)
        - If JSON: returns (List[Document], [])
        - If TXT: returns (List[str], List[dict])
    """
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)

    try:
        # JSON file handling (with metadata and images)
        if filepath.lower().endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for entry in data:
                text = entry.get('text', '') or entry.get('content', '')
                if not text:
                    continue
                
                # get source name and title
                doc_id = entry.get("id", "")
                source_name = entry.get("name", entry.get("source", "")).strip()
                title = entry.get("title", "").strip()
                description = entry.get("description", "").strip()

                date_str = extract_date_from_id(doc_id)
                
                enriched_text = (
                    f"Ngày phát hành: {date_str}\n"
                    f"Tiêu đề: {title}\n"
                    f"Nguồn: {source_name}\n"
                    f"Mô tả: {description}\n"
                    f"Nội dung chi tiết: {text}"
                )
                
                # Creare a full source string combining name and title for better metadata (instead of just name)
                # Ex: "Báo Dân Trí: Bão tuyết đốt chục tỷ USD..."
                if source_name and title:
                    full_source = f"{source_name}: {title}"
                else:
                    full_source = source_name or title or "Unknown Source"

                # metadata
                metadata = {
                    "id": entry.get("id", ""),
                    "title": title,
                    "date": date_str,
                    "source": full_source,
                    "description": description,
                    "category": entry.get("category", ""),
                    "images": json.dumps(entry.get("images", []))
                }
                
                # Create Documeent object with text and metadata
                doc = Document(text=enriched_text, metadata=metadata)
                documents.append(doc)
            
            # Retuen list of Document objects and empty metadata (since metadata is embedded in Document)
            return documents, []

        # TXT file handling (simple split by double newlines, with basic metadata)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                documents = content.split('\n\n')
                docs = [doc.strip() for doc in documents if doc.strip()]
                
                metadata = []
                for i, doc in enumerate(docs):
                    first_line = doc.split('\n')[0].strip()
                    title = first_line[:100] if len(first_line) > 100 else first_line
                    metadata.append({
                        "source": title or f"Document {i+1}",
                        "full_text": doc
                    })
                
                return docs, metadata

    except Exception as e:
        print(f"Error loading knowledge base '{filepath}': {e}")
        sys.exit(1)


def print_result(result):
    """
    Print the fact-checking result in citation
    format, including verdict, confidence, explanation, and detailed sources with images.
    """
    
    print(f"\nVerdict: {result.verdict.upper()}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nExplanation:")
    print(result.explanation)
    
    print("\n" + "="*30)
    print("DETAILED SOURCES & IMAGES")
    print("="*30)

    # Check if result has source_nodes (from fact-cheker.py)
    if hasattr(result, 'source_nodes') and result.source_nodes:
        for idx, node in enumerate(result.source_nodes, 1):
            # 1. Get text content from node
            content = node.node.get_text().strip()
            
            # 2. get metadata (source name, title, images) from node
            meta = node.node.metadata
            source_name = meta.get('source', 'Unknown Source')
            
            print(f"\nSource {idx}:")
            print(f"Origin: {source_name}")
            #print(f"Content: \"{content[:300]}...\"") 
            print(f"Content: {content}")
            
            # 3. Get images from metadata if available (handle both JSON string and list)
            if 'images' in meta:
                img_data = meta['images']
                img_list = []
                
                # Handling parseing of images data which can be either a JSON string or a list
                if isinstance(img_data, str):
                    try:
                        import json
                        # if JSON list "['url']"
                        if img_data.startswith('[') and img_data.endswith(']'):
                            img_list = json.loads(img_data)
                        else:
                            img_list = [img_data]
                    except:
                        img_list = [img_data] # Fallback if cannot parse JSON, treat as single URL string
                elif isinstance(img_data, list):
                    img_list = img_data

                # print images if we have any valid URLs
                if img_list:
                    print(f"Evidence Image(s):")
                    for img in img_list:
                        if img and isinstance(img, str) and len(img) > 5:
                            print(f"   {img}")
            
            print("-" * 20)
            
    else:
        # Fallback if no source_nodes, just print citations and images from result if available
        if result.citations:
            print("\nSources:")
            for cite in result.citations:
                print(f"  • {cite}")
        
        if result.images:
            print(f"\nAll Images:")
            for img in result.images:
                print(f"  • {img}")

    print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="News Fact-Checking System using LlamaIndex and Groq API"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Check single claim
    check_parser = subparsers.add_parser("check", help="Check a single claim")
    check_parser.add_argument("claim", help="The claim to fact-check")
    check_parser.add_argument("--kb", type=str, help="Path to knowledge base file (txt or json)")
    check_parser.add_argument("--top-k", type=int, default=5, help="Top K sources to retrieve")
    
    # Check multiple claims from file
    batch_parser = subparsers.add_parser("batch", help="Check multiple claims from file")
    batch_parser.add_argument("claims_file", help="Path to file with claims (one per line)")
    batch_parser.add_argument("--kb", type=str, help="Path to knowledge base file")
    batch_parser.add_argument("--output", type=str, help="Output file for results")
    batch_parser.add_argument("--format", choices=["json", "csv", "markdown", "html"], 
                            default="json", help="Output format")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive fact-checking mode")
    interactive_parser.add_argument("--kb", type=str, help="Path to knowledge base file")
    
    # Config command
    subparsers.add_parser("config", help="View current configuration")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    
    # Handle commands
    if args.command == "config":
        print("\nCurrent Configuration:")
        print("="*50)
        for key, value in config.to_dict().items():
            print(f"{key:.<30} {value}")
        print("="*50)
    
    elif args.command == "check":
        print(f"\nFact-Checking: {args.claim}")
        print("="*60)
        
        try:
            fact_checker = NewsFactChecker(
                hf_token=config.hf_token,
                collection_name=config.chroma_collection_name,
                persist_dir=config.chroma_persist_dir,
                language="vi" # Changed default to 'vi' based on context
            )
            
            if args.kb:
                kb_docs, kb_metadata = load_knowledge_base(args.kb)
                # kb_docs now can be List[Document] or List[str]
                fact_checker.add_knowledge_base(kb_docs, kb_metadata)
                print(f"Loaded {len(kb_docs)} documents from knowledge base\n")
            
            result = fact_checker.check_claim(args.claim, top_k=args.top_k)
            print_result(result)
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == "batch":
        print(f"\nBatch Fact-Checking")
        print("="*60)
        
        try:
            claims = load_claims_from_file(args.claims_file)
            print(f"Loaded {len(claims)} claims from {args.claims_file}")
            
            claims = ClaimValidator.validate_claims(claims)
            print(f"{len(claims)} valid claims to check\n")
            
            fact_checker = NewsFactChecker(
                hf_token=config.hf_token,
                collection_name=config.chroma_collection_name,
                persist_dir=config.chroma_persist_dir,
                language="vi"
            )
            
            if args.kb:
                kb_docs, kb_metadata = load_knowledge_base(args.kb)
                fact_checker.add_knowledge_base(kb_docs, kb_metadata)
                print(f"Loaded {len(kb_docs)} documents from knowledge base\n")
            
            print("Checking claims...")
            results = fact_checker.batch_check_claims(claims)
            
            print("\n" + "="*60)
            StatisticsAnalyzer.print_report(results)
            
            if args.output:
                output_file = args.output if args.output.endswith(f".{args.format}") else f"{args.output}.{args.format}"
                if args.format == "json":
                    ResultExporter.to_json(results, output_file)
                elif args.format == "csv":
                    ResultExporter.to_csv(results, output_file)
                elif args.format == "markdown":
                    ResultExporter.to_markdown(results, output_file)
                elif args.format == "html":
                    ResultExporter.to_html(results, output_file)
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == "interactive":
        print("\nInteractive Fact-Checking Mode")
        print("="*60)
        print("Type claims to fact-check, 'exit' to quit\n")
        
        try:
            fact_checker = NewsFactChecker(
                hf_token=config.hf_token,
                collection_name=config.chroma_collection_name,
                persist_dir=config.chroma_persist_dir,
                language="vi"
            )
            
            if args.kb:
                kb_docs, kb_metadata = load_knowledge_base(args.kb)
                fact_checker.add_knowledge_base(kb_docs, kb_metadata)
                print(f"Loaded {len(kb_docs)} documents from knowledge base\n")
            
            while True:
                claim = input("Enter claim (or 'exit' to quit): ").strip()
                
                if claim.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                
                if not claim:
                    continue
                
                try:
                    print("\nAnalyzing claim...")
                    result = fact_checker.check_claim(claim)
                    print_result(result) 
                
                except Exception as e:
                    print(f"Error analyzing claim: {e}\n")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()