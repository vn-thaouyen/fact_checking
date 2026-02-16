"""
Utility functions for the fact-checking system
"""

import json
import csv
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from src.fact_checker import FactCheckResult


class ResultExporter:
    """Export fact-checking results in various formats"""
    
    @staticmethod
    def to_json(results: List[FactCheckResult], filepath: str = "results.json") -> None:
        """
        Export results to JSON format
        
        Args:
            results: List of FactCheckResult objects
            filepath: Output file path
        """
        data = {
            "export_date": datetime.now().isoformat(),
            "total_claims": len(results),
            "results": [
                {
                    "claim": r.claim,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "explanation": r.explanation,
                    "citations": r.citations,
                    "supporting_evidence": r.supporting_evidence,
                    "images": r.images
                }
                for r in results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results exported to {filepath}")
    
    @staticmethod
    def to_csv(results: List[FactCheckResult], filepath: str = "results.csv") -> None:
        """
        Export results to CSV format
        
        Args:
            results: List of FactCheckResult objects
            filepath: Output file path
        """
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Claim",
                "Verdict",
                "Confidence",
                "Explanation",
                "Citations",
                "Images"
            ])
            
            # Write data
            for result in results:
                writer.writerow([
                    result.claim,
                    result.verdict,
                    f"{result.confidence:.2%}",
                    result.explanation[:200],  # Truncate for readability
                    "; ".join(result.citations) if result.citations else "",
                    "; ".join(result.images) if result.images else ""
                ])
        
        print(f"✓ Results exported to {filepath}")
    
    @staticmethod
    def to_markdown(results: List[FactCheckResult], filepath: str = "results.md") -> None:
        """
        Export results to Markdown format
        
        Args:
            results: List of FactCheckResult objects
            filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# Fact-Checking Results\n\n")
            f.write(f"**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Claims:** {len(results)}\n\n")
            
            # Statistics
            verdicts = [r.verdict for r in results]
            f.write("## Summary\n\n")
            f.write(f"- **True:** {verdicts.count('true')} ({verdicts.count('true')/len(results)*100:.1f}%)\n")
            f.write(f"- **False:** {verdicts.count('false')} ({verdicts.count('false')/len(results)*100:.1f}%)\n")
            f.write(f"- **Uncertain:** {verdicts.count('not_enough_information')} ({verdicts.count('not_enough_information')/len(results)*100:.1f}%)\n\n")
            
            # Individual results
            f.write("## Detailed Results\n\n")
            for i, result in enumerate(results, 1):
                verdict_emoji = {
                    "true": "✅",
                    "false": "❌",
                    "not_enough_information": "❓"
                }.get(result.verdict, "❓")
                
                f.write(f"### {i}. {result.claim}\n\n")
                f.write(f"**Verdict:** {verdict_emoji} {result.verdict.upper()}\n\n")
                f.write(f"**Confidence:** {result.confidence:.2%}\n\n")
                f.write(f"**Explanation:**\n{result.explanation}\n\n")
                
                if result.citations:
                    f.write(f"**Sources:**\n")
                    for citation in result.citations:
                        f.write(f"- {citation}\n")
                    f.write("\n")
                
                if result.images:
                    f.write(f"**Images:**\n")
                    for img in result.images:
                        f.write(f"- ![]({img})\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        print(f"✓ Results exported to {filepath}")
    
    @staticmethod
    def to_html(results: List[FactCheckResult], filepath: str = "results.html") -> None:
        """
        Export results to HTML format
        
        Args:
            results: List of FactCheckResult objects
            filepath: Output file path
        """
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fact-Checking Results</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, segoe ui, helvetica neue, Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 30px;
                }
                .stat-card {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .stat-value {
                    font-size: 32px;
                    font-weight: bold;
                    color: #667eea;
                }
                .stat-label {
                    color: #666;
                    margin-top: 5px;
                }
                .result {
                    background: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    border-left: 4px solid #ccc;
                }
                .result.true {
                    border-left-color: #28a745;
                }
                .result.false {
                    border-left-color: #dc3545;
                }
                .result.uncertain {
                    border-left-color: #ffc107;
                }
                .verdict {
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .verdict.true {
                    background-color: #d4edda;
                    color: #155724;
                }
                .verdict.false {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                .verdict.uncertain {
                    background-color: #fff3cd;
                    color: #856404;
                }
                .confidence {
                    margin: 10px 0;
                }
                .confidence-bar {
                    height: 8px;
                    background-color: #e9ecef;
                    border-radius: 4px;
                    overflow: hidden;
                }
                .confidence-fill {
                    height: 100%;
                    background-color: #667eea;
                    transition: width 0.3s;
                }
                .claim {
                    font-size: 18px;
                    font-weight: 500;
                    margin-bottom: 10px;
                }
                .explanation {
                    color: #555;
                    line-height: 1.6;
                    margin: 10px 0;
                }
                .sources {
                    margin-top: 10px;
                    padding-top: 10px;
                    border-top: 1px solid #eee;
                }
                .source {
                    color: #667eea;
                    text-decoration: none;
                    margin-right: 10px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Fact-Checking Results</h1>
                <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
        """
        
        # Add statistics
        verdicts = [r.verdict for r in results]
        true_count = verdicts.count("true")
        false_count = verdicts.count("false")
        uncertain_count = verdicts.count("not_enough_information")
        
        html_content += f"""
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Total Claims</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{true_count}</div>
                    <div class="stat-label">True (✅)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{false_count}</div>
                    <div class="stat-label">False (❌)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{uncertain_count}</div>
                    <div class="stat-label">Uncertain (❓)</div>
                </div>
            </div>
        """
        
        # Add results
        for result in results:
            confidence_percent = result.confidence * 100
            
            html_content += f"""
            <div class="result {result.verdict}">
                <div class="claim">{result.claim}</div>
                <span class="verdict {result.verdict}">{result.verdict.upper()}</span>
                <div class="confidence">
                    <strong>Confidence:</strong> {confidence_percent:.1f}%
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_percent}%"></div>
                    </div>
                </div>
                <div class="explanation"><strong>Explanation:</strong> {result.explanation}</div>
            """
            
            if result.citations:
                html_content += '<div class="sources"><strong>Sources:</strong><br>'
                for citation in result.citations:
                    html_content += f'<a href="#" class="source">{citation}</a>'
                html_content += '</div>'
            
            if result.images:
                html_content += '<div class="images" style="margin-top:10px;"><strong>Images:</strong><br>'
                for img in result.images:
                    html_content += f'<img src="{img}" style="max-height:100px; margin:5px; border-radius:4px;">'
                html_content += '</div>'
            
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Results exported to {filepath}")


class StatisticsAnalyzer:
    """Analyze fact-checking results"""
    
    @staticmethod
    def get_summary(results: List[FactCheckResult]) -> Dict[str, Any]:
        """
        Get summary statistics
        
        Args:
            results: List of FactCheckResult objects
        
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        verdicts = [r.verdict for r in results]
        confidences = [r.confidence for r in results]
        
        return {
            "total": len(results),
            "true": verdicts.count("true"),
            "false": verdicts.count("false"),
            "uncertain": verdicts.count("not_enough_information"),
            "true_percentage": (verdicts.count("true") / len(results)) * 100,
            "false_percentage": (verdicts.count("false") / len(results)) * 100,
            "uncertain_percentage": (verdicts.count("not_enough_information") / len(results)) * 100,
            "avg_confidence": sum(confidences) / len(confidences),
            "high_confidence_claims": sum(1 for r in results if r.confidence > 0.8),
            "low_confidence_claims": sum(1 for r in results if r.confidence < 0.5),
        }
    
    @staticmethod
    def print_report(results: List[FactCheckResult]) -> None:
        """Print a formatted statistics report"""
        stats = StatisticsAnalyzer.get_summary(results)
        
        if not stats:
            print("No results to analyze")
            return
        
        print("\n" + "="*60)
        print("FACT-CHECKING REPORT")
        print("="*60)
        print(f"Total Claims Checked: {stats['total']}")
        print(f"  ✅ True:     {stats['true']:>3} ({stats['true_percentage']:.1f}%)")
        print(f"  ❌ False:    {stats['false']:>3} ({stats['false_percentage']:.1f}%)")
        print(f"  ❓ Uncertain: {stats['uncertain']:>3} ({stats['uncertain_percentage']:.1f}%)")
        print(f"\nConfidence Analysis:")
        print(f"  Average Confidence: {stats['avg_confidence']:.2%}")
        print(f"  High Confidence (>80%): {stats['high_confidence_claims']}")
        print(f"  Low Confidence (<50%): {stats['low_confidence_claims']}")
        print("="*60 + "\n")


class ClaimValidator:
    """Validate and preprocess claims"""
    
    @staticmethod
    def clean_claim(claim: str) -> str:
        """Remove common claim artifacts"""
        claim = claim.strip()
        # Remove common question marks/punctuation
        if claim.endswith("?"):
            claim = claim[:-1].strip()
        return claim
    
    @staticmethod
    def validate_claims(claims: List[str]) -> List[str]:
        """Validate list of claims"""
        valid_claims = []
        for claim in claims:
            if not claim or len(claim) < 5:
                print(f"⚠️  Skipping invalid claim: '{claim}'")
                continue
            valid_claims.append(ClaimValidator.clean_claim(claim))
        return valid_claims


if __name__ == "__main__":
    print("Utilities module - use with fact_checker.py")
