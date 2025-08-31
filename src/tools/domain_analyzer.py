"""Domain analyzer tool for insurance business domains."""

from typing import Dict, Any, List, Optional
from ..config.config import get_domain_config


class DomainAnalyzer:
    """Tool for analyzing insurance business domains."""
    
    def __init__(self):
        """Initialize domain analyzer."""
        self.domain_config = get_domain_config()
    
    def analyze_domain(self, text: str) -> Dict[str, Any]:
        """Analyze text to identify insurance domains.
        
        Args:
            text: Text to analyze
            
        Returns:
            Domain analysis results
        """
        text_lower = text.lower()
        domain_scores = {}
        
        for domain_name, config in self.domain_config.DOMAINS.items():
            score = 0
            matched_keywords = []
            matched_rules = []
            
            # Check keywords
            for keyword in config["keywords"]:
                if keyword.lower() in text_lower:
                    score += 2
                    matched_keywords.append(keyword)
            
            # Check business rules
            for rule in config["business_rules"]:
                if rule.lower() in text_lower:
                    score += 3
                    matched_rules.append(rule)
            
            if score > 0:
                domain_scores[domain_name] = {
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "matched_rules": matched_rules,
                    "confidence": self._calculate_confidence(score, len(text_lower))
                }
        
        # Identify primary domain
        primary_domain = None
        if domain_scores:
            primary_domain = max(domain_scores.items(), key=lambda x: x[1]["score"])[0]
        
        return {
            "primary_domain": primary_domain,
            "all_domains": domain_scores,
            "domain_count": len(domain_scores)
        }
    
    def _calculate_confidence(self, score: int, text_length: int) -> str:
        """Calculate confidence level for domain identification.
        
        Args:
            score: Domain score
            text_length: Length of analyzed text
            
        Returns:
            Confidence level
        """
        # Normalize score based on text length
        if text_length < 100:
            normalized_score = score * 2
        elif text_length < 500:
            normalized_score = score * 1.5
        else:
            normalized_score = score
        
        if normalized_score >= 15:
            return "high"
        elif normalized_score >= 8:
            return "medium"
        else:
            return "low"
    
    def get_domain_context(self, domain_name: str) -> Dict[str, Any]:
        """Get context for a specific domain.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            Domain context
        """
        if domain_name in self.domain_config.DOMAINS:
            return self.domain_config.DOMAINS[domain_name]
        return {}
    
    def identify_cross_domain_dependencies(
        self,
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """Identify dependencies between domains.
        
        Args:
            domains: List of domain names
            
        Returns:
            Cross-domain dependencies
        """
        dependencies = []
        
        # Common cross-domain dependencies in insurance
        dependency_map = {
            ("policy_management", "billing_finance"): "Premium calculation and collection",
            ("policy_management", "claims_processing"): "Coverage validation for claims",
            ("claims_processing", "billing_finance"): "Claim payments and deductibles",
            ("customer_management", "policy_management"): "Customer policy association",
            ("regulatory_compliance", "all"): "Compliance requirements across all domains"
        }
        
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                key = tuple(sorted([domain1, domain2]))
                if key in dependency_map:
                    dependencies.append({
                        "domains": list(key),
                        "description": dependency_map[key],
                        "importance": "high"
                    })
        
        # Check for regulatory compliance
        if "regulatory_compliance" in domains:
            for domain in domains:
                if domain != "regulatory_compliance":
                    dependencies.append({
                        "domains": ["regulatory_compliance", domain],
                        "description": f"Compliance requirements for {domain}",
                        "importance": "critical"
                    })
        
        return dependencies