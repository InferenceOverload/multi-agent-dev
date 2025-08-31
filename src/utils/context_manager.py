"""Context window management for large codebases."""

from typing import Dict, Any, List, Optional
import hashlib
from datetime import datetime, timedelta

from ..utils.logger import get_logger


logger = get_logger(__name__)


class ContextWindowManager:
    """Manages context windows for large codebase analysis."""
    
    def __init__(
        self,
        max_context_size: int = 100000,
        compression_ratio: float = 0.3
    ):
        """Initialize context window manager.
        
        Args:
            max_context_size: Maximum context size in tokens
            compression_ratio: Target compression ratio
        """
        self.max_context_size = max_context_size
        self.compression_ratio = compression_ratio
        self.context_cache = {}
        self.context_utilization_target = 0.85
    
    async def build_story_context(
        self,
        story: Dict[str, Any],
        repo_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build focused context for story implementation.
        
        Args:
            story: Story information
            repo_analysis: Repository analysis results
            
        Returns:
            Compressed context for story
        """
        # Calculate optimal context size
        optimal_size = self.calculate_optimal_context_size(story, repo_analysis)
        
        # Select relevant files
        relevant_files = self._select_relevant_files(
            story, repo_analysis.get("files", []), optimal_size
        )
        
        # Compress context if needed
        if self._estimate_context_size(relevant_files) > optimal_size:
            compressed_files = await self._compress_context(
                relevant_files, optimal_size
            )
        else:
            compressed_files = relevant_files
        
        # Build implementation context
        implementation_context = self._build_implementation_context(
            compressed_files, story
        )
        
        return {
            "files": compressed_files,
            "patterns": self._extract_patterns(compressed_files),
            "dependencies": self._extract_dependencies(compressed_files),
            "approach": implementation_context["approach"],
            "complexity": implementation_context["complexity"]
        }
    
    def calculate_optimal_context_size(
        self,
        story: Dict[str, Any],
        repo_analysis: Dict[str, Any]
    ) -> int:
        """Calculate optimal context window size.
        
        Args:
            story: Story information
            repo_analysis: Repository analysis
            
        Returns:
            Optimal context size
        """
        # Base calculation
        story_complexity = story.get("complexity", "medium")
        repo_size = len(repo_analysis.get("files", []))
        
        # Complexity multipliers
        complexity_multipliers = {
            "low": 0.7,
            "medium": 1.0,
            "high": 1.5,
            "very_high": 2.0
        }
        
        # Repository size multipliers
        if repo_size < 100:
            repo_multiplier = 0.8
        elif repo_size < 500:
            repo_multiplier = 1.0
        elif repo_size < 1000:
            repo_multiplier = 1.3
        else:
            repo_multiplier = 1.5
        
        base_size = self.max_context_size * 0.5
        complexity_mult = complexity_multipliers.get(story_complexity, 1.0)
        
        optimal_size = int(base_size * complexity_mult * repo_multiplier)
        
        # Apply target utilization
        optimal_size = int(optimal_size * self.context_utilization_target)
        
        return min(optimal_size, self.max_context_size)
    
    def _select_relevant_files(
        self,
        story: Dict[str, Any],
        files: List[Dict[str, Any]],
        max_size: int
    ) -> List[Dict[str, Any]]:
        """Select most relevant files for story.
        
        Args:
            story: Story information
            files: Available files
            max_size: Maximum context size
            
        Returns:
            Selected relevant files
        """
        # Extract story keywords
        story_keywords = self._extract_story_keywords(story)
        
        # Calculate relevance scores
        file_scores = []
        for file_info in files:
            score = self._calculate_file_relevance(
                file_info, story_keywords, story
            )
            file_scores.append((file_info, score))
        
        # Sort by relevance
        file_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select files within size limit
        selected_files = []
        current_size = 0
        
        for file_info, score in file_scores:
            file_size = self._estimate_file_size(file_info)
            if current_size + file_size <= max_size:
                selected_files.append(file_info)
                current_size += file_size
            elif score > 0.7:  # High relevance files
                # Try to include partial content
                partial_file = self._create_partial_file(
                    file_info, max_size - current_size
                )
                if partial_file:
                    selected_files.append(partial_file)
                    break
        
        logger.info(
            f"Selected {len(selected_files)} files "
            f"(utilization: {current_size/max_size:.2%})"
        )
        
        return selected_files
    
    def _extract_story_keywords(self, story: Dict[str, Any]) -> List[str]:
        """Extract keywords from story.
        
        Args:
            story: Story information
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # From title
        if "title" in story:
            keywords.extend(story["title"].lower().split())
        
        # From description
        if "description" in story:
            desc_words = story["description"].lower().split()[:30]
            keywords.extend(desc_words)
        
        # From technical areas
        if "technical_areas" in story:
            keywords.extend(story["technical_areas"])
        
        # From domain
        if "domain" in story:
            keywords.append(story["domain"])
        
        # Remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "as"
        }
        
        keywords = [
            k for k in keywords 
            if k not in stop_words and len(k) > 2
        ]
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        return unique_keywords[:20]
    
    def _calculate_file_relevance(
        self,
        file_info: Dict[str, Any],
        keywords: List[str],
        story: Dict[str, Any]
    ) -> float:
        """Calculate file relevance score.
        
        Args:
            file_info: File information
            keywords: Story keywords
            story: Story information
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        weights = {
            "path_match": 0.3,
            "content_match": 0.4,
            "domain_match": 0.2,
            "pattern_match": 0.1
        }
        
        # Path matching
        path = file_info.get("path", "").lower()
        path_matches = sum(1 for k in keywords if k in path)
        path_score = min(path_matches / len(keywords), 1.0) if keywords else 0
        score += path_score * weights["path_match"]
        
        # Content matching (if available)
        content = file_info.get("content", "")[:2000].lower()
        if content:
            content_matches = sum(1 for k in keywords if k in content)
            content_score = min(content_matches / len(keywords), 1.0) if keywords else 0
            score += content_score * weights["content_match"]
        
        # Domain matching
        file_domain = file_info.get("domain")
        story_domain = story.get("domain")
        if file_domain and story_domain and file_domain == story_domain:
            score += weights["domain_match"]
        
        # Pattern matching
        file_patterns = set(file_info.get("patterns", []))
        story_patterns = set(story.get("required_patterns", []))
        if file_patterns and story_patterns:
            pattern_overlap = len(file_patterns & story_patterns)
            pattern_score = pattern_overlap / len(story_patterns)
            score += pattern_score * weights["pattern_match"]
        
        return min(score, 1.0)
    
    def _estimate_file_size(self, file_info: Dict[str, Any]) -> int:
        """Estimate file size in tokens.
        
        Args:
            file_info: File information
            
        Returns:
            Estimated size in tokens
        """
        # Rough estimation: 1 token ≈ 4 characters
        content = file_info.get("content", "")
        if content:
            return len(content) // 4
        
        # Fallback to file size if content not available
        size = file_info.get("size", 0)
        return size // 4
    
    def _estimate_context_size(self, files: List[Dict[str, Any]]) -> int:
        """Estimate total context size.
        
        Args:
            files: List of files
            
        Returns:
            Estimated total size
        """
        return sum(self._estimate_file_size(f) for f in files)
    
    async def _compress_context(
        self,
        files: List[Dict[str, Any]],
        target_size: int
    ) -> List[Dict[str, Any]]:
        """Compress context to target size.
        
        Args:
            files: Files to compress
            target_size: Target context size
            
        Returns:
            Compressed files
        """
        compressed_files = []
        current_size = 0
        
        for file_info in files:
            file_size = self._estimate_file_size(file_info)
            
            if current_size + file_size <= target_size:
                # Include full file
                compressed_files.append(file_info)
                current_size += file_size
            else:
                # Compress file content
                remaining_size = target_size - current_size
                if remaining_size > 1000:  # Minimum useful size
                    compressed_file = await self._compress_file(
                        file_info, remaining_size
                    )
                    if compressed_file:
                        compressed_files.append(compressed_file)
                        current_size += self._estimate_file_size(compressed_file)
                
                if current_size >= target_size * 0.95:
                    break
        
        logger.info(
            f"Compressed {len(files)} files to {len(compressed_files)} "
            f"(size: {current_size}/{target_size})"
        )
        
        return compressed_files
    
    async def _compress_file(
        self,
        file_info: Dict[str, Any],
        target_size: int
    ) -> Optional[Dict[str, Any]]:
        """Compress single file to target size.
        
        Args:
            file_info: File to compress
            target_size: Target size in tokens
            
        Returns:
            Compressed file or None
        """
        content = file_info.get("content", "")
        if not content:
            return None
        
        # Strategy 1: Extract most relevant sections
        relevant_sections = self._extract_relevant_sections(content, file_info)
        
        compressed_file = file_info.copy()
        compressed_content = []
        current_size = 0
        
        for section in relevant_sections:
            section_size = len(section) // 4
            if current_size + section_size <= target_size:
                compressed_content.append(section)
                current_size += section_size
            else:
                # Truncate section
                remaining = (target_size - current_size) * 4
                if remaining > 100:
                    compressed_content.append(section[:remaining])
                break
        
        compressed_file["content"] = "\n...\n".join(compressed_content)
        compressed_file["compressed"] = True
        compressed_file["original_size"] = len(content)
        compressed_file["compressed_size"] = len(compressed_file["content"])
        
        return compressed_file
    
    def _extract_relevant_sections(
        self,
        content: str,
        file_info: Dict[str, Any]
    ) -> List[str]:
        """Extract most relevant sections from file content.
        
        Args:
            content: File content
            file_info: File information
            
        Returns:
            List of relevant sections
        """
        sections = []
        
        # Strategy: Extract classes, functions, and important blocks
        language = file_info.get("language", "unknown")
        
        if language == "Python":
            # Extract classes and functions
            lines = content.split('\n')
            current_section = []
            in_class = False
            in_function = False
            
            for line in lines:
                if line.startswith('class '):
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                    in_class = True
                elif line.startswith('def '):
                    if not in_class and current_section:
                        sections.append('\n'.join(current_section))
                    current_section.append(line) if in_class else [line]
                    in_function = True
                elif line and not line[0].isspace():
                    # New top-level element
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                    in_class = False
                    in_function = False
                elif current_section:
                    current_section.append(line)
            
            if current_section:
                sections.append('\n'.join(current_section))
        
        elif language in ["Java", "C#"]:
            # Extract classes and methods
            lines = content.split('\n')
            current_section = []
            brace_count = 0
            
            for line in lines:
                if 'class ' in line or 'interface ' in line:
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                    brace_count = line.count('{') - line.count('}')
                elif current_section:
                    current_section.append(line)
                    brace_count += line.count('{') - line.count('}')
                    if brace_count == 0 and len(current_section) > 1:
                        sections.append('\n'.join(current_section))
                        current_section = []
            
            if current_section:
                sections.append('\n'.join(current_section))
        
        else:
            # Fallback: Split into chunks
            chunk_size = 500
            lines = content.split('\n')
            for i in range(0, len(lines), chunk_size):
                sections.append('\n'.join(lines[i:i+chunk_size]))
        
        return sections[:10]  # Limit to top 10 sections
    
    def _create_partial_file(
        self,
        file_info: Dict[str, Any],
        max_size: int
    ) -> Optional[Dict[str, Any]]:
        """Create partial file within size limit.
        
        Args:
            file_info: Original file
            max_size: Maximum size allowed
            
        Returns:
            Partial file or None
        """
        content = file_info.get("content", "")
        if not content:
            return None
        
        # Calculate how much content we can include
        max_chars = max_size * 4  # Rough token to char conversion
        
        if len(content) <= max_chars:
            return file_info
        
        partial_file = file_info.copy()
        partial_file["content"] = content[:max_chars] + "\n... [truncated]"
        partial_file["partial"] = True
        partial_file["original_size"] = len(content)
        
        return partial_file
    
    def _build_implementation_context(
        self,
        files: List[Dict[str, Any]],
        story: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build implementation context from files.
        
        Args:
            files: Selected files
            story: Story information
            
        Returns:
            Implementation context
        """
        # Analyze file patterns
        patterns = self._extract_patterns(files)
        
        # Determine approach based on patterns
        approach = self._determine_approach(patterns, story)
        
        # Assess complexity
        complexity = self._assess_complexity(files, story)
        
        return {
            "approach": approach,
            "complexity": complexity,
            "key_files": [f["path"] for f in files[:5]],
            "patterns": patterns
        }
    
    def _extract_patterns(self, files: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns from files.
        
        Args:
            files: List of files
            
        Returns:
            List of patterns
        """
        all_patterns = []
        
        for file_info in files:
            patterns = file_info.get("patterns", [])
            all_patterns.extend(patterns)
        
        # Count pattern occurrences
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Return most common patterns
        sorted_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [p[0] for p in sorted_patterns[:10]]
    
    def _extract_dependencies(self, files: List[Dict[str, Any]]) -> List[str]:
        """Extract dependencies from files.
        
        Args:
            files: List of files
            
        Returns:
            List of dependencies
        """
        all_deps = []
        
        for file_info in files:
            deps = file_info.get("dependencies", [])
            all_deps.extend(deps)
        
        # Deduplicate while preserving order
        seen = set()
        unique_deps = []
        for dep in all_deps:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)
        
        return unique_deps[:20]
    
    def _determine_approach(
        self,
        patterns: List[str],
        story: Dict[str, Any]
    ) -> str:
        """Determine implementation approach.
        
        Args:
            patterns: Identified patterns
            story: Story information
            
        Returns:
            Suggested approach
        """
        approach_parts = []
        
        # Based on patterns
        if "rest-api" in patterns:
            approach_parts.append("Implement RESTful endpoints")
        if "service-layer" in patterns:
            approach_parts.append("Follow service layer pattern")
        if "repository-pattern" in patterns:
            approach_parts.append("Use repository pattern for data access")
        
        # Based on story type
        story_type = story.get("type", "").lower()
        if "api" in story_type:
            approach_parts.append("Create API with validation")
        if "ui" in story_type:
            approach_parts.append("Implement UI components")
        if "integration" in story_type:
            approach_parts.append("Build integration layer")
        
        if not approach_parts:
            approach_parts.append("Follow existing patterns in codebase")
        
        return ". ".join(approach_parts)
    
    def _assess_complexity(
        self,
        files: List[Dict[str, Any]],
        story: Dict[str, Any]
    ) -> str:
        """Assess implementation complexity.
        
        Args:
            files: Selected files
            story: Story information
            
        Returns:
            Complexity assessment
        """
        complexity_score = 0
        
        # File count factor
        file_count = len(files)
        if file_count <= 3:
            complexity_score += 1
        elif file_count <= 7:
            complexity_score += 2
        elif file_count <= 15:
            complexity_score += 3
        else:
            complexity_score += 4
        
        # Story complexity factor
        story_complexity = story.get("complexity", "medium")
        complexity_map = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "very_high": 4
        }
        complexity_score += complexity_map.get(story_complexity, 2)
        
        # Pattern complexity factor
        complex_patterns = ["distributed", "async", "transaction", "security"]
        patterns = self._extract_patterns(files)
        complex_count = sum(1 for p in complex_patterns if p in patterns)
        complexity_score += complex_count
        
        # Determine final complexity
        if complexity_score <= 4:
            return "low"
        elif complexity_score <= 7:
            return "medium"
        elif complexity_score <= 10:
            return "high"
        else:
            return "very_high"