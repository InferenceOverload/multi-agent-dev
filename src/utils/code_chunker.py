"""Intelligent code chunking for large files."""

import ast
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.base import FileContext
from ..utils.logger import get_logger


logger = get_logger(__name__)


class IntelligentFileChunker:
    """Smart file chunking for large insurance system files."""
    
    def __init__(self, chunk_threshold: int = 10000):
        """Initialize file chunker.
        
        Args:
            chunk_threshold: Size threshold for chunking
        """
        self.chunk_threshold = chunk_threshold
    
    async def chunk_large_file(
        self,
        file_context: FileContext,
        story_context: Dict[str, Any]
    ) -> List[FileContext]:
        """Chunk large file based on story relevance.
        
        Args:
            file_context: File to chunk
            story_context: Story context for relevance
            
        Returns:
            List of file chunks
        """
        if file_context.size <= self.chunk_threshold:
            return [file_context]
        
        content = file_context.content
        language = file_context.language
        
        # Create semantic chunks based on language
        if language == "Python":
            chunks = self._chunk_python_file(content, file_context, story_context)
        elif language == "Java":
            chunks = self._chunk_java_file(content, file_context, story_context)
        elif language in ["JavaScript", "TypeScript"]:
            chunks = self._chunk_javascript_file(content, file_context, story_context)
        else:
            chunks = self._chunk_generic_file(content, file_context, story_context)
        
        # Filter chunks by relevance
        relevant_chunks = []
        for chunk in chunks:
            if chunk.relevance_score > 0.3:
                relevant_chunks.append(chunk)
        
        # If no relevant chunks, include the most important parts
        if not relevant_chunks and chunks:
            relevant_chunks = sorted(
                chunks, 
                key=lambda x: x.relevance_score, 
                reverse=True
            )[:3]
        
        return relevant_chunks if relevant_chunks else [file_context]
    
    def _chunk_python_file(
        self,
        content: str,
        file_context: FileContext,
        story_context: Dict[str, Any]
    ) -> List[FileContext]:
        """Chunk Python file by classes and functions.
        
        Args:
            content: File content
            file_context: Original file context
            story_context: Story context
            
        Returns:
            List of chunks
        """
        chunks = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract class content
                    class_content = self._extract_node_content(lines, node)
                    relevance = self._calculate_chunk_relevance(
                        class_content, story_context
                    )
                    
                    chunks.append(FileContext(
                        path=file_context.path,
                        content=class_content,
                        language="Python",
                        size=len(class_content),
                        relevance_score=relevance,
                        chunk_type="class",
                        chunk_name=node.name,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno + 50),
                        dependencies=file_context.dependencies,
                        patterns=self._extract_chunk_patterns(class_content, "Python")
                    ))
                
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    # Top-level function
                    func_content = self._extract_node_content(lines, node)
                    relevance = self._calculate_chunk_relevance(
                        func_content, story_context
                    )
                    
                    chunks.append(FileContext(
                        path=file_context.path,
                        content=func_content,
                        language="Python",
                        size=len(func_content),
                        relevance_score=relevance,
                        chunk_type="function",
                        chunk_name=node.name,
                        line_start=node.lineno,
                        line_end=getattr(node, 'end_lineno', node.lineno + 20),
                        dependencies=file_context.dependencies,
                        patterns=self._extract_chunk_patterns(func_content, "Python")
                    ))
        
        except SyntaxError:
            # Fallback to generic chunking
            return self._chunk_generic_file(content, file_context, story_context)
        
        return chunks if chunks else [file_context]
    
    def _chunk_java_file(
        self,
        content: str,
        file_context: FileContext,
        story_context: Dict[str, Any]
    ) -> List[FileContext]:
        """Chunk Java file by classes and methods.
        
        Args:
            content: File content
            file_context: Original file context
            story_context: Story context
            
        Returns:
            List of chunks
        """
        chunks = []
        lines = content.split('\n')
        
        # Simple regex-based chunking for Java
        class_pattern = re.compile(r'(public|private|protected)?\s*(abstract|final)?\s*class\s+(\w+)')
        method_pattern = re.compile(r'(public|private|protected)?\s*(static|final)?\s*\w+\s+(\w+)\s*\(')
        
        current_class = None
        current_chunk = []
        current_start = 0
        brace_count = 0
        
        for i, line in enumerate(lines):
            # Check for class declaration
            class_match = class_pattern.search(line)
            if class_match:
                # Save previous chunk if exists
                if current_chunk and current_class:
                    chunk_content = '\n'.join(current_chunk)
                    relevance = self._calculate_chunk_relevance(
                        chunk_content, story_context
                    )
                    
                    chunks.append(FileContext(
                        path=file_context.path,
                        content=chunk_content,
                        language="Java",
                        size=len(chunk_content),
                        relevance_score=relevance,
                        chunk_type="class",
                        chunk_name=current_class,
                        line_start=current_start,
                        line_end=i,
                        dependencies=file_context.dependencies,
                        patterns=self._extract_chunk_patterns(chunk_content, "Java")
                    ))
                
                current_class = class_match.group(3)
                current_chunk = [line]
                current_start = i + 1
                brace_count = line.count('{') - line.count('}')
            
            elif current_chunk:
                current_chunk.append(line)
                brace_count += line.count('{') - line.count('}')
                
                # Check if class/method ended
                if brace_count == 0 and len(current_chunk) > 10:
                    chunk_content = '\n'.join(current_chunk)
                    relevance = self._calculate_chunk_relevance(
                        chunk_content, story_context
                    )
                    
                    chunks.append(FileContext(
                        path=file_context.path,
                        content=chunk_content,
                        language="Java",
                        size=len(chunk_content),
                        relevance_score=relevance,
                        chunk_type="class" if current_class else "block",
                        chunk_name=current_class or "block",
                        line_start=current_start,
                        line_end=i + 1,
                        dependencies=file_context.dependencies,
                        patterns=self._extract_chunk_patterns(chunk_content, "Java")
                    ))
                    
                    current_chunk = []
                    current_class = None
        
        # Save last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            relevance = self._calculate_chunk_relevance(
                chunk_content, story_context
            )
            
            chunks.append(FileContext(
                path=file_context.path,
                content=chunk_content,
                language="Java",
                size=len(chunk_content),
                relevance_score=relevance,
                chunk_type="class" if current_class else "block",
                chunk_name=current_class or "block",
                line_start=current_start,
                line_end=len(lines),
                dependencies=file_context.dependencies,
                patterns=self._extract_chunk_patterns(chunk_content, "Java")
            ))
        
        return chunks if chunks else [file_context]
    
    def _chunk_javascript_file(
        self,
        content: str,
        file_context: FileContext,
        story_context: Dict[str, Any]
    ) -> List[FileContext]:
        """Chunk JavaScript/TypeScript file.
        
        Args:
            content: File content
            file_context: Original file context
            story_context: Story context
            
        Returns:
            List of chunks
        """
        chunks = []
        lines = content.split('\n')
        
        # Patterns for JS/TS
        class_pattern = re.compile(r'class\s+(\w+)')
        function_pattern = re.compile(r'(function\s+(\w+)|const\s+(\w+)\s*=\s*(async\s+)?\()')
        component_pattern = re.compile(r'(export\s+)?(default\s+)?(function|const)\s+(\w+)')
        
        current_chunk = []
        current_name = None
        current_type = None
        current_start = 0
        brace_count = 0
        
        for i, line in enumerate(lines):
            # Check for class
            class_match = class_pattern.search(line)
            function_match = function_pattern.search(line)
            component_match = component_pattern.search(line)
            
            if class_match or function_match or component_match:
                # Save previous chunk
                if current_chunk and current_name:
                    chunk_content = '\n'.join(current_chunk)
                    relevance = self._calculate_chunk_relevance(
                        chunk_content, story_context
                    )
                    
                    chunks.append(FileContext(
                        path=file_context.path,
                        content=chunk_content,
                        language=file_context.language,
                        size=len(chunk_content),
                        relevance_score=relevance,
                        chunk_type=current_type or "block",
                        chunk_name=current_name,
                        line_start=current_start,
                        line_end=i,
                        dependencies=file_context.dependencies,
                        patterns=self._extract_chunk_patterns(
                            chunk_content, file_context.language
                        )
                    ))
                
                # Start new chunk
                if class_match:
                    current_name = class_match.group(1)
                    current_type = "class"
                elif function_match:
                    current_name = function_match.group(2) or function_match.group(3)
                    current_type = "function"
                elif component_match:
                    current_name = component_match.group(4)
                    current_type = "component"
                
                current_chunk = [line]
                current_start = i + 1
                brace_count = line.count('{') - line.count('}')
            
            elif current_chunk:
                current_chunk.append(line)
                brace_count += line.count('{') - line.count('}')
                
                # Check if block ended
                if brace_count == 0 and len(current_chunk) > 5:
                    chunk_content = '\n'.join(current_chunk)
                    relevance = self._calculate_chunk_relevance(
                        chunk_content, story_context
                    )
                    
                    chunks.append(FileContext(
                        path=file_context.path,
                        content=chunk_content,
                        language=file_context.language,
                        size=len(chunk_content),
                        relevance_score=relevance,
                        chunk_type=current_type or "block",
                        chunk_name=current_name or "block",
                        line_start=current_start,
                        line_end=i + 1,
                        dependencies=file_context.dependencies,
                        patterns=self._extract_chunk_patterns(
                            chunk_content, file_context.language
                        )
                    ))
                    
                    current_chunk = []
                    current_name = None
                    current_type = None
        
        return chunks if chunks else [file_context]
    
    def _chunk_generic_file(
        self,
        content: str,
        file_context: FileContext,
        story_context: Dict[str, Any]
    ) -> List[FileContext]:
        """Generic file chunking by size.
        
        Args:
            content: File content
            file_context: Original file context
            story_context: Story context
            
        Returns:
            List of chunks
        """
        chunks = []
        lines = content.split('\n')
        chunk_size = 100  # Lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            relevance = self._calculate_chunk_relevance(
                chunk_content, story_context
            )
            
            chunks.append(FileContext(
                path=file_context.path,
                content=chunk_content,
                language=file_context.language,
                size=len(chunk_content),
                relevance_score=relevance,
                chunk_type="block",
                chunk_name=f"lines_{i+1}_{min(i+chunk_size, len(lines))}",
                line_start=i + 1,
                line_end=min(i + chunk_size, len(lines)),
                dependencies=file_context.dependencies,
                patterns=[]
            ))
        
        return chunks
    
    def _extract_node_content(self, lines: List[str], node: ast.AST) -> str:
        """Extract content for an AST node.
        
        Args:
            lines: File lines
            node: AST node
            
        Returns:
            Node content
        """
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line + 50)
        
        # Ensure valid bounds
        start_line = max(0, start_line)
        end_line = min(len(lines), end_line)
        
        return '\n'.join(lines[start_line:end_line])
    
    def _calculate_chunk_relevance(
        self,
        content: str,
        story_context: Dict[str, Any]
    ) -> float:
        """Calculate chunk relevance to story.
        
        Args:
            content: Chunk content
            story_context: Story context
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        content_lower = content.lower()
        
        # Extract story keywords
        keywords = []
        if "title" in story_context:
            keywords.extend(story_context["title"].lower().split())
        if "description" in story_context:
            keywords.extend(story_context["description"].lower().split()[:20])
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
        
        if not keywords:
            return 0.5  # Default relevance
        
        # Calculate keyword matches
        matches = sum(1 for k in keywords if k in content_lower)
        score = min(matches / len(keywords), 1.0)
        
        # Boost score for specific patterns
        if "class" in content_lower or "def" in content_lower or "function" in content_lower:
            score *= 1.2
        
        # Check for domain-specific terms
        insurance_terms = [
            "policy", "claim", "coverage", "premium", "deductible",
            "underwriting", "risk", "insured", "beneficiary"
        ]
        insurance_matches = sum(1 for term in insurance_terms if term in content_lower)
        if insurance_matches > 0:
            score *= (1 + insurance_matches * 0.1)
        
        return min(score, 1.0)
    
    def _extract_chunk_patterns(self, content: str, language: str) -> List[str]:
        """Extract patterns from chunk.
        
        Args:
            content: Chunk content
            language: Programming language
            
        Returns:
            List of patterns
        """
        patterns = []
        content_lower = content.lower()
        
        # Common patterns
        if "try" in content_lower and ("except" in content_lower or "catch" in content_lower):
            patterns.append("error-handling")
        
        if "async" in content_lower or "await" in content_lower:
            patterns.append("async")
        
        if "test" in content_lower:
            patterns.append("testing")
        
        if "@" in content and language in ["Python", "Java"]:
            patterns.append("decorators" if language == "Python" else "annotations")
        
        # Language-specific patterns
        if language == "Python":
            if "class" in content:
                patterns.append("class-based")
            if "__init__" in content:
                patterns.append("constructor")
            if "@property" in content:
                patterns.append("properties")
        
        elif language == "Java":
            if "@RestController" in content or "@Controller" in content:
                patterns.append("rest-controller")
            if "@Service" in content:
                patterns.append("service")
            if "@Repository" in content:
                patterns.append("repository")
            if "@Entity" in content:
                patterns.append("entity")
        
        elif language in ["JavaScript", "TypeScript"]:
            if "react" in content_lower:
                patterns.append("react")
            if "component" in content_lower:
                patterns.append("component")
            if "express" in content_lower:
                patterns.append("express")
            if "mongoose" in content_lower:
                patterns.append("mongoose")
        
        return patterns