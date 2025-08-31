"""Repository Analysis Agent for large codebase understanding."""

import os
import ast
import asyncio
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import git
import lizard
import radon.complexity as radon_cc
import radon.metrics as radon_metrics

from ..agents.base_agent import HartfordBaseAgent
from ..models.base import (
    RepositoryInfo, FileContext, CodePattern,
    BusinessDomain, ComplexityLevel
)
from ..config.config import get_config, get_domain_config
from ..utils.logger import get_logger
from ..utils.context_manager import ContextWindowManager
from ..utils.code_chunker import IntelligentFileChunker


logger = get_logger(__name__)


class RepositoryAnalystAgent(HartfordBaseAgent):
    """Agent for analyzing large insurance codebases with intelligent context management."""
    
    ARCHITECTURE_PATTERNS = {
        "microservices": {
            "indicators": ["service", "api", "gateway", "docker", "kubernetes"],
            "file_patterns": ["*Service.java", "*Controller.java", "Dockerfile"]
        },
        "monolithic": {
            "indicators": ["app", "application", "main", "server"],
            "file_patterns": ["Application.java", "app.py", "server.js"]
        },
        "layered": {
            "indicators": ["controller", "service", "repository", "model"],
            "file_patterns": ["*Controller.*", "*Service.*", "*Repository.*"]
        },
        "event_driven": {
            "indicators": ["event", "handler", "listener", "publisher", "subscriber"],
            "file_patterns": ["*Event.*", "*Handler.*", "*Listener.*"]
        }
    }
    
    def __init__(self, **kwargs):
        """Initialize Repository Analyst Agent."""
        super().__init__(
            name="RepositoryAnalyst",
            description=(
                "Analyzes large insurance codebases to understand architecture, "
                "patterns, and context for story implementation. "
                "Specializes in incremental analysis and context compression."
            ),
            **kwargs
        )
        self.config = get_config()
        self.domain_config = get_domain_config()
        self.context_manager = ContextWindowManager(
            max_context_size=self.config.max_context_length
        )
        self.file_chunker = IntelligentFileChunker()
        self.repo_cache = {}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process repository for story-specific context.
        
        Args:
            input_data: Contains 'repo_url', 'story_context', optional 'branch'
            
        Returns:
            Repository analysis and implementation context
        """
        repo_url = input_data.get("repo_url")
        story_context = input_data.get("story_context", {})
        branch = input_data.get("branch", "main")
        
        # Check cache first
        cache_key = self._generate_cache_key(repo_url, story_context)
        if cache_key in self.repo_cache:
            logger.info(f"Using cached analysis for {repo_url}")
            cached = self.repo_cache[cache_key]
            # Update with story-specific context
            cached["implementation_context"] = await self._build_implementation_context(
                cached["repository_info"],
                cached["analysis"],
                story_context
            )
            return cached
        
        # Stage 1: Quick Repository Discovery
        logger.info(f"Stage 1: Quick repository discovery for {repo_url}")
        repo_info = await self._quick_repository_scan(repo_url, branch)
        
        # Stage 2: Business Domain Mapping
        logger.info("Stage 2: Business domain mapping")
        domain_context = await self._map_business_domains(repo_info, story_context)
        
        # Stage 3: Focused Context Retrieval
        logger.info("Stage 3: Story-relevant context retrieval")
        relevant_context = await self._retrieve_story_relevant_context(
            repo_info, story_context, domain_context
        )
        
        # Stage 4: Implementation Context Building
        logger.info("Stage 4: Building implementation context")
        implementation_context = await self._build_implementation_context(
            repo_info, relevant_context, story_context
        )
        
        # Stage 5: Pattern Recognition
        logger.info("Stage 5: Pattern recognition and learning")
        patterns = await self._identify_patterns(relevant_context)
        
        result = {
            "repository_info": repo_info.dict(),
            "analysis": {
                "business_domains": domain_context,
                "relevant_files": relevant_context,
                "patterns": patterns,
                "architecture_type": repo_info.architecture_type,
                "complexity_metrics": await self._calculate_complexity_metrics(
                    relevant_context
                )
            },
            "implementation_context": implementation_context,
            "recommendations": await self._generate_recommendations(
                implementation_context, patterns
            )
        }
        
        # Cache the result
        self.repo_cache[cache_key] = result
        
        # Store in long-term memory
        await self.save_to_memory(
            f"repo_analysis_{cache_key}",
            result,
            "long_term"
        )
        
        return result
    
    def _generate_cache_key(self, repo_url: str, story_context: Dict) -> str:
        """Generate cache key for repository analysis.
        
        Args:
            repo_url: Repository URL
            story_context: Story context
            
        Returns:
            Cache key
        """
        context_str = str(sorted(story_context.items()))
        combined = f"{repo_url}_{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _quick_repository_scan(
        self,
        repo_url: str,
        branch: str
    ) -> RepositoryInfo:
        """Perform quick repository scan for metadata.
        
        Args:
            repo_url: Repository URL
            branch: Branch to analyze
            
        Returns:
            Repository information
        """
        # Clone repository (shallow clone for speed)
        repo_path = await self._clone_repository(repo_url, branch, shallow=True)
        
        # Gather basic statistics
        file_stats = await self._gather_file_statistics(repo_path)
        language_stats = await self._analyze_languages(repo_path)
        architecture_type = await self._detect_architecture(repo_path)
        framework = await self._detect_framework(repo_path)
        
        repo_info = RepositoryInfo(
            url=repo_url,
            name=Path(repo_url).stem,
            organization=self._extract_organization(repo_url),
            default_branch=branch,
            language_stats=language_stats,
            total_files=file_stats["total_files"],
            total_lines=file_stats["total_lines"],
            architecture_type=architecture_type,
            framework=framework,
            last_analyzed=datetime.utcnow()
        )
        
        return repo_info
    
    async def _clone_repository(
        self,
        repo_url: str,
        branch: str,
        shallow: bool = True
    ) -> Path:
        """Clone repository for analysis.
        
        Args:
            repo_url: Repository URL
            branch: Branch to clone
            shallow: Whether to do shallow clone
            
        Returns:
            Path to cloned repository
        """
        # Create temporary directory for clone
        repo_name = Path(repo_url).stem
        clone_path = Path(f"/tmp/hartford_repos/{repo_name}_{branch}")
        
        if clone_path.exists():
            # Pull latest changes if already cloned
            repo = git.Repo(clone_path)
            origin = repo.remotes.origin
            origin.pull()
        else:
            # Clone repository
            clone_path.parent.mkdir(parents=True, exist_ok=True)
            if shallow:
                git.Repo.clone_from(
                    repo_url, 
                    clone_path,
                    branch=branch,
                    depth=1
                )
            else:
                git.Repo.clone_from(repo_url, clone_path, branch=branch)
        
        return clone_path
    
    async def _gather_file_statistics(self, repo_path: Path) -> Dict[str, int]:
        """Gather basic file statistics.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            File statistics
        """
        total_files = 0
        total_lines = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
                'node_modules', 'venv', 'env', '__pycache__', 'build', 'dist'
            ]]
            
            for file in files:
                if self._is_code_file(file):
                    total_files += 1
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            total_lines += sum(1 for _ in f)
                    except:
                        pass
        
        return {"total_files": total_files, "total_lines": total_lines}
    
    def _is_code_file(self, filename: str) -> bool:
        """Check if file is a code file.
        
        Args:
            filename: File name
            
        Returns:
            True if code file
        """
        code_extensions = [
            '.py', '.java', '.js', '.ts', '.jsx', '.tsx',
            '.cs', '.cpp', '.c', '.h', '.go', '.rs',
            '.sql', '.xml', '.json', '.yaml', '.yml'
        ]
        return any(filename.endswith(ext) for ext in code_extensions)
    
    async def _analyze_languages(self, repo_path: Path) -> Dict[str, float]:
        """Analyze language distribution in repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Language distribution percentages
        """
        language_lines = {}
        total_lines = 0
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
                'node_modules', 'venv', 'env', '__pycache__', 'build', 'dist'
            ]]
            
            for file in files:
                if self._is_code_file(file):
                    language = self._detect_language(file)
                    if language:
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = sum(1 for _ in f)
                                language_lines[language] = language_lines.get(language, 0) + lines
                                total_lines += lines
                        except:
                            pass
        
        # Convert to percentages
        if total_lines > 0:
            return {
                lang: (lines / total_lines) * 100
                for lang, lines in language_lines.items()
            }
        return {}
    
    def _detect_language(self, filename: str) -> Optional[str]:
        """Detect programming language from filename.
        
        Args:
            filename: File name
            
        Returns:
            Language name or None
        """
        extension_map = {
            '.py': 'Python',
            '.java': 'Java',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React TypeScript',
            '.cs': 'C#',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.sql': 'SQL',
            '.xml': 'XML',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML'
        }
        
        for ext, lang in extension_map.items():
            if filename.endswith(ext):
                return lang
        return None
    
    async def _detect_architecture(self, repo_path: Path) -> str:
        """Detect repository architecture type.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Architecture type
        """
        architecture_scores = {}
        
        for arch_type, config in self.ARCHITECTURE_PATTERNS.items():
            score = 0
            
            # Check for indicator keywords in file names
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    file_lower = file.lower()
                    for indicator in config["indicators"]:
                        if indicator in file_lower:
                            score += 1
                    
                    # Check file patterns
                    for pattern in config["file_patterns"]:
                        if self._matches_pattern(file, pattern):
                            score += 2
            
            architecture_scores[arch_type] = score
        
        # Return architecture with highest score
        if architecture_scores:
            best_arch = max(architecture_scores.items(), key=lambda x: x[1])
            if best_arch[1] > 0:
                return best_arch[0]
        
        return "unknown"
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern.
        
        Args:
            filename: File name
            pattern: Pattern to match (supports * wildcard)
            
        Returns:
            True if matches
        """
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    async def _detect_framework(self, repo_path: Path) -> Optional[str]:
        """Detect framework used in repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Framework name or None
        """
        framework_files = {
            "Spring Boot": ["pom.xml", "build.gradle"],
            "Django": ["manage.py", "settings.py"],
            "Flask": ["app.py", "flask"],
            "Express": ["package.json", "express"],
            "React": ["package.json", "react"],
            "Angular": ["angular.json", "angular"],
            "Vue": ["vue.config.js", "vue"],
            ".NET": ["*.csproj", "*.sln"],
            "Rails": ["Gemfile", "rails"]
        }
        
        for framework, indicators in framework_files.items():
            for indicator in indicators:
                if "*" in indicator:
                    # Pattern matching
                    for root, dirs, files in os.walk(repo_path):
                        for file in files:
                            if self._matches_pattern(file, indicator):
                                return framework
                else:
                    # Exact file or content check
                    if (repo_path / indicator).exists():
                        return framework
                    
                    # Check package.json for JS frameworks
                    if framework in ["Express", "React"] and (repo_path / "package.json").exists():
                        try:
                            import json
                            with open(repo_path / "package.json", 'r') as f:
                                package = json.load(f)
                                deps = package.get("dependencies", {})
                                dev_deps = package.get("devDependencies", {})
                                if indicator.lower() in str(deps).lower() or indicator.lower() in str(dev_deps).lower():
                                    return framework
                        except:
                            pass
        
        return None
    
    def _extract_organization(self, repo_url: str) -> str:
        """Extract organization from repository URL.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            Organization name
        """
        # Parse GitHub/GitLab style URLs
        parts = repo_url.replace("https://", "").replace("http://", "").split("/")
        if len(parts) >= 2:
            return parts[1]
        return "unknown"
    
    async def _map_business_domains(
        self,
        repo_info: RepositoryInfo,
        story_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map repository to business domains.
        
        Args:
            repo_info: Repository information
            story_context: Story context
            
        Returns:
            Business domain mapping
        """
        # Clone full repository for detailed analysis
        repo_path = await self._clone_repository(
            repo_info.url,
            repo_info.default_branch,
            shallow=False
        )
        
        domain_mapping = {}
        
        for domain_name, config in self.domain_config.DOMAINS.items():
            domain_files = []
            domain_score = 0
            
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if self._is_code_file(file):
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(repo_path)
                        
                        # Check if file matches domain patterns
                        for pattern in config["file_patterns"]:
                            if self._matches_pattern(str(relative_path), pattern):
                                domain_files.append(str(relative_path))
                                domain_score += 2
                                break
                        
                        # Check file content for domain keywords
                        if domain_score < 10:  # Limit deep content analysis
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read(1000)  # Read first 1000 chars
                                    content_lower = content.lower()
                                    for keyword in config["keywords"]:
                                        if keyword in content_lower:
                                            domain_files.append(str(relative_path))
                                            domain_score += 1
                                            break
                            except:
                                pass
            
            if domain_files:
                domain_mapping[domain_name] = {
                    "files": domain_files[:20],  # Limit to top 20 files
                    "score": domain_score,
                    "config": config
                }
        
        # Identify primary domain based on story context
        story_domain = story_context.get("domain")
        if story_domain and story_domain in domain_mapping:
            domain_mapping[story_domain]["primary"] = True
        elif domain_mapping:
            # Set highest scoring domain as primary
            best_domain = max(domain_mapping.items(), key=lambda x: x[1]["score"])
            domain_mapping[best_domain[0]]["primary"] = True
        
        return domain_mapping
    
    async def _retrieve_story_relevant_context(
        self,
        repo_info: RepositoryInfo,
        story_context: Dict[str, Any],
        domain_context: Dict[str, Any]
    ) -> List[FileContext]:
        """Retrieve story-relevant context from repository.
        
        Args:
            repo_info: Repository information
            story_context: Story context
            domain_context: Business domain mapping
            
        Returns:
            List of relevant file contexts
        """
        repo_path = await self._clone_repository(
            repo_info.url,
            repo_info.default_branch,
            shallow=False
        )
        
        relevant_files = []
        story_keywords = self._extract_story_keywords(story_context)
        
        # Get primary domain files first
        primary_domain = None
        for domain_name, domain_info in domain_context.items():
            if domain_info.get("primary"):
                primary_domain = domain_name
                for file_path in domain_info["files"][:10]:
                    full_path = repo_path / file_path
                    if full_path.exists():
                        file_context = await self._create_file_context(
                            full_path, repo_path, story_keywords
                        )
                        if file_context:
                            relevant_files.append(file_context)
        
        # Add files based on story keywords
        keyword_files = await self._find_keyword_relevant_files(
            repo_path, story_keywords, limit=10
        )
        for file_path in keyword_files:
            file_context = await self._create_file_context(
                file_path, repo_path, story_keywords
            )
            if file_context and file_context not in relevant_files:
                relevant_files.append(file_context)
        
        # Sort by relevance score
        relevant_files.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply intelligent chunking for large files
        chunked_files = []
        for file_context in relevant_files[:20]:  # Limit to top 20 files
            if file_context.size > 10000:
                chunks = await self.file_chunker.chunk_large_file(
                    file_context, story_context
                )
                chunked_files.extend(chunks)
            else:
                chunked_files.append(file_context)
        
        return chunked_files
    
    def _extract_story_keywords(self, story_context: Dict[str, Any]) -> List[str]:
        """Extract keywords from story context.
        
        Args:
            story_context: Story context
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Extract from title and description
        if "title" in story_context:
            keywords.extend(story_context["title"].lower().split())
        if "description" in story_context:
            keywords.extend(story_context["description"].lower().split()[:20])
        
        # Extract from user story
        if "user_story" in story_context:
            keywords.extend(story_context["user_story"].lower().split()[:10])
        
        # Extract from technical areas
        if "technical_areas" in story_context:
            keywords.extend(story_context["technical_areas"])
        
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        keywords = [k for k in keywords if k not in stop_words and len(k) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique_keywords.append(k)
        
        return unique_keywords[:15]  # Limit to top 15 keywords
    
    async def _find_keyword_relevant_files(
        self,
        repo_path: Path,
        keywords: List[str],
        limit: int = 10
    ) -> List[Path]:
        """Find files relevant to keywords.
        
        Args:
            repo_path: Path to repository
            keywords: List of keywords
            limit: Maximum number of files
            
        Returns:
            List of file paths
        """
        file_scores = {}
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
                'node_modules', 'venv', 'env', '__pycache__', 'build', 'dist'
            ]]
            
            for file in files:
                if self._is_code_file(file):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(repo_path)
                    
                    # Score based on filename
                    score = 0
                    file_lower = file.lower()
                    for keyword in keywords:
                        if keyword in file_lower:
                            score += 2
                    
                    # Score based on path
                    path_lower = str(relative_path).lower()
                    for keyword in keywords:
                        if keyword in path_lower:
                            score += 1
                    
                    if score > 0:
                        file_scores[file_path] = score
        
        # Return top scoring files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_files[:limit]]
    
    async def _create_file_context(
        self,
        file_path: Path,
        repo_path: Path,
        keywords: List[str]
    ) -> Optional[FileContext]:
        """Create file context object.
        
        Args:
            file_path: Path to file
            repo_path: Repository root path
            keywords: Story keywords
            
        Returns:
            File context or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            relative_path = file_path.relative_to(repo_path)
            language = self._detect_language(file_path.name)
            
            # Calculate relevance score
            relevance_score = self._calculate_file_relevance(
                str(relative_path), content, keywords
            )
            
            # Extract patterns
            patterns = self._extract_file_patterns(content, language)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(content, language)
            
            return FileContext(
                path=str(relative_path),
                content=content[:50000],  # Limit content size
                language=language or "unknown",
                size=len(content),
                relevance_score=relevance_score,
                patterns=patterns,
                dependencies=dependencies
            )
        except Exception as e:
            logger.error(f"Error creating file context for {file_path}: {e}")
            return None
    
    def _calculate_file_relevance(
        self,
        path: str,
        content: str,
        keywords: List[str]
    ) -> float:
        """Calculate file relevance score.
        
        Args:
            path: File path
            content: File content
            keywords: Story keywords
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        max_score = len(keywords) * 3  # Maximum possible score
        
        path_lower = path.lower()
        content_lower = content[:5000].lower()  # Check first 5000 chars
        
        for keyword in keywords:
            # Path matching (higher weight)
            if keyword in path_lower:
                score += 2.0
            
            # Content matching
            if keyword in content_lower:
                score += 1.0
        
        # Normalize to 0-1 range
        if max_score > 0:
            return min(score / max_score, 1.0)
        return 0.0
    
    def _extract_file_patterns(self, content: str, language: str) -> List[str]:
        """Extract code patterns from file.
        
        Args:
            content: File content
            language: Programming language
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        if language == "Python":
            # Check for common Python patterns
            if "class " in content:
                patterns.append("class-based")
            if "def __init__" in content:
                patterns.append("constructor")
            if "@property" in content:
                patterns.append("properties")
            if "async def" in content:
                patterns.append("async")
            if "unittest" in content or "pytest" in content:
                patterns.append("testing")
        
        elif language == "Java":
            # Check for common Java patterns
            if "@RestController" in content or "@Controller" in content:
                patterns.append("rest-api")
            if "@Service" in content:
                patterns.append("service-layer")
            if "@Repository" in content:
                patterns.append("repository-pattern")
            if "@Entity" in content:
                patterns.append("jpa-entity")
            if "@Test" in content:
                patterns.append("testing")
        
        # Common patterns across languages
        if "try" in content and ("catch" in content or "except" in content):
            patterns.append("error-handling")
        if "log" in content.lower():
            patterns.append("logging")
        if "validate" in content.lower() or "validation" in content.lower():
            patterns.append("validation")
        
        return patterns
    
    def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies from file.
        
        Args:
            content: File content
            language: Programming language
            
        Returns:
            List of dependencies
        """
        dependencies = []
        
        if language == "Python":
            # Extract imports
            import_lines = [line for line in content.split('\n') if line.startswith('import ') or line.startswith('from ')]
            for line in import_lines[:20]:  # Limit to first 20 imports
                parts = line.split()
                if len(parts) > 1:
                    dependencies.append(parts[1].split('.')[0])
        
        elif language == "Java":
            # Extract imports
            import_lines = [line for line in content.split('\n') if line.startswith('import ')]
            for line in import_lines[:20]:
                parts = line.split()
                if len(parts) > 1:
                    dependencies.append(parts[1].split('.')[0])
        
        elif language in ["JavaScript", "TypeScript"]:
            # Extract imports/requires
            import_lines = [line for line in content.split('\n') if 'import ' in line or 'require(' in line]
            for line in import_lines[:20]:
                if 'from' in line:
                    parts = line.split('from')
                    if len(parts) > 1:
                        dep = parts[1].strip().strip(';').strip('"').strip("'")
                        dependencies.append(dep)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_deps = []
        for dep in dependencies:
            if dep not in seen:
                seen.add(dep)
                unique_deps.append(dep)
        
        return unique_deps
    
    async def _build_implementation_context(
        self,
        repo_info: RepositoryInfo,
        relevant_context: List[FileContext],
        story_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build implementation context for story.
        
        Args:
            repo_info: Repository information
            relevant_context: Relevant file contexts
            story_context: Story context
            
        Returns:
            Implementation context
        """
        # Compress context to fit window
        compressed_context = await self.context_manager.build_story_context(
            story_context, {"files": relevant_context}
        )
        
        # Identify implementation approach
        approach = self._suggest_implementation_approach(
            repo_info, compressed_context, story_context
        )
        
        # Identify test strategy
        test_strategy = self._suggest_test_strategy(
            repo_info, compressed_context
        )
        
        # Assess implementation complexity
        complexity = self._assess_implementation_complexity(
            compressed_context, story_context
        )
        
        return {
            "relevant_files": compressed_context["files"],
            "suggested_approach": approach,
            "test_strategy": test_strategy,
            "estimated_complexity": complexity,
            "key_patterns": compressed_context.get("patterns", []),
            "dependencies": compressed_context.get("dependencies", []),
            "integration_points": self._identify_integration_points(
                compressed_context
            )
        }
    
    def _suggest_implementation_approach(
        self,
        repo_info: RepositoryInfo,
        context: Dict[str, Any],
        story_context: Dict[str, Any]
    ) -> str:
        """Suggest implementation approach.
        
        Args:
            repo_info: Repository information
            context: Compressed context
            story_context: Story context
            
        Returns:
            Suggested approach
        """
        approach_parts = []
        
        # Based on architecture
        if repo_info.architecture_type == "microservices":
            approach_parts.append("Implement as a new service endpoint or extend existing service")
        elif repo_info.architecture_type == "layered":
            approach_parts.append("Follow layered architecture: Controller -> Service -> Repository")
        
        # Based on framework
        if repo_info.framework:
            approach_parts.append(f"Use {repo_info.framework} conventions and patterns")
        
        # Based on story type
        if "api" in str(story_context).lower():
            approach_parts.append("Create RESTful API endpoints with proper validation")
        if "ui" in str(story_context).lower() or "frontend" in str(story_context).lower():
            approach_parts.append("Implement UI components with responsive design")
        
        # Based on patterns found
        patterns = context.get("patterns", [])
        if "repository-pattern" in patterns:
            approach_parts.append("Use repository pattern for data access")
        if "service-layer" in patterns:
            approach_parts.append("Implement business logic in service layer")
        
        return ". ".join(approach_parts) if approach_parts else "Standard implementation following existing patterns"
    
    def _suggest_test_strategy(
        self,
        repo_info: RepositoryInfo,
        context: Dict[str, Any]
    ) -> str:
        """Suggest testing strategy.
        
        Args:
            repo_info: Repository information
            context: Compressed context
            
        Returns:
            Test strategy
        """
        test_parts = []
        
        # Check for existing test frameworks
        files = context.get("files", [])
        has_unit_tests = any("test" in f.get("path", "").lower() for f in files)
        
        if has_unit_tests:
            test_parts.append("Follow existing test patterns and structure")
        
        # Language-specific recommendations
        if "Python" in repo_info.language_stats:
            test_parts.append("Write pytest/unittest tests with mocking")
        elif "Java" in repo_info.language_stats:
            test_parts.append("Write JUnit tests with Mockito for mocking")
        elif "JavaScript" in repo_info.language_stats:
            test_parts.append("Write Jest/Mocha tests with appropriate mocking")
        
        # General recommendations
        test_parts.append("Include unit tests for business logic")
        test_parts.append("Add integration tests for API endpoints")
        test_parts.append("Ensure 80%+ code coverage")
        
        return ". ".join(test_parts)
    
    def _assess_implementation_complexity(
        self,
        context: Dict[str, Any],
        story_context: Dict[str, Any]
    ) -> str:
        """Assess implementation complexity.
        
        Args:
            context: Compressed context
            story_context: Story context
            
        Returns:
            Complexity assessment
        """
        complexity_score = 0
        
        # Based on number of files to modify
        files_count = len(context.get("files", []))
        if files_count <= 2:
            complexity_score += 1
        elif files_count <= 5:
            complexity_score += 2
        elif files_count <= 10:
            complexity_score += 3
        else:
            complexity_score += 4
        
        # Based on dependencies
        dependencies = context.get("dependencies", [])
        if len(dependencies) > 10:
            complexity_score += 2
        elif len(dependencies) > 5:
            complexity_score += 1
        
        # Based on story complexity
        story_complexity = story_context.get("complexity")
        if story_complexity == "very_high":
            complexity_score += 3
        elif story_complexity == "high":
            complexity_score += 2
        elif story_complexity == "medium":
            complexity_score += 1
        
        # Determine final complexity
        if complexity_score <= 3:
            return "low"
        elif complexity_score <= 6:
            return "medium"
        elif complexity_score <= 9:
            return "high"
        else:
            return "very_high"
    
    def _identify_integration_points(self, context: Dict[str, Any]) -> List[str]:
        """Identify integration points in the code.
        
        Args:
            context: Compressed context
            
        Returns:
            List of integration points
        """
        integration_points = []
        
        files = context.get("files", [])
        for file_context in files:
            content = file_context.get("content", "")
            
            # Check for API calls
            if "http" in content.lower() or "request" in content.lower():
                integration_points.append("HTTP/REST APIs")
            
            # Check for database operations
            if "select" in content.lower() or "insert" in content.lower():
                integration_points.append("Database")
            
            # Check for messaging
            if "queue" in content.lower() or "publish" in content.lower():
                integration_points.append("Message Queue")
            
            # Check for file operations
            if "file" in content.lower() or "fs." in content:
                integration_points.append("File System")
        
        # Remove duplicates
        return list(set(integration_points))
    
    async def _identify_patterns(
        self,
        relevant_context: List[FileContext]
    ) -> List[CodePattern]:
        """Identify code patterns in relevant files.
        
        Args:
            relevant_context: List of file contexts
            
        Returns:
            List of identified patterns
        """
        patterns = []
        pattern_counts = {}
        
        for file_context in relevant_context:
            file_patterns = file_context.patterns
            
            for pattern_name in file_patterns:
                if pattern_name not in pattern_counts:
                    pattern_counts[pattern_name] = {
                        "count": 0,
                        "files": []
                    }
                pattern_counts[pattern_name]["count"] += 1
                pattern_counts[pattern_name]["files"].append(file_context.path)
        
        # Create pattern objects
        for pattern_name, info in pattern_counts.items():
            patterns.append(CodePattern(
                name=pattern_name,
                type="structural",
                description=f"Pattern found in {info['count']} files",
                file_path=info["files"][0],
                line_range=(0, 0),  # Would need detailed analysis for exact lines
                confidence=min(info["count"] / len(relevant_context), 1.0),
                usage_count=info["count"]
            ))
        
        return patterns
    
    async def _calculate_complexity_metrics(
        self,
        relevant_context: List[FileContext]
    ) -> Dict[str, Any]:
        """Calculate complexity metrics for relevant files.
        
        Args:
            relevant_context: List of file contexts
            
        Returns:
            Complexity metrics
        """
        total_complexity = 0
        file_complexities = []
        
        for file_context in relevant_context[:10]:  # Limit to top 10 files
            if file_context.language == "Python":
                try:
                    # Use radon for Python complexity
                    complexity = radon_cc.cc_visit(file_context.content)
                    file_complexity = sum(block.complexity for block in complexity)
                    file_complexities.append({
                        "file": file_context.path,
                        "complexity": file_complexity,
                        "language": "Python"
                    })
                    total_complexity += file_complexity
                except:
                    pass
            
            # Use lizard for other languages
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix=file_context.path.split('.')[-1], delete=False) as tmp:
                    tmp.write(file_context.content)
                    tmp.flush()
                    
                    analysis = lizard.analyze_file(tmp.name)
                    file_complexity = analysis.average_cyclomatic_complexity
                    file_complexities.append({
                        "file": file_context.path,
                        "complexity": file_complexity,
                        "language": file_context.language
                    })
                    total_complexity += file_complexity
            except:
                pass
        
        avg_complexity = total_complexity / len(file_complexities) if file_complexities else 0
        
        return {
            "average_complexity": avg_complexity,
            "total_complexity": total_complexity,
            "file_complexities": file_complexities,
            "complexity_rating": self._rate_complexity(avg_complexity)
        }
    
    def _rate_complexity(self, avg_complexity: float) -> str:
        """Rate complexity level.
        
        Args:
            avg_complexity: Average cyclomatic complexity
            
        Returns:
            Complexity rating
        """
        if avg_complexity <= 5:
            return "low"
        elif avg_complexity <= 10:
            return "medium"
        elif avg_complexity <= 20:
            return "high"
        else:
            return "very_high"
    
    async def _generate_recommendations(
        self,
        implementation_context: Dict[str, Any],
        patterns: List[CodePattern]
    ) -> List[str]:
        """Generate implementation recommendations.
        
        Args:
            implementation_context: Implementation context
            patterns: Identified patterns
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Based on complexity
        complexity = implementation_context.get("estimated_complexity")
        if complexity in ["high", "very_high"]:
            recommendations.append("Break down implementation into smaller, incremental changes")
            recommendations.append("Consider pair programming or extra code review")
        
        # Based on patterns
        pattern_names = [p.name for p in patterns]
        if "testing" not in pattern_names:
            recommendations.append("Establish testing patterns if not present")
        if "error-handling" not in pattern_names:
            recommendations.append("Implement comprehensive error handling")
        if "logging" not in pattern_names:
            recommendations.append("Add appropriate logging for debugging and monitoring")
        
        # Based on integration points
        integration_points = implementation_context.get("integration_points", [])
        if "Database" in integration_points:
            recommendations.append("Use transactions for database operations")
        if "HTTP/REST APIs" in integration_points:
            recommendations.append("Implement retry logic and error handling for API calls")
        if "Message Queue" in integration_points:
            recommendations.append("Ensure idempotent message processing")
        
        # General recommendations
        recommendations.append("Follow existing code style and conventions")
        recommendations.append("Update documentation and README if needed")
        recommendations.append("Consider performance implications for large datasets")
        
        return recommendations[:7]  # Limit to top 7 recommendations