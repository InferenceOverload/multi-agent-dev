"""Discovery Agent - Identifies codebase type and architecture."""

from typing import Dict, Any, List, Optional
import os
import json
from pathlib import Path
from ..tools.core_tools import CoreTools

class DiscoveryAgent:
    """Agent that discovers what type of codebase we're dealing with."""
    
    def __init__(self):
        self.core_tools = CoreTools()
        
        # Detection patterns for different tech stacks
        self.stack_patterns = {
            "frontend": {
                "react": ["package.json", "node_modules", ".jsx", ".tsx", "react"],
                "vue": ["vue.config.js", ".vue", "nuxt.config"],
                "angular": ["angular.json", ".component.ts", "@angular"],
            },
            "backend": {
                "node": ["package.json", "server.js", "app.js", "express"],
                "python": ["requirements.txt", "setup.py", "Pipfile", ".py"],
                "java": ["pom.xml", "build.gradle", ".java", "src/main/java"],
                "go": ["go.mod", "go.sum", ".go"],
                "dotnet": [".csproj", ".sln", "Program.cs"],
            },
            "data": {
                "sql": [".sql", "migrations", "schema.sql"],
                "spark": ["spark-submit", "pyspark", "SparkContext"],
                "airflow": ["dags/", "airflow.cfg", "DAG("],
                "dbt": ["dbt_project.yml", "models/", "macros/"],
            },
            "infrastructure": {
                "terraform": [".tf", "terraform.tfstate", "provider.tf"],
                "kubernetes": [".yaml", "deployment.yaml", "service.yaml", "kubectl"],
                "docker": ["Dockerfile", "docker-compose.yml", ".dockerignore"],
                "ansible": ["playbook.yml", "inventory", "ansible.cfg"],
            },
            "mobile": {
                "ios": [".swift", "Info.plist", ".xcodeproj"],
                "android": ["AndroidManifest.xml", ".gradle", ".kt", ".java"],
                "react_native": ["react-native", "metro.config.js"],
                "flutter": ["pubspec.yaml", ".dart", "flutter"],
            },
            "ml": {
                "tensorflow": ["tensorflow", ".h5", "model.save"],
                "pytorch": ["torch", ".pth", "model.state_dict"],
                "sklearn": ["sklearn", "joblib", ".pkl"],
            }
        }
    
    async def discover(self, repo_path: str) -> Dict[str, Any]:
        """
        Discover the type of codebase and its characteristics.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Discovery results including tech stack, architecture, and more
        """
        results = {
            "path": repo_path,
            "tech_stack": {},
            "primary_language": None,
            "architecture_type": None,
            "frameworks": [],
            "has_tests": False,
            "has_ci_cd": False,
            "has_docker": False,
            "package_managers": [],
            "entry_points": [],
            "config_files": [],
            "size_category": None
        }
        
        # Get directory structure
        structure = await self.core_tools.get_directory_structure(repo_path, max_depth=2)
        
        # Analyze file extensions
        file_stats = await self._analyze_file_types(repo_path)
        results["file_statistics"] = file_stats
        
        # Detect tech stack
        results["tech_stack"] = await self._detect_tech_stack(repo_path)
        
        # Determine primary technology
        results["primary_language"] = self._determine_primary_language(file_stats)
        
        # Detect architecture type
        results["architecture_type"] = await self._detect_architecture(repo_path, structure)
        
        # Find frameworks
        results["frameworks"] = await self._detect_frameworks(repo_path)
        
        # Check for tests
        results["has_tests"] = await self._has_tests(repo_path)
        
        # Check for CI/CD
        results["has_ci_cd"] = await self._has_ci_cd(repo_path)
        
        # Find entry points
        results["entry_points"] = await self._find_entry_points(repo_path, results["primary_language"])
        
        # Categorize size
        results["size_category"] = self._categorize_size(file_stats.get("total_files", 0))
        
        return results
    
    async def _analyze_file_types(self, repo_path: str) -> Dict[str, Any]:
        """Analyze file types and counts."""
        stats = {
            "total_files": 0,
            "by_extension": {},
            "total_lines": 0
        }
        
        for root, _, files in os.walk(repo_path):
            # Skip common ignore directories
            if any(ignore in root for ignore in ['node_modules', '.git', '__pycache__']):
                continue
                
            for file in files:
                stats["total_files"] += 1
                ext = Path(file).suffix
                if ext:
                    stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
                    
        return stats
    
    async def _detect_tech_stack(self, repo_path: str) -> Dict[str, List[str]]:
        """Detect technology stack."""
        detected = {}
        
        for category, patterns in self.stack_patterns.items():
            detected[category] = []
            for tech, indicators in patterns.items():
                for indicator in indicators:
                    # Check if indicator exists (file, pattern, etc.)
                    if await self._check_indicator(repo_path, indicator):
                        if tech not in detected[category]:
                            detected[category].append(tech)
                        break
                        
        return detected
    
    async def _check_indicator(self, repo_path: str, indicator: str) -> bool:
        """Check if an indicator exists in the repo."""
        # Check if it's a file/directory
        path = Path(repo_path) / indicator
        if path.exists():
            return True
            
        # Check if it's a file extension
        if indicator.startswith('.'):
            files = await self.core_tools.find_files_by_extension(repo_path, [indicator], max_files=1)
            return len(files) > 0
            
        # Check if it's a pattern in files - escape special regex characters
        import re
        escaped_indicator = re.escape(indicator)
        results = await self.core_tools.search_pattern(repo_path, escaped_indicator, max_results=1)
        return len(results) > 0
    
    def _determine_primary_language(self, file_stats: Dict) -> Optional[str]:
        """Determine primary programming language."""
        language_extensions = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "go": [".go"],
            "csharp": [".cs"],
            "sql": [".sql"],
            "swift": [".swift"],
            "kotlin": [".kt"],
            "rust": [".rs"],
            "cpp": [".cpp", ".cc", ".cxx"],
        }
        
        language_counts = {}
        by_ext = file_stats.get("by_extension", {})
        
        for lang, extensions in language_extensions.items():
            count = sum(by_ext.get(ext, 0) for ext in extensions)
            if count > 0:
                language_counts[lang] = count
                
        if language_counts:
            return max(language_counts, key=language_counts.get)
        return None
    
    async def _detect_architecture(self, repo_path: str, structure: Dict) -> str:
        """Detect architecture type."""
        # Check for microservices
        if await self._check_indicator(repo_path, "docker-compose"):
            services_count = await self.core_tools.count_patterns(
                repo_path, 
                {"services": r"^\s*services:\s*$"}
            )
            if services_count.get("services", 0) > 2:
                return "microservices"
                
        # Check for serverless
        if await self._check_indicator(repo_path, "serverless.yml"):
            return "serverless"
            
        # Check for monorepo
        if Path(repo_path, "lerna.json").exists() or Path(repo_path, "nx.json").exists():
            return "monorepo"
            
        # Check for standard patterns
        if Path(repo_path, "src").exists():
            if Path(repo_path, "src/main").exists():
                return "java-standard"
            return "standard"
            
        return "monolithic"
    
    async def _detect_frameworks(self, repo_path: str) -> List[str]:
        """Detect frameworks being used."""
        frameworks = []
        
        # Check package.json for JS frameworks
        package_json_path = Path(repo_path) / "package.json"
        if package_json_path.exists():
            try:
                with open(package_json_path) as f:
                    package = json.load(f)
                    deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
                    
                    framework_map = {
                        "express": "Express",
                        "react": "React",
                        "vue": "Vue",
                        "angular": "Angular",
                        "next": "Next.js",
                        "gatsby": "Gatsby",
                        "nestjs": "NestJS",
                    }
                    
                    for key, name in framework_map.items():
                        if key in deps:
                            frameworks.append(name)
            except:
                pass
                
        # Check for Python frameworks
        if await self._check_indicator(repo_path, "django"):
            frameworks.append("Django")
        if await self._check_indicator(repo_path, "flask"):
            frameworks.append("Flask")
        if await self._check_indicator(repo_path, "fastapi"):
            frameworks.append("FastAPI")
            
        return frameworks
    
    async def _has_tests(self, repo_path: str) -> bool:
        """Check if repository has tests."""
        test_indicators = ["test", "spec", "__tests__", "tests/"]
        for indicator in test_indicators:
            if await self._check_indicator(repo_path, indicator):
                return True
        return False
    
    async def _has_ci_cd(self, repo_path: str) -> bool:
        """Check for CI/CD configuration."""
        ci_indicators = [
            ".github/workflows",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            ".circleci",
            "azure-pipelines.yml"
        ]
        for indicator in ci_indicators:
            if await self._check_indicator(repo_path, indicator):
                return True
        return False
    
    async def _find_entry_points(self, repo_path: str, primary_language: str) -> List[str]:
        """Find application entry points."""
        entry_points = []
        
        common_entry_points = {
            "python": ["main.py", "app.py", "run.py", "manage.py", "__main__.py"],
            "javascript": ["index.js", "app.js", "server.js", "main.js"],
            "java": ["Main.java", "Application.java"],
            "go": ["main.go"],
            "csharp": ["Program.cs"],
        }
        
        if primary_language in common_entry_points:
            for entry in common_entry_points[primary_language]:
                files = await self.core_tools.find_files_by_extension(
                    repo_path, 
                    [entry], 
                    max_files=10
                )
                entry_points.extend(files)
                
        return entry_points
    
    def _categorize_size(self, total_files: int) -> str:
        """Categorize repository size."""
        if total_files < 50:
            return "small"
        elif total_files < 500:
            return "medium"
        elif total_files < 5000:
            return "large"
        else:
            return "very_large"