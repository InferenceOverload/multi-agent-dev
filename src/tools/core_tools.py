"""Core universal tools that work with any codebase."""

import os
import re
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import mimetypes

class CoreTools:
    """Universal tools that work on any codebase."""
    
    @staticmethod
    async def read_file(file_path: str, max_lines: int = 1000) -> Dict[str, Any]:
        """
        Read a file's content.
        
        Args:
            file_path: Path to the file
            max_lines: Maximum lines to read (for large files)
            
        Returns:
            Dict with file content and metadata
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
                
            # Detect file type
            mime_type, _ = mimetypes.guess_type(file_path)
            
            # Read content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
                    
            return {
                "path": file_path,
                "content": ''.join(lines),
                "lines_read": len(lines),
                "mime_type": mime_type,
                "size": path.stat().st_size,
                "extension": path.suffix
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    async def search_pattern(
        directory: str, 
        pattern: str, 
        file_extensions: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for a regex pattern in files.
        
        Args:
            directory: Directory to search in
            pattern: Regex pattern to search for
            file_extensions: List of extensions to search (e.g., ['.py', '.js'])
            max_results: Maximum number of results
            
        Returns:
            List of matches with file path and line information
        """
        results = []
        regex = re.compile(pattern, re.IGNORECASE)
        count = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file_extensions:
                    if not any(file.endswith(ext) for ext in file_extensions):
                        continue
                        
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append({
                                    "file": file_path,
                                    "line_number": line_num,
                                    "line": line.strip(),
                                    "match": regex.search(line).group()
                                })
                                count += 1
                                if count >= max_results:
                                    return results
                except:
                    continue
                    
        return results
    
    @staticmethod
    async def get_directory_structure(
        directory: str, 
        max_depth: int = 3,
        ignore_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get directory structure as a tree.
        
        Args:
            directory: Root directory
            max_depth: Maximum depth to traverse
            ignore_patterns: Patterns to ignore (e.g., ['node_modules', '.git'])
            
        Returns:
            Directory structure as nested dict
        """
        if ignore_patterns is None:
            ignore_patterns = ['node_modules', '.git', '__pycache__', '.venv', 'venv']
            
        def should_ignore(name):
            return any(pattern in name for pattern in ignore_patterns)
            
        def build_tree(path, current_depth=0):
            if current_depth >= max_depth:
                return None
                
            tree = {
                "name": os.path.basename(path) or path,
                "type": "directory" if os.path.isdir(path) else "file",
                "path": path
            }
            
            if os.path.isdir(path) and not should_ignore(path):
                tree["children"] = []
                try:
                    for item in sorted(os.listdir(path)):
                        if not should_ignore(item):
                            item_path = os.path.join(path, item)
                            child = build_tree(item_path, current_depth + 1)
                            if child:
                                tree["children"].append(child)
                except PermissionError:
                    pass
                    
            return tree
            
        return build_tree(directory)
    
    @staticmethod
    async def find_files_by_extension(
        directory: str, 
        extensions: List[str],
        max_files: int = 500
    ) -> List[str]:
        """
        Find all files with given extensions.
        
        Args:
            directory: Directory to search
            extensions: List of extensions (e.g., ['.py', '.js'])
            max_files: Maximum files to return
            
        Returns:
            List of file paths
        """
        files = []
        count = 0
        
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
                    count += 1
                    if count >= max_files:
                        return files
                        
        return files
    
    @staticmethod
    async def analyze_dependencies(file_path: str) -> Dict[str, List[str]]:
        """
        Analyze imports/dependencies in a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict of dependencies by type
        """
        dependencies = {
            "imports": [],
            "requires": [],
            "includes": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Python imports
            python_imports = re.findall(r'^(?:from|import)\s+([^\s]+)', content, re.MULTILINE)
            dependencies["imports"].extend(python_imports)
            
            # JavaScript/TypeScript imports and requires
            js_imports = re.findall(r"(?:import|from)\s+['\"]([^'\"]+)['\"]", content)
            js_requires = re.findall(r"require\(['\"]([^'\"]+)['\"]\)", content)
            dependencies["imports"].extend(js_imports)
            dependencies["requires"].extend(js_requires)
            
            # C/C++ includes
            includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
            dependencies["includes"].extend(includes)
            
        except Exception as e:
            dependencies["error"] = str(e)
            
        return dependencies
    
    @staticmethod
    async def count_patterns(
        directory: str,
        patterns: Dict[str, str],
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Count occurrences of multiple patterns.
        
        Args:
            directory: Directory to search
            patterns: Dict of pattern_name -> regex_pattern
            file_extensions: Optional file extensions to filter
            
        Returns:
            Dict of pattern_name -> count
        """
        counts = {name: 0 for name in patterns}
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file_extensions:
                    if not any(file.endswith(ext) for ext in file_extensions):
                        continue
                        
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        for name, pattern in patterns.items():
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            counts[name] += len(matches)
                except:
                    continue
                    
        return counts