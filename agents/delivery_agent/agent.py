"""
Delivery Agent - Responsible for creating proper folder structure and file formatting
Takes refined code and delivers it as a well-organized project
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os
import re
import json
import shutil
from datetime import datetime
import ast
import black
import isort
import autopep8
import logging

import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OPENTELEMETRY_SUPPRESS_INSTRUMENTATION"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DEFAULT_OUTPUT_DIR = Path("delivered_projects")
PYTHON_FILE_EXTENSIONS = {'.py', '.pyw'}
CONFIG_FILE_PATTERNS = {
    'requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml',
    'Dockerfile', '.dockerignore', '.gitignore', 'README.md',
    'Makefile', 'pytest.ini', 'tox.ini', '.env.example'
}

# --- CODE ANALYSIS TOOLS ---

def analyze_code_structure(code: str) -> Dict[str, Any]:
    """Analyze code to determine project structure needs"""
    try:
        structure = {
            "type": "unknown",
            "has_main": False,
            "has_classes": False,
            "has_functions": False,
            "imports": [],
            "suggested_files": [],
            "dependencies": set(),
            "is_single_file": True,
            "components": {}
        }
        
        # Check for obvious patterns
        if "from flask import" in code or "from fastapi import" in code:
            structure["type"] = "web_api"
        elif "import django" in code or "from django" in code:
            structure["type"] = "django"
        elif "if __name__ == '__main__':" in code:
            structure["type"] = "script"
            structure["has_main"] = True
        
        # Parse imports
        import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(.+)$'
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            module = match.group(1) or match.group(2).split(',')[0].strip()
            structure["imports"].append(module)
            
            # Extract dependencies (third-party)
            base_module = module.split('.')[0]
            if base_module not in ['os', 'sys', 'json', 'datetime', 'pathlib', 're', 'typing']:
                structure["dependencies"].add(base_module)
        
        # Check for classes and functions
        structure["has_classes"] = "class " in code
        structure["has_functions"] = "def " in code
        
        # Determine if this should be split into multiple files
        lines = code.split('\n')
        if len(lines) > 200 or structure["has_classes"]:
            structure["is_single_file"] = False
            
            # Try to identify components
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        structure["components"][node.name] = "class"
                    elif isinstance(node, ast.FunctionDef):
                        if not any(node.name.startswith(prefix) for prefix in ['_', 'test_']):
                            structure["components"][node.name] = "function"
            except:
                pass
        
        # Suggest file structure based on type
        if structure["type"] == "web_api":
            structure["suggested_files"] = [
                "main.py", "app.py", "models.py", "routes.py", 
                "utils.py", "config.py", "requirements.txt"
            ]
        elif structure["type"] == "django":
            structure["suggested_files"] = [
                "manage.py", "settings.py", "urls.py", "views.py",
                "models.py", "requirements.txt"
            ]
        else:
            if structure["is_single_file"]:
                structure["suggested_files"] = ["main.py", "requirements.txt"]
            else:
                structure["suggested_files"] = [
                    "main.py", "utils.py", "models.py", "config.py", "requirements.txt"
                ]
        
        return structure
        
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        return {
            "type": "unknown",
            "suggested_files": ["main.py", "requirements.txt"],
            "dependencies": set(),
            "error": str(e)
        }

def split_code_into_components(code: str, structure: Dict[str, Any]) -> Dict[str, str]:
    """Split code into logical components for multiple files"""
    try:
        components = {
            "main": [],
            "models": [],
            "utils": [],
            "config": [],
            "routes": [],
            "imports": set()
        }
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except:
            # If parsing fails, return as single file
            return {"main.py": code}
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    components["imports"].add(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                names = ', '.join([alias.name for alias in node.names])
                components["imports"].add(f"from {node.module} import {names}")
        
        # Categorize code blocks
        lines = code.split('\n')
        current_component = "main"
        current_block = []
        
        for line in lines:
            # Skip import lines as they're already extracted
            if line.strip().startswith(('import ', 'from ')) and ' import ' in line:
                continue
                
            # Detect component switches
            if 'class' in line and ('Model' in line or 'Schema' in line):
                if current_block:
                    components[current_component].extend(current_block)
                current_component = "models"
                current_block = [line]
            elif 'def ' in line and any(util in line for util in ['parse', 'validate', 'format', 'convert', 'calculate']):
                if current_block:
                    components[current_component].extend(current_block)
                current_component = "utils"
                current_block = [line]
            elif '@app.route' in line or '@router' in line:
                if current_block:
                    components[current_component].extend(current_block)
                current_component = "routes"
                current_block = [line]
            else:
                current_block.append(line)
        
        # Add remaining block
        if current_block:
            components[current_component].extend(current_block)
        
        # Build file contents
        files = {}
        
        # Create files only if they have content
        for component, content_lines in components.items():
            if component == "imports":
                continue
            if content_lines:
                # Add appropriate imports to each file
                file_content = []
                
                # Add common imports
                file_content.extend(sorted(components["imports"]))
                file_content.append("")
                file_content.append("")
                
                # Add content
                file_content.extend(content_lines)
                
                filename = f"{component}.py"
                files[filename] = '\n'.join(file_content)
        
        # If no splitting occurred, return original
        if not files or len(files) == 1:
            return {"main.py": code}
            
        return files
        
    except Exception as e:
        logger.error(f"Code splitting error: {e}")
        return {"main.py": code}

def format_python_code(code: str) -> str:
    """Format Python code using black, isort, and autopep8"""
    try:
        # First, fix basic PEP8 issues with autopep8
        code = autopep8.fix_code(code, options={'aggressive': 1})
        
        # Sort imports with isort
        code = isort.code(code, profile="black", line_length=88)
        
        # Format with black
        try:
            code = black.format_str(code, mode=black.Mode(line_length=88))
        except:
            # If black fails, continue with isort result
            pass
            
        return code
        
    except Exception as e:
        logger.warning(f"Code formatting error: {e}, returning original")
        return code

def generate_requirements_txt(dependencies: set) -> str:
    """Generate requirements.txt content"""
    # Map common imports to package names
    package_mapping = {
        'flask': 'flask>=2.0.0',
        'fastapi': 'fastapi>=0.68.0\nuvicorn>=0.15.0',
        'django': 'django>=4.0.0',
        'requests': 'requests>=2.26.0',
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.3.0',
        'pytest': 'pytest>=6.2.0',
        'black': 'black>=22.0.0',
        'isort': 'isort>=5.10.0',
        'sqlalchemy': 'sqlalchemy>=1.4.0',
        'pydantic': 'pydantic>=1.8.0',
        'aiohttp': 'aiohttp>=3.8.0',
        'beautifulsoup4': 'beautifulsoup4>=4.10.0',
        'scipy': 'scipy>=1.7.0',
        'matplotlib': 'matplotlib>=3.4.0',
        'seaborn': 'seaborn>=0.11.0',
        'sklearn': 'scikit-learn>=1.0.0',
        'torch': 'torch>=1.10.0',
        'tensorflow': 'tensorflow>=2.7.0'
    }
    
    requirements = []
    for dep in sorted(dependencies):
        if dep in package_mapping:
            requirements.append(package_mapping[dep])
        else:
            # For unknown packages, add without version
            requirements.append(dep)
    
    return '\n'.join(requirements)

def generate_readme(project_name: str, structure: Dict[str, Any]) -> str:
    """Generate a README.md file"""
    readme = f"""# {project_name}

## Description
This project was automatically generated and organized by the Delivery Agent.

## Project Type
{structure.get('type', 'Python Application').replace('_', ' ').title()}

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure
"""
    
    if structure.get("suggested_files"):
        readme += "\n```\n"
        for file in structure["suggested_files"]:
            if file.endswith('.py'):
                readme += f"├── {file}\n"
        readme += "├── requirements.txt\n"
        readme += "└── README.md\n```\n"
    
    readme += "\n## Dependencies\n"
    if structure.get("dependencies"):
        for dep in sorted(structure["dependencies"]):
            readme += f"- {dep}\n"
    
    readme += f"\n## Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return readme

def generate_gitignore() -> str:
    """Generate a .gitignore file"""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Environment
.env
.env.local

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Project specific
*.sqlite3
*.db
"""

# --- MAIN DELIVERY FUNCTIONS ---

def create_project_structure(
    project_name: str,
    code: str,
    output_dir: str = None,
    create_venv: bool = False
) -> Dict[str, Any]:
    """Create complete project structure from code"""
    try:
        # Setup paths
        base_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        project_dir = base_dir / project_name.replace(' ', '_').lower()
        
        # Create directory
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze code structure
        structure = analyze_code_structure(code)
        
        # Split code if needed
        if structure["is_single_file"]:
            files = {"main.py": code}
        else:
            files = split_code_into_components(code, structure)
        
        # Format and save Python files
        created_files = []
        for filename, content in files.items():
            if filename.endswith('.py'):
                # Format the code
                formatted_content = format_python_code(content)
                
                # Save file
                filepath = project_dir / filename
                filepath.write_text(formatted_content, encoding='utf-8')
                created_files.append(str(filepath))
                logger.info(f"Created: {filepath}")
        
        # Generate and save requirements.txt
        requirements_content = generate_requirements_txt(structure["dependencies"])
        requirements_path = project_dir / "requirements.txt"
        requirements_path.write_text(requirements_content, encoding='utf-8')
        created_files.append(str(requirements_path))
        
        # Generate and save README.md
        readme_content = generate_readme(project_name, structure)
        readme_path = project_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        created_files.append(str(readme_path))
        
        # Generate and save .gitignore
        gitignore_path = project_dir / ".gitignore"
        gitignore_path.write_text(generate_gitignore(), encoding='utf-8')
        created_files.append(str(gitignore_path))
        
        # Create virtual environment if requested
        if create_venv:
            venv_path = project_dir / "venv"
            if not venv_path.exists():
                import subprocess
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
                logger.info(f"Created virtual environment: {venv_path}")
        
        return {
            "status": "success",
            "project_dir": str(project_dir),
            "created_files": created_files,
            "structure": structure,
            "file_count": len(created_files),
            "has_venv": create_venv,
            "message": f"Successfully created project '{project_name}' with {len(created_files)} files"
        }
        
    except Exception as e:
        logger.error(f"Project creation error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to create project: {e}"
        }

def validate_project_structure(project_dir: str) -> Dict[str, Any]:
    """Validate that all files exist and are properly formatted"""
    try:
        project_path = Path(project_dir)
        
        if not project_path.exists():
            return {
                "is_valid": False,
                "message": "Project directory does not exist",
                "issues": ["Directory not found"]
            }
        
        validation_results = {
            "is_valid": True,
            "files_checked": [],
            "issues": [],
            "warnings": []
        }
        
        # Check for essential files
        essential_files = ["main.py", "requirements.txt", "README.md"]
        for file in essential_files:
            filepath = project_path / file
            if filepath.exists():
                validation_results["files_checked"].append(file)
                
                # Validate Python files
                if file.endswith('.py'):
                    content = filepath.read_text(encoding='utf-8')
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        validation_results["issues"].append(f"Syntax error in {file}: {e}")
                        validation_results["is_valid"] = False
            else:
                if file != "main.py":  # main.py might be named differently
                    validation_results["warnings"].append(f"Missing {file}")
        
        # Check if any Python files exist
        py_files = list(project_path.glob("*.py"))
        if not py_files:
            validation_results["issues"].append("No Python files found")
            validation_results["is_valid"] = False
        
        # Check file sizes
        for filepath in project_path.iterdir():
            if filepath.is_file():
                size = filepath.stat().st_size
                if size == 0:
                    validation_results["warnings"].append(f"Empty file: {filepath.name}")
                validation_results["files_checked"].append(filepath.name)
        
        validation_results["message"] = "Valid" if validation_results["is_valid"] else "Invalid"
        validation_results["total_files"] = len(validation_results["files_checked"])
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "is_valid": False,
            "error": str(e),
            "message": f"Validation failed: {e}"
        }

def package_project(
    project_dir: str,
    output_format: str = "zip",
    include_venv: bool = False
) -> Dict[str, Any]:
    """Package the project into a distributable format"""
    try:
        project_path = Path(project_dir)
        
        if not project_path.exists():
            return {
                "status": "error",
                "message": "Project directory not found"
            }
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{project_path.name}_{timestamp}"
        
        if output_format == "zip":
            import zipfile
            output_file = project_path.parent / f"{output_name}.zip"
            
            with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in project_path.rglob('*'):
                    if file_path.is_file():
                        # Skip venv if not included
                        if not include_venv and 'venv' in file_path.parts:
                            continue
                        # Skip cache files
                        if '__pycache__' in file_path.parts:
                            continue
                        
                        arcname = file_path.relative_to(project_path.parent)
                        zipf.write(file_path, arcname)
            
            return {
                "status": "success",
                "package_path": str(output_file),
                "format": output_format,
                "size": output_file.stat().st_size,
                "message": f"Project packaged as {output_file.name}"
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unsupported format: {output_format}"
            }
            
    except Exception as e:
        logger.error(f"Packaging error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to package: {e}"
        }

def generate_deployment_files(
    project_dir: str,
    deployment_type: str = "docker"
) -> Dict[str, Any]:
    """Generate deployment configuration files"""
    try:
        project_path = Path(project_dir)
        created_files = []
        
        if deployment_type == "docker":
            # Generate Dockerfile
            dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
"""
            dockerfile_path = project_path / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            created_files.append("Dockerfile")
            
            # Generate docker-compose.yml
            compose_content = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - .:/app
"""
            compose_path = project_path / "docker-compose.yml"
            compose_path.write_text(compose_content)
            created_files.append("docker-compose.yml")
            
            # Generate .dockerignore
            dockerignore_content = """__pycache__
*.pyc
venv/
.env
.git/
.gitignore
*.log
.pytest_cache/
.coverage
"""
            dockerignore_path = project_path / ".dockerignore"
            dockerignore_path.write_text(dockerignore_content)
            created_files.append(".dockerignore")
        
        return {
            "status": "success",
            "deployment_type": deployment_type,
            "created_files": created_files,
            "message": f"Created {len(created_files)} deployment files"
        }
        
    except Exception as e:
        logger.error(f"Deployment file generation error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# --- DELIVERY AGENT DEFINITION ---

delivery_agent = LlmAgent(
    name="DeliveryAgent",
    model="gemini-2.0-flash",
    description="Delivers refined code as a properly structured project with formatted files",
    instruction="""You are the Delivery Agent responsible for creating well-organized projects from refined code.

Your workflow:
1. When you receive refined code, first use create_project_structure to:
   - Create a project with an appropriate name based on the code functionality
   - The function will analyze the code and create appropriate files
   - It will also format the code and generate supporting files

2. After creating the project, use validate_project_structure to:
   - Ensure all files were created properly
   - Check for any syntax errors or issues
   - Verify the project is ready for use

3. Optionally, you can:
   - Use package_project to create a zip file for easy distribution
   - Use generate_deployment_files to add Docker support

4. Provide a comprehensive summary including:
   - Project location
   - Files created
   - Any warnings or issues
   - Next steps for the user

Always ensure the delivered project is complete, well-organized, and ready to use.

Remember:
- Choose meaningful project names based on the code's purpose
- The system automatically handles code formatting and file splitting
- Always validate after creation
- Provide clear instructions for using the delivered project""",
    tools=[
        FunctionTool(create_project_structure),
        FunctionTool(validate_project_structure),
        FunctionTool(package_project),
        FunctionTool(generate_deployment_files),
        FunctionTool(analyze_code_structure)
    ],
    output_key="delivery_result"
)

# For standalone testing or importing
root_agent = delivery_agent

if __name__ == "__main__":
    print("\n📦 DELIVERY AGENT READY")
    print("=" * 60)
    
    print("\n🎯 CAPABILITIES:")
    print("- Analyzes code to determine project structure")
    print("- Splits large code into logical components")
    print("- Formats code with black, isort, and autopep8")
    print("- Creates proper folder structure")
    print("- Generates requirements.txt automatically")
    print("- Creates README.md with project info")
    print("- Adds .gitignore for Python projects")
    print("- Validates project structure")
    print("- Packages projects as zip files")
    print("- Generates Docker deployment files")
    
    print("\n📁 DEFAULT PROJECT STRUCTURE:")
    print("project_name/")
    print("├── main.py          # Main application file")
    print("├── utils.py         # Utility functions (if needed)")
    print("├── models.py        # Data models (if needed)")
    print("├── config.py        # Configuration (if needed)")
    print("├── requirements.txt # Dependencies")
    print("├── README.md        # Project documentation")
    print("├── .gitignore       # Git ignore rules")
    print("└── venv/            # Virtual environment (optional)")
    
    print("\n🚀 USAGE:")
    print("The agent receives refined code and automatically:")
    print("1. Analyzes the code structure")
    print("2. Creates appropriate project layout")
    print("3. Formats all Python files")
    print("4. Generates supporting files")
    print("5. Validates everything works")
    
    print("\n✨ FEATURES:")
    print("- Automatic code splitting for large files")
    print("- Smart dependency detection")
    print("- Professional code formatting")
    print("- Complete project documentation")
    print("- Ready-to-deploy structure")
    
    print("\n📝 Example input:")
    print('{')
    print('    "refined_code": "# Your refined code here...",')
    print('    "project_name": "my_awesome_api"  # Optional')
    print('}')
    
    print("\n✅ Delivery Agent initialized successfully!")
    print("=" * 60)