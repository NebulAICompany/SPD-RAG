import os
from langchain_core.tools import tool
from backend.shared.logger import get_logger

logger = get_logger("SKILLS_TOOL")

@tool(parse_docstring=True)
def load_skill(skill_name: str) -> str:
    """Load a specialized skill instruction from the skills library.

    Use this tool to load specific domain knowledge or instructions when needed.
    
    Args:
        skill_name: The name of the skill to load (e.g., "Churn"). Do not include the .md extension.
    """
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        skills_path = os.path.join(base_path, "skills")
        
        file_path = os.path.join(skills_path, f"{skill_name}.md")
        
        logger.info(f"Loading skill: {skill_name} from {file_path}")
        
        if not os.path.exists(file_path):
            return f"Skill '{skill_name}' not found. Available skills are in the skills directory."
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return content
        
    except Exception as e:
        logger.error(f"Error loading skill {skill_name}: {e}")
        return f"Error loading skill: {str(e)}"
