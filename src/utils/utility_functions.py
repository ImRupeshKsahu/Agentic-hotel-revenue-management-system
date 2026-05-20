import unicodedata
import re

def normalize_reasoning(text):
    if not text:
        return ""
    
    # 1. Normalize Unicode characters (converts fancy numbers to standard ones)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Fix the "missing gaps" by ensuring spaces after punctuation and numbers
    text = re.sub(r'(%|[\d\)])([A-Za-z])', r'\1 \2', text)
    
    # 3. Replace all types of whitespace (tabs, newlines, non-breaking spaces) with a single standard space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def escape_streamlit_markdown(text):
    if not text:
        return ""
    # Streamlit renders Markdown inside alert boxes. Escape currency markers so
    # amounts like $538 do not get interpreted as inline math.
    return str(text).replace("$", r"\$")

def load_prompt(file_path, **kwargs):
    with open(file_path, 'r') as f:
        template = f.read()
    return template.format(**kwargs)
