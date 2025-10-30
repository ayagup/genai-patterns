"""
Extract pattern implementations from implementations.py into separate files
"""
import re
import os

def extract_patterns():
    # Read the implementations file
    with open('implementations.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_pattern = None
    pattern_content = []
    patterns_extracted = []
    in_code_block = False
    
    for i, line in enumerate(lines):
        # Check for pattern start markers
        if '```python patterns/' in line or '```python' in line and '_pattern' in line.lower():
            # Extract filename
            match = re.search(r'(\d+_[\w_]+)\.py', line)
            if match:
                # Save previous pattern if exists
                if current_pattern and pattern_content:
                    save_pattern(current_pattern, pattern_content)
                    patterns_extracted.append(current_pattern)
                
                current_pattern = match.group(1) + '.py'
                pattern_content = []
                in_code_block = True
                continue
        
        # Check for code block that's not a pattern
        elif line.strip().startswith('```') and current_pattern:
            # Check if this is closing the pattern
            if in_code_block and pattern_content:
                # Save the pattern
                save_pattern(current_pattern, pattern_content)
                patterns_extracted.append(current_pattern)
                current_pattern = None
                pattern_content = []
                in_code_block = False
                continue
        
        # Collect content if we're in a pattern
        if current_pattern and in_code_block:
            # Skip the opening ```python line
            if not line.strip().startswith('```'):
                pattern_content.append(line)
    
    # Save last pattern if exists
    if current_pattern and pattern_content:
        save_pattern(current_pattern, pattern_content)
        patterns_extracted.append(current_pattern)
    
    return patterns_extracted

def save_pattern(filename, content):
    """Save pattern to a file"""
    # Clean up the content
    cleaned_content = []
    for line in content:
        # Remove markdown artifacts
        if line.strip() and not line.strip().startswith('```'):
            cleaned_content.append(line)
    
    # Write to file
    filepath = filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_content)
    
    print(f"✓ Extracted: {filename} ({len(cleaned_content)} lines)")

if __name__ == "__main__":
    print("=" * 60)
    print("Extracting Pattern Implementations")
    print("=" * 60)
    print()
    
    patterns = extract_patterns()
    
    print()
    print("=" * 60)
    print(f"Successfully extracted {len(patterns)} patterns")
    print("=" * 60)
    
    for i, pattern in enumerate(patterns, 1):
        print(f"{i:2d}. {pattern}")
