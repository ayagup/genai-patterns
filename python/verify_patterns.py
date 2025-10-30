#!/usr/bin/env python3
"""
Pattern File Verifier and Index Generator
Verifies all extracted pattern files and generates an index
"""
import os
import glob
import py_compile
from pathlib import Path

def verify_patterns():
    """Verify all pattern Python files"""
    pattern_files = sorted(glob.glob('[0-9][0-9]_*.py'))
    
    print("=" * 70)
    print("Pattern File Verification")
    print("=" * 70)
    print()
    
    valid_files = []
    invalid_files = []
    
    for filepath in pattern_files:
        try:
            py_compile.compile(filepath, doraise=True)
            lines = len(open(filepath, encoding='utf-8').readlines())
            valid_files.append((filepath, lines))
            print(f"✓ {filepath:45s} ({lines:4d} lines)")
        except Exception as e:
            invalid_files.append((filepath, str(e)))
            print(f"✗ {filepath:45s} ERROR: {e}")
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Valid files:   {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")
    print(f"Total lines:   {sum(lines for _, lines in valid_files):,}")
    
    return valid_files, invalid_files

def generate_index(valid_files):
    """Generate an index of all patterns"""
    
    index_content = """# Pattern Implementation Index

## All Extracted Patterns

This index lists all extracted pattern implementations with line counts.

### Pattern Files

"""
    
    # Group patterns by category (first digit)
    categories = {
        '0': 'Core Architectural Patterns (01-09)',
        '1': 'Intermediate Patterns (10-19)',
        '2': 'Advanced Patterns (20-29)',
        '3': 'Specialized Patterns (30-39)',
        '4': 'Complex Patterns (40-49)',
        '5': 'Expert Patterns (50-59)',
        '7': 'Resource Management (70-79)',
        '9': 'Infrastructure Patterns (90-99)',
    }
    
    current_category = None
    
    for filepath, lines in valid_files:
        # Determine category
        first_digit = filepath[0]
        category = categories.get(first_digit, 'Other Patterns')
        
        if category != current_category:
            index_content += f"\n### {category}\n\n"
            current_category = category
        
        # Extract pattern name from filename
        pattern_name = filepath.replace('_', ' ').replace('.py', '').title()
        index_content += f"- **{filepath}** - {lines:,} lines - {pattern_name}\n"
    
    index_content += f"""

## Statistics

- **Total Patterns**: {len(valid_files)}
- **Total Lines of Code**: {sum(lines for _, lines in valid_files):,}
- **Average Lines per Pattern**: {sum(lines for _, lines in valid_files) // len(valid_files):,}
- **Largest Pattern**: {max(valid_files, key=lambda x: x[1])[0]} ({max(valid_files, key=lambda x: x[1])[1]:,} lines)
- **Smallest Pattern**: {min(valid_files, key=lambda x: x[1])[0]} ({min(valid_files, key=lambda x: x[1])[1]:,} lines)

## Usage

To run any pattern:

```bash
python <pattern_file.py>
```

Example:
```bash
python 01_react_pattern.py
python 53_mixture_of_agents.py
```

## Pattern Categories

### By Functionality:
- **Reasoning**: 01, 02, 03, 10, 23
- **Planning**: 04, 11, 24, 25
- **Multi-Agent**: 07, 15, 39, 40, 52, 53, 54
- **Memory**: 09, 18, 29, 30, 31
- **Tools**: 11, 12, 13, 19, 27
- **Safety**: 14, 15, 36
- **Optimization**: 20, 21, 38, 43, 79, 90
- **RAG**: 06, 17, 37
- **Streaming**: 22, 28

### By Difficulty:
- **Beginner**: 01, 02, 06, 08, 11
- **Intermediate**: 03, 04, 07, 09, 13, 20, 21, 26, 27
- **Advanced**: 05, 10, 15, 16, 17, 18, 23, 24, 25, 28, 29, 30, 31, 35, 36, 37, 38
- **Expert**: 39, 40, 43, 52, 53, 54, 79, 90

---

Generated: {Path(__file__).name}
"""
    
    with open('PATTERN_INDEX.md', 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print("\n✓ Generated PATTERN_INDEX.md")

if __name__ == "__main__":
    valid, invalid = verify_patterns()
    
    if valid:
        generate_index(valid)
    
    print("\n✓ Verification complete!")
