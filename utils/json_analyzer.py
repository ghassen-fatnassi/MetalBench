#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

def infer_type(value):
    """Infer the type of a value."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    return "unknown"

def analyze_structure(data, collected=None):
    """Analyze JSON structure and collect type information and unique values."""
    if collected is None:
        collected = {}
    
    if isinstance(data, dict):
        result = {"type": "object", "properties": {}}
        for key, value in data.items():
            result["properties"][key] = analyze_structure(value, collected)
        return result
    
    elif isinstance(data, list):
        if not data:
            return {"type": "array", "items": "unknown"}
        
        # Analyze all items to find types
        item_types = defaultdict(int)
        examples = []
        
        for item in data:
            item_structure = analyze_structure(item, collected)
            item_type = json.dumps(item_structure, sort_keys=True)
            item_types[item_type] += 1
            if len(examples) < 3:
                examples.append(item)
        
        # If all items have same structure, use that
        if len(item_types) == 1:
            return {"type": "array", "items": json.loads(list(item_types.keys())[0])}
        else:
            # Mixed types
            return {"type": "array", "items": "mixed", "count": len(data)}
    
    else:
        # Scalar value
        return {"type": infer_type(data)}

def collect_unique_values(data, path="", collected=None):
    """Collect unique values for non-numeric fields."""
    if collected is None:
        collected = defaultdict(set)
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            collect_unique_values(value, new_path, collected)
    
    elif isinstance(data, list):
        for item in data:
            collect_unique_values(item, path, collected)
    
    else:
        # Scalar value
        if not isinstance(data, (int, float)) and data is not None:
            collected[path].add(str(data))
    
    return collected

def format_structure(structure, unique_values, indent=0):
    """Format the structure in a readable way for LLMs."""
    lines = []
    prefix = "  " * indent
    
    if structure["type"] == "object":
        lines.append(f"{prefix}{{")
        for key, value in structure["properties"].items():
            path = key
            lines.append(f"{prefix}  \"{key}\": ", )
            
            if value["type"] in ["object", "array"]:
                sub_lines = format_structure(value, unique_values, indent + 1)
                # Remove the prefix from first line since we already added the key
                first_line = sub_lines[0].lstrip()
                lines[-1] += first_line
                lines.extend(sub_lines[1:])
            else:
                type_info = f"<{value['type']}>"
                
                # Add unique values if available and reasonable count
                if path in unique_values and len(unique_values[path]) <= 15:
                    values = sorted(unique_values[path])[:10]
                    values_str = ", ".join(f'"{v}"' for v in values)
                    if len(unique_values[path]) > 10:
                        values_str += f", ... ({len(unique_values[path])} unique values)"
                    type_info += f" // Possible values: {values_str}"
                elif path in unique_values:
                    type_info += f" // {len(unique_values[path])} unique values"
                
                lines[-1] += type_info
        
        lines.append(f"{prefix}}}")
    
    elif structure["type"] == "array":
        if isinstance(structure.get("items"), dict):
            lines.append(f"{prefix}[")
            sub_lines = format_structure(structure["items"], unique_values, indent + 1)
            lines.extend(sub_lines)
            lines.append(f"{prefix}]")
        else:
            item_type = structure.get("items", "unknown")
            count_info = f" // {structure['count']} items" if "count" in structure else ""
            lines.append(f"{prefix}[<{item_type}>]{count_info}")
    
    return lines

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_schema_generator.py <json_file>")
        print("\nGenerates a structure description suitable for passing to LLMs")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Analyze structure
    structure = analyze_structure(data)
    
    # Collect unique values
    unique_values = collect_unique_values(data)
    
    # Format output
    print("\n" + "="*70)
    print(f"JSON STRUCTURE: {filename}")
    print("="*70 + "\n")
    
    lines = format_structure(structure, unique_values)
    for line in lines:
        print(line)
    
    print("\n" + "="*70)
    print("FIELD DESCRIPTIONS:")
    print("="*70 + "\n")
    
    # Print detailed info about fields with many unique values
    for path, values in sorted(unique_values.items()):
        if len(values) > 15:
            sample = sorted(values)[:5]
            print(f"{path}: <string>")
            print(f"  • {len(values)} unique values")
            print(f"  • Sample: {', '.join(repr(v) for v in sample)}")
            print()

if __name__ == '__main__':
    main()