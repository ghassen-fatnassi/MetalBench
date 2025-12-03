#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

def analyze_json(data, prefix='', result=None):
    """Recursively analyze JSON structure and collect unique values."""
    if result is None:
        result = defaultdict(set)
    
    if isinstance(data, dict):
        for key, value in data.items():
            path = f'{prefix}.{key}' if prefix else key
            
            if isinstance(value, (dict, list)):
                analyze_json(value, path, result)
            elif isinstance(value, bool):
                result[path].add(f'bool: {value}')
            elif isinstance(value, (int, float)):
                # Skip integers/floats but note the type
                if path not in result:
                    result[path] = set(['<numeric>'])
            elif value is None:
                result[path].add('null')
            else:
                result[path].add(str(value))
    
    elif isinstance(data, list):
        if not data:
            result[prefix if prefix else '<root>'].add('<empty list>')
        else:
            for item in data:
                analyze_json(item, prefix, result)
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_analyzer.py <json_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    result = analyze_json(data)
    
    print(f"\n{'='*60}")
    print(f"JSON Structure Analysis: {filename}")
    print(f"{'='*60}\n")
    
    for path in sorted(result.keys()):
        values = result[path]
        
        if '<numeric>' in values:
            print(f"📊 {path}: numeric type (integers/floats)")
        else:
            print(f"📌 {path}: {len(values)} unique value(s)")
            
            # Show values if reasonable number
            if len(values) <= 20 and '<numeric>' not in values:
                for value in sorted(values):
                    print(f"   • {value}")
            elif len(values) > 20:
                sample = sorted(values)[:10]
                for value in sample:
                    print(f"   • {value}")
                print(f"   ... and {len(values) - 10} more")
        print()

if __name__ == '__main__':
    main()