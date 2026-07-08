import json
import sys
import os
import argparse

def convert_json_to_md(input_filepath, output_filepath):
    try:
        # Read the JSON file
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the text content
        # Assuming the structure is:
        # data["content"][0]["text"]
        text_content = data.get("content", [])[0].get("text", "")
        
        if not text_content:
            print("Error: Could not find text content in the JSON structure.", file=sys.stderr)
            sys.exit(1)

        # Write to the markdown file
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)
            
        print(f"Successfully converted '{input_filepath}' to '{output_filepath}'")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_filepath}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{input_filepath}' is not a valid JSON file", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract markdown text from a JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("-o", "--output", help="Path to the output Markdown file (optional)", default=None)
    
    args = parser.parse_args()
    
    # If no output file is provided, change the extension of the input file to .md
    output_file = args.output
    if not output_file:
        base_name, _ = os.path.splitext(args.input_file)
        output_file = f"{base_name}.md"
        
    convert_json_to_md(args.input_file, output_file)
