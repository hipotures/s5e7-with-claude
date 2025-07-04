#!/bin/bash
# Script to fix dataset paths in all Python files
# Changes paths like "../datasets/playground-series-s5e7/train.csv" to "../../train.csv"

echo "Fixing dataset paths in Python files..."
echo "======================================="

# Count files before changes
total_files=$(find . -name "*.py" | wc -l)
echo "Total Python files: $total_files"

# Find all Python files and fix the paths
files_changed=0

# Common path patterns to replace
# Pattern 1: ../datasets/playground-series-s5e7/
# Pattern 2: datasets/playground-series-s5e7/
# Pattern 3: Any variation with train.csv or test.csv

for file in *.py; do
    if [ -f "$file" ]; then
        # Check if file contains the old path
        if grep -q "datasets/playground-series-s5e7" "$file"; then
            echo "Fixing: $file"
            
            # Create backup
            cp "$file" "$file.bak"
            
            # Replace various path patterns
            sed -i 's|"\.\./datasets/playground-series-s5e7/train\.csv"|"../../train.csv"|g' "$file"
            sed -i 's|"\.\./datasets/playground-series-s5e7/test\.csv"|"../../test.csv"|g' "$file"
            sed -i 's|"\.\./datasets/playground-series-s5e7/sample_submission\.csv"|"../../sample_submission.csv"|g' "$file"
            
            # Handle single quotes too
            sed -i "s|'\.\.\/datasets\/playground-series-s5e7\/train\.csv'|'../../train.csv'|g" "$file"
            sed -i "s|'\.\.\/datasets\/playground-series-s5e7\/test\.csv'|'../../test.csv'|g" "$file"
            sed -i "s|'\.\.\/datasets\/playground-series-s5e7\/sample_submission\.csv'|'../../sample_submission.csv'|g" "$file"
            
            # Handle paths without ../
            sed -i 's|"datasets/playground-series-s5e7/train\.csv"|"../../train.csv"|g' "$file"
            sed -i 's|"datasets/playground-series-s5e7/test\.csv"|"../../test.csv"|g' "$file"
            sed -i 's|"datasets/playground-series-s5e7/sample_submission\.csv"|"../../sample_submission.csv"|g' "$file"
            
            # Check if changes were made
            if ! diff -q "$file" "$file.bak" > /dev/null; then
                ((files_changed++))
                # Remove backup if successful
                rm "$file.bak"
            else
                echo "  No changes needed in $file"
                rm "$file.bak"
            fi
        fi
    fi
done

echo ""
echo "Summary:"
echo "--------"
echo "Files checked: $total_files"
echo "Files modified: $files_changed"

# Show a few examples of the changes
echo ""
echo "Example of changes made:"
echo "------------------------"
grep -h "\.\.\/.*\.csv" *.py | grep -v "\.bak" | head -5 | while read -r line; do
    echo "  $line"
done

echo ""
echo "Dataset paths have been updated!"
echo "Old: ../datasets/playground-series-s5e7/train.csv"
echo "New: ../../train.csv"