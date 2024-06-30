#!/bin/bash

# Path to the dictionary file
dictionary="/usr/share/dict/words"

# Function to generate a text file of a given size with human-readable text
generate_text_file() {
    local size="$1"
    local filename="$2"

    # Generate random text by repeatedly reading from the dictionary file
    local current_size=0
    while [[ $current_size -lt $size ]]; do
        cat "$dictionary" >> "$filename"
        current_size=$(stat -c %s "$filename")
    done
    # Truncate the file to the desired size
    head -c "$size" "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"
}

# # Generate 1MB text file
# generate_text_file $((1024 * 1024)) "random_text_1mb.txt"
# echo "Random text file generated: random_text_1mb.txt"

# # Generate 10MB text file
# generate_text_file $((10 * 1024 * 1024)) "random_text_10mb.txt"
# echo "Random text file generated: random_text_10mb.txt"

# Generate 100MB text file
# generate_text_file $((100 * 1024 * 1024)) "random_text_100mb.txt"
# echo "Random text file generated: random_text_100mb.txt"

# Generate 100MB text file
# generate_text_file $((600 * 1024 * 1024)) "random_text_600mb.txt"
# echo "Random text file generated: random_text_600mb.txt"

# Generate 1GB text file
generate_text_file $((1024 * 1024 * 1024)) "random_text_1gb.txt"
echo "Random text file generated: random_text_1gb.txt"

# Generate 10GB text file (This may take a while)
# generate_text_file $((10 * 1024 * 1024 * 1024)) "random_text_10gb.txt"
# echo "Random text file generated: random_text_10gb.txt"