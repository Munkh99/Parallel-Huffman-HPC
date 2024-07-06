#!/bin/bash

# Path to your existing 10MB file
SOURCE_FILE="file_10M.txt"

# Define the sizes for the output files in bytes
TARGET_SIZES=(128 256 512 1024 2048 4096)  # in MB

# Loop through each target size
for SIZE_MB in "${TARGET_SIZES[@]}"
do
    # Calculate the target size in bytes
    TARGET_SIZE=$((SIZE_MB * 1024 * 1024))

    # Calculate how many times to repeat the source file to reach the target size
    REPEAT=$(($TARGET_SIZE / $(stat -c %s "$SOURCE_FILE")))

    # Output file name
    OUTPUT_FILE="file_${SIZE_MB}M.txt"

    # Create the output file by repeating the source file
    echo "Generating ${SIZE_MB}MB file: $OUTPUT_FILE"
    for ((i=0; i<$REPEAT; i++))
    do
        cat "$SOURCE_FILE" >> "$OUTPUT_FILE"
    done

    echo "File generated: $OUTPUT_FILE"
done

echo "All files generated."
