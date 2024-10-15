#!/bin/bash

# Set the folder containing the files
input_folder="debug/cut_rotated"

# Output folder
output_folder="exports/frames"
rm -rf "$output_folder"
mkdir -p "$output_folder"

count=0
# Loop through each file in the folder, sorted alphabetically
find "$input_folder" -type f | sort | while read -r input_file; do
  count=$((count+1))
  file_format="${input_file##*.}"
  count_padded=$(printf "%03d" $count)
  output_file="$output_folder/$count_padded.$file_format"
  cp "$input_file" "$output_file"
done
ls -l "$output_folder"

# Create the video
ffmpeg -framerate 2 \
  -pattern_type glob -i "$output_folder/*.jpg" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2, pad=max(iw\,ih):(ow/2)*2:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 \
  -profile:v high \
  -crf 20 \
  -pix_fmt yuv420p \
  "exports/video.mp4"
