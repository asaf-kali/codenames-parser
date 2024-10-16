#/bin/bash

mkdir -p exports

CASE=$1
SOURCE_DIR="debug/${CASE}"
EXPORT_DIR="exports"
EXPORT_FILE="${EXPORT_DIR}/${CASE}.gif"
DUPLICATE_SUFFIX="_gif_dup"

# Duplicate last frame, only if it's not already duplicated
LAST_FILE=$(ls -r ${SOURCE_DIR} | head -n 1)
echo "Last file: '${LAST_FILE}'"
if [[ ! "${LAST_FILE}" == *"${DUPLICATE_SUFFIX}"* ]]; then
  echo "Duplicating last file"
  cp "${SOURCE_DIR}/${LAST_FILE}" "${SOURCE_DIR}/${LAST_FILE}${DUPLICATE_SUFFIX}1.jpg"
  cp "${SOURCE_DIR}/${LAST_FILE}" "${SOURCE_DIR}/${LAST_FILE}${DUPLICATE_SUFFIX}2.jpg"
else
  echo "Last file already duplicated"
fi

# Create gif
ffmpeg \
  -framerate 1.5 \
  -pattern_type glob \
  -i "${SOURCE_DIR}/*.jpg" \
  -r 1.5 \
  -vf "scale=512:-1, pad=512:512:(ow-iw)/2:(oh-ih)/2" \
  -pix_fmt yuv420p \
  -y ${EXPORT_FILE}
