#!/bin/bash

# Script to copy files needed for standalone Cropwizard website
# Usage: ./copy_for_cropwizard.sh /path/to/destination

if [ -z "$1" ]; then
    echo "Usage: $0 <destination_directory>"
    echo "Example: $0 ../cropwizard-website"
    exit 1
fi

DEST="$1"

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$DEST"/{leaderboards,static/{css,js,figures/{Category_Examples,LLM_Judge_Examples,logo,MMMT,MMST},images,audio}}

# Copy main HTML (rename it)
echo "Copying main HTML file..."
cp index_cropwizard.html "$DEST/index.html"

# Copy leaderboards
echo "Copying leaderboard data..."
cp leaderboards/*.json "$DEST/leaderboards/"

# Copy CSS (excluding mirage.css)
echo "Copying CSS files..."
cp static/css/bulma.min.css "$DEST/static/css/"
cp static/css/bulma-carousel.min.css "$DEST/static/css/"
cp static/css/bulma-slider.min.css "$DEST/static/css/"
cp static/css/fontawesome.all.min.css "$DEST/static/css/"
cp static/css/index.css "$DEST/static/css/"
cp static/css/cropwizard.css "$DEST/static/css/"

# Copy JS
echo "Copying JavaScript files..."
cp static/js/*.js "$DEST/static/js/"

# Copy figures (excluding Error_Analysis)
echo "Copying figure files..."
cp -r static/figures/Category_Examples "$DEST/static/figures/"
cp -r static/figures/LLM_Judge_Examples "$DEST/static/figures/"
cp -r static/figures/logo "$DEST/static/figures/"
cp -r static/figures/MMMT "$DEST/static/figures/"
cp -r static/figures/MMST "$DEST/static/figures/"

# Copy main image
echo "Copying images..."
cp static/images/main.png "$DEST/static/images/"

# Copy audio (optional - will skip if doesn't exist)
echo "Copying audio files (if available)..."
cp static/audio/*.wav "$DEST/static/audio/" 2>/dev/null || echo "  (No audio files found, skipping)"

# Copy README if it exists
if [ -f "README.md" ]; then
    echo "Copying README..."
    cp README.md "$DEST/"
fi

echo ""
echo "✅ Copy complete!"
echo ""
echo "Next steps:"
echo "1. Replace MIRAGE logos in $DEST/static/figures/logo/ with Cropwizard logos"
echo "2. Update logo references in $DEST/index.html if filenames change"
echo "3. Update links (GitHub, HuggingFace, paper) in $DEST/index.html"
echo "4. Update README.md for Cropwizard project"
echo ""
echo "Files copied to: $DEST"

