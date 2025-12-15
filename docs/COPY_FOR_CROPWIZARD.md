# Files to Copy for Standalone Cropwizard Website

## Required Files Structure

```
cropwizard-website/
├── index.html                    (rename from index_cropwizard.html)
├── README.md                     (optional, update for Cropwizard)
├── leaderboards/
│   ├── MMMT.json
│   └── MMST_Standard.json
└── static/
    ├── css/
    │   ├── bulma.min.css
    │   ├── bulma-carousel.min.css
    │   ├── bulma-slider.min.css
    │   ├── fontawesome.all.min.css
    │   ├── index.css
    │   └── cropwizard.css
    ├── js/
    │   ├── bulma-carousel.min.js
    │   ├── bulma-slider.min.js
    │   ├── fontawesome.all.min.js
    │   └── index.js
    ├── figures/
    │   ├── Category_Examples/     (all PNG files)
    │   ├── LLM_Judge_Examples/    (all PNG files)
    │   ├── logo/                  (all logo files - you'll want to replace MIRAGE logos)
    │   ├── MMMT/                  (all files)
    │   └── MMST/                  (all files)
    ├── images/
    │   └── main.png               (main overview image)
    └── audio/                     (optional - only if keeping audio feature)
        └── MIRAGE_ EN2.wav        (rename/replace with Cropwizard audio if needed)
```

## Detailed Copy List

### 1. Main HTML File
- ✅ `index_cropwizard.html` → rename to `index.html` in new project

### 2. Leaderboard Data (REQUIRED)
- ✅ `leaderboards/MMMT.json`
- ✅ `leaderboards/MMST_Standard.json`

### 3. CSS Files (REQUIRED)
- ✅ `static/css/bulma.min.css`
- ✅ `static/css/bulma-carousel.min.css`
- ✅ `static/css/bulma-slider.min.css`
- ✅ `static/css/fontawesome.all.min.css`
- ✅ `static/css/index.css`
- ✅ `static/css/cropwizard.css`

### 4. JavaScript Files (REQUIRED)
- ✅ `static/js/bulma-carousel.min.js`
- ✅ `static/js/bulma-slider.min.js`
- ✅ `static/js/fontawesome.all.min.js`
- ✅ `static/js/index.js`

### 5. Images & Figures (REQUIRED)
- ✅ `static/images/main.png`
- ✅ `static/figures/Category_Examples/` (all PNG files)
- ✅ `static/figures/LLM_Judge_Examples/` (all PNG files)
- ✅ `static/figures/MMMT/` (all files)
- ✅ `static/figures/MMST/` (all files)
- ⚠️ `static/figures/logo/` (all files - but replace MIRAGE logos with Cropwizard logos)

### 6. Audio Files (OPTIONAL)
- ⚠️ `static/audio/MIRAGE_ EN2.wav` (only if keeping audio feature - rename/replace)

## Files to SKIP (Not Needed)

- ❌ `index.html` (original MIRAGE file)
- ❌ `index_old.html`
- ❌ `leaderboard_data.json` (not used in current setup)
- ❌ `task_pipeline_player.js` (not used)
- ❌ `static/css/mirage.css` (replaced by cropwizard.css)
- ❌ `static/figures/Error_Analysis/` (entire folder - removed from website)
- ❌ `static/images/farmer-robot.png`, `farmers.png`, `opena.webp` (not referenced)

## Quick Copy Command

If you want to copy everything needed, you can use:

```bash
# Create new directory structure
mkdir -p cropwizard-website/{leaderboards,static/{css,js,figures/{Category_Examples,LLM_Judge_Examples,logo,MMMT,MMST},images,audio}}

# Copy main HTML (rename it)
cp index_cropwizard.html cropwizard-website/index.html

# Copy leaderboards
cp leaderboards/*.json cropwizard-website/leaderboards/

# Copy CSS (excluding mirage.css)
cp static/css/{bulma*.css,fontawesome*.css,index.css,cropwizard.css} cropwizard-website/static/css/

# Copy JS
cp static/js/*.js cropwizard-website/static/js/

# Copy figures (excluding Error_Analysis)
cp -r static/figures/{Category_Examples,LLM_Judge_Examples,logo,MMMT,MMST} cropwizard-website/static/figures/

# Copy main image
cp static/images/main.png cropwizard-website/static/images/

# Copy audio (optional)
cp static/audio/*.wav cropwizard-website/static/audio/ 2>/dev/null || true
```

## After Copying - Things to Update

1. **Logo Files**: Replace MIRAGE logos in `static/figures/logo/` with Cropwizard logos
   - Update references in `index.html` if logo filenames change
   
2. **Audio File**: If keeping audio, rename/replace `MIRAGE_ EN2.wav` and update reference in HTML

3. **README.md**: Create/update README for Cropwizard project

4. **Links**: Update any GitHub, HuggingFace, or paper links in `index.html` to point to Cropwizard resources

5. **Meta Tags**: Already updated in `index_cropwizard.html`, but double-check

6. **Favicon**: Update favicon links in HTML head section if you have Cropwizard-specific icons

## Minimal Required Files (Smallest Working Set)

If you want the absolute minimum:

```
index.html
leaderboards/MMMT.json
leaderboards/MMST_Standard.json
static/css/{bulma.min.css,index.css,cropwizard.css}
static/js/{bulma-carousel.min.js,index.js}
static/figures/{Category_Examples,LLM_Judge_Examples,MMMT,MMST,logo}
static/images/main.png
```

Note: External CDN dependencies (Swiper, KaTeX, jQuery, DataTables) are loaded from CDN, so no need to copy those.

