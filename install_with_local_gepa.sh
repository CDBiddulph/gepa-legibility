#!/bin/bash
# Script to install the project with local gepa, handling dspy conflicts

set -e  # Exit on error

echo "🔧 Installing project with local gepa..."

# 1. First, uninstall any existing gepa to avoid conflicts
echo "📦 Removing any existing gepa installation..."
pip uninstall gepa -y 2>/dev/null || true

# 2. Install the main project dependencies (without gepa)
echo "📦 Installing project dependencies..."
pip install -e .

# 3. Patch dspy's metadata to accept any gepa version
echo "🔧 Patching dspy metadata to accept gepa>=0.0.7..."
python -c "
import os
import site
import re

# Find site-packages directory
for site_pkg in site.getsitepackages():
    if os.path.exists(site_pkg):
        # Look for dspy dist-info directories
        for dirname in os.listdir(site_pkg):
            if dirname.startswith('dspy-') and dirname.endswith('.dist-info'):
                metadata_path = os.path.join(site_pkg, dirname, 'METADATA')

                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        content = f.read()

                    # Replace the pinned gepa version with a flexible one
                    updated_content = re.sub(
                        r'Requires-Dist: gepa\[dspy\]==[\d.]+',
                        'Requires-Dist: gepa[dspy]>=0.0.7',
                        content
                    )

                    if content != updated_content:
                        with open(metadata_path, 'w') as f:
                            f.write(updated_content)
                        print(f'  ✓ Patched {metadata_path}')
" || echo "  ⚠️ Could not patch dspy metadata (may not be installed)"

# 4. Install local gepa in editable mode
echo "📦 Installing local gepa from ./gepa..."
pip install -e gepa/ --no-deps  # Use --no-deps to avoid dependency conflicts

# 5. Verify the installation
echo ""
echo "✅ Verifying installation..."
python -c "
import gepa
import os
gepa_location = gepa.__file__
if 'gepa/gepa/src' in gepa_location or 'gepa/gepa' in os.path.dirname(gepa_location):
    print(f'  ✓ Using local gepa from: {gepa_location}')
else:
    print(f'  ⚠️ Warning: gepa loaded from unexpected location: {gepa_location}')

# Check version
try:
    import importlib.metadata
    version = importlib.metadata.version('gepa')
    print(f'  ✓ gepa version: {version}')
except:
    print('  ℹ️ Could not determine gepa version')
"

echo ""
echo "🎉 Installation complete! Your project is now using the local gepa."
echo ""
echo "To update in the future, just run: ./install_with_local_gepa.sh"