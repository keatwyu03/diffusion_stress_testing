#!/bin/bash

# Cleanup script to remove wandb credentials and personal information
# Run this before sharing or uploading the project publicly

echo "=========================================="
echo "Wandb Privacy Cleanup"
echo "=========================================="

echo ""
echo "[1/5] Removing global wandb credentials..."
rm -rf ~/.netrc
rm -rf ~/.config/wandb/
rm -rf ~/.cache/wandb/
echo "✓ Global credentials removed"

echo ""
echo "[2/5] Removing project wandb files..."
rm -rf wandb/
rm -rf .wandb/
echo "✓ Project wandb files removed"

echo ""
echo "[3/5] Checking for wandb in logs..."
if ls logs/*wandb* 2>/dev/null; then
    rm -f logs/*wandb*
    echo "✓ Wandb logs removed"
else
    echo "✓ No wandb logs found"
fi

echo ""
echo "[4/5] Checking config file..."
if grep -q 'entity: str = "' config/config.py 2>/dev/null; then
    echo "⚠ WARNING: Found entity value in config/config.py"
    echo "Please manually edit config/config.py and set:"
    echo "  entity: str = None"
else
    echo "✓ Config file looks clean"
fi

echo ""
echo "[5/5] Verifying cleanup..."
REMAINING=$(find . -name "*wandb*" -type d 2>/dev/null | wc -l | tr -d ' ')
if [ "$REMAINING" -eq 0 ]; then
    echo "✓ No wandb directories remaining"
else
    echo "⚠ Found $REMAINING wandb directories:"
    find . -name "*wandb*" -type d
fi

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Manually check config/config.py for entity value"
echo "2. Review the checklist in PRIVACY.md"
echo "3. Verify with: grep -r 'entity\|wandb' config/"
echo ""
echo "=========================================="
