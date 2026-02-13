# Privacy Guide: Removing Wandb Credentials

Before sharing or uploading this project publicly, follow these steps to remove your Weights & Biases credentials and personal information.

## Quick Clean (Recommended)

Run this script to remove all wandb-related privacy information:

```bash
# Remove wandb credentials
rm -rf ~/.netrc
rm -rf ~/.config/wandb/

# Remove wandb cache from project
rm -rf wandb/
rm -rf .wandb/

# Remove wandb logs
rm -rf logs/*wandb*
```

## Detailed Steps

### 1. Remove Global Wandb Credentials

Wandb stores API keys in your home directory:

```bash
# Remove netrc file (contains API key)
rm -rf ~/.netrc

# Remove wandb config directory
rm -rf ~/.config/wandb/

# Remove wandb cache
rm -rf ~/.cache/wandb/
```

### 2. Remove Project-Specific Wandb Files

Clean wandb directories from your project:

```bash
cd /path/to/cdg_finance

# Remove wandb run directories
rm -rf wandb/
rm -rf .wandb/

# Remove any wandb logs
find . -type d -name "wandb" -exec rm -rf {} +
```

### 3. Update Configuration Files

Edit `config/config.py` and remove your personal information:

```python
@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "cdg-finance"
    entity: str = None  # ← Change this to None (remove your username/team)
    run_name: str = None
    tags: List[str] = None
    notes: str = None
```

### 4. Check for Sensitive Data in Logs

Review log files for any API keys or personal information:

```bash
# Search for potential API keys
grep -r "api_key\|token\|secret" logs/

# If found, remove those logs
rm logs/suspicious_file.log
```

### 5. Verify Clean State

```bash
# Check for remaining wandb files
find . -name "*wandb*" -type f
find . -name "*wandb*" -type d

# Check config file
grep -i "entity" config/config.py
```

## What Gets Removed

| Item | Location | Contains |
|------|----------|----------|
| API Key | `~/.netrc` | Wandb authentication token |
| Config | `~/.config/wandb/` | User settings, entity info |
| Cache | `~/.cache/wandb/` | Cached data |
| Run logs | `wandb/` | Local run data, may contain entity |
| Settings | `.wandb/` | Local settings |

## Safe to Share After Cleanup

After following these steps, the following files are safe to share:

✅ All source code (`.py` files)
✅ Configuration with `entity=None`
✅ README and documentation
✅ Shell scripts
✅ Requirements file
✅ Model checkpoints (if you want to share them)

## Still Want to Use Wandb Locally?

If you want to continue using wandb locally while keeping your project clean:

```bash
# Before uploading/sharing
./cleanup_wandb.sh  # Create this script with the commands above

# After downloading on another machine
wandb login  # Re-authenticate with your API key
# Edit config/config.py and add your entity back
```

## Create a Cleanup Script

Create `cleanup_wandb.sh` for easy cleanup:

```bash
#!/bin/bash
echo "Cleaning wandb credentials and cache..."

# Remove global credentials
rm -rf ~/.netrc
rm -rf ~/.config/wandb/
rm -rf ~/.cache/wandb/

# Remove project wandb files
rm -rf wandb/
rm -rf .wandb/

# Update config file
sed -i.bak 's/entity: str = ".*"/entity: str = None/' config/config.py

echo "✓ Cleanup complete!"
echo "Remember to check config/config.py manually"
```

Make it executable:
```bash
chmod +x cleanup_wandb.sh
```

## Additional Security

For extra security before uploading to GitHub/GitLab:

1. **Add to .gitignore**:
```bash
echo "wandb/" >> .gitignore
echo ".wandb/" >> .gitignore
echo "*.log" >> .gitignore
echo ".netrc" >> .gitignore
```

2. **Check git history** (if already committed):
```bash
# See if API keys were ever committed
git log -p | grep -i "api_key\|token"

# If found, consider rewriting history (DANGEROUS)
# Consult Git documentation first
```

3. **Use environment variables** (for future):
```python
# Instead of hardcoding in config
import os
entity = os.getenv("WANDB_ENTITY", None)
```

## Need Help?

If you're unsure whether your wandb data is fully cleaned:

1. Search for your wandb username in the project: `grep -r "your_username" .`
2. Check wandb website for any public runs you don't want shared
3. Contact wandb support to delete specific runs

## Wandb Privacy Best Practices

Going forward:

- ✅ Use `--no-wandb` flag for sensitive experiments
- ✅ Set `entity=None` in config before sharing
- ✅ Add `wandb/` to `.gitignore`
- ✅ Use environment variables for credentials
- ❌ Never commit API keys to version control
- ❌ Don't share screenshots with run URLs visible
