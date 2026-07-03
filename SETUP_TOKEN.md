# Setting Up HuggingFace Token for Faster Downloads

## Why Set Up a Token?

- **Faster downloads**: 10-100x faster than unauthenticated downloads
- **Higher rate limits**: No throttling
- **Access to gated models**: Some models require authentication
- **Better reliability**: Fewer download failures

## Quick Setup (3 Methods)

### Method 1: Environment Variable (Recommended)

**1. Get your token:**
- Go to https://huggingface.co/settings/tokens
- Click "New token"
- Select "Read" access
- Copy the token

**2. Set it temporarily (current session only):**
```powershell
# Windows PowerShell
$env:HF_TOKEN="your_token_here"

# Windows CMD
set HF_TOKEN=your_token_here

# Linux/Mac
export HF_TOKEN="your_token_here"
```

**3. Set it permanently:**
```powershell
# Windows PowerShell (adds to user environment variables)
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'your_token_here', 'User')

# Then restart your terminal/IDE
```

### Method 2: .env File (Easiest)

**1. Copy the example file:**
```bash
cp .env.example .env
```

**2. Edit `.env` file:**
```env
HF_TOKEN=your_token_here
```

**3. The script will automatically load it**

### Method 3: Config File

**1. Edit `configs/base_config.yaml`:**
```yaml
huggingface:
  token: "your_token_here"
```

## Verify It's Working

Run training and you should see:
```
Using HuggingFace token for authenticated downloads
```

Instead of:
```
Warning: You are sending unauthenticated requests to the HF Hub
```

## Security Notes

- **Never commit `.env` file to git** (it's in .gitignore)
- **Keep your token secret** - it gives access to your account
- **Use "Read" only tokens** for training (no need for write access)
- **Rotate tokens periodically** for security

## Troubleshooting

**"Invalid API key" error:**
- Make sure you copied the full token
- Check for extra spaces or quotes
- Ensure token has "Read" access

**Token not being used:**
- Check if `.env` file is in project root
- Verify environment variable is set: `echo $HF_TOKEN`
- Check config file has correct format

**Still slow downloads:**
- Token might not be the issue (could be network speed)
- Try using a different network
- Model might be large (15GB for Qwen 2.5 7B)

## Alternative: Use Smaller Model

If download speed is an issue, consider using Phi-3-Mini instead:
```bash
python scripts/train_rl.py --config configs/phi3_config.yaml
```

Phi-3-Mini is only ~7GB (half the size of Qwen 2.5 7B).
