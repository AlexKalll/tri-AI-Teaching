# TRI AI Saturdays Cohort 10 — Setup Guide

## Overview

This guide covers two ways to set up your environment for the TRI AI sessions:

1. **Google Colab** (recommended for beginners) — zero setup, runs in your browser
2. **Local Python** (recommended if you want to run things offline)

Choose the option that best fits your experience level and hardware.

---

## Option 1: Google Colab (Recommended for Beginners)

Colab runs Jupyter notebooks in your browser on Google's servers. You don't need to install anything.

### Steps

1. Go to https://colab.research.google.com/
2. Sign in with a Google account (create one if you don't have it)
3. Click **File → Upload Notebook** and select the `.ipynb` file from the session
4. Add this as the first cell and run it:

```python
!pip install tokenizers numpy matplotlib pandas
```

5. You're ready to go.

### Pros
- No installation needed
- Free GPU access (Runtime → Change runtime type → T4 GPU)
- Works on any computer including Chromebooks and tablets
- Auto-saves to Google Drive

### Cons
- Requires internet connection
- Limited to 12+ hours of continuous runtime (free tier)
- Cannot save large files permanently

---

## Option 2: Local Python Setup

### Prerequisites

- Python 3.9, 3.10, or 3.11 (3.12 works but may have some library issues)
- pip (Python package manager, comes with Python)
- Basic familiarity with the terminal / command prompt

### Step 1: Check if Python is installed

Open a terminal and run:

```bash
python --version
```

If you see `Python 3.9.x` or higher, you're good. If not, download from:

https://www.python.org/downloads/

**Important:** During installation on Windows, check "Add Python to PATH".

### Step 2: Create a virtual environment

Navigate to where you want to store your project files, then:

**Windows:**
```bash
python -m venv tri_env
tri_env\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv tri_env
source tri_env/bin/activate
```

Your terminal prompt should now show `(tri_env)` at the beginning, meaning the environment is active.

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install tokenizers numpy matplotlib pandas jupyter
```

### Step 4: Start Jupyter

```bash
jupyter notebook
```

This opens a browser tab showing your file directory. Navigate to where you saved the session notebook and open it.

### Step 5: Shutting down

When done, press `Ctrl+C` in the terminal to stop Jupyter. Close the browser tab. Deactivate the environment:

```bash
deactivate
```

Next time, just run steps 2 (activate) and 4 again.

---

## Option 3: Using VS Code with Jupyter Extension

If you already use VS Code, this is a middle ground between Colab and full local setup.

### Steps

1. Install Python 3.9+ (see Step 1 above)
2. Install VS Code from https://code.visualstudio.com/
3. Install the **Python** and **Jupyter** extensions from the VS Code marketplace
4. Create your virtual environment (see Step 2 above)
5. Install dependencies (see Step 3 above)
6. Open the `.ipynb` file in VS Code — it will render as an interactive notebook
7. Select your virtual environment as the kernel (click "Select Kernel" in the top right)

---

## Downloading the Notebooks

Session notebooks will be shared via:
- GitHub repository (link provided in session)
- Direct download link
- Google Drive folder

**To download:**
1. Click the file link
2. Click **Download** or **Save As**
3. Save as `.ipynb` extension
4. If using Colab: **File → Upload Notebook**
5. If using local Jupyter: move the file to your project folder and open via Jupyter

---

## Testing Your Setup

Run this short script to verify everything is installed correctly:

```python
# test_setup.py
import sys
import tokenizers
import numpy as np
import matplotlib
import pandas as pd

print(f"Python version: {sys.version}")
print(f"tokenizers version: {tokenizers.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Pandas version: {pd.__version__}")

# Quick BPE test
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
trainer = trainers.BpeTrainer(vocab_size=100, min_frequency=1)

# Train on a tiny corpus
corpus = ["cassava", "maize", "yam", "plantain"]
tokenizer.train_from_iterator(corpus, trainer)

# Test tokenization
output = tokenizer.encode("cassava")
print(f"\nBPE test passed!")
print(f"Input: 'cassava'")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")
print("\n=== Setup verified successfully! ===")
```

Save this as `test_setup.py` and run:

```bash
python test_setup.py
```

If you see "Setup verified successfully!" at the end, everything is working.

---

## Troubleshooting

### `python` command not found
**Solution:** On Windows, reinstall Python and check "Add Python to PATH". On macOS, use `python3` instead of `python`. On Linux, install via your package manager: `sudo apt install python3 python3-pip`.

### pip install fails with "permission denied"
**Solution:** You are not in a virtual environment, or you are trying to install system-wide. Create and activate a virtual environment (see Step 2 above). If that doesn't help, try `pip install --user <package>`.

### `jupyter` command not found
**Solution:** After installing via pip, make sure your virtual environment is active. On Windows, `Scripts\activate`. On macOS/Linux, `source bin/activate`.

### "No module named tokenizers" in Jupyter
**Solution:** You installed tokenizers in one Python environment but Jupyter is using a different one. In the notebook, run:
```python
import sys
print(sys.executable)
```
This shows which Python Jupyter is using. Make sure it matches the one where you installed packages. Alternatively, install the `ipykernel` package matching your virtual environment.

### Colab: "Cannot find notebook file"
**Solution:** Make sure you uploaded the `.ipynb` file (not the `.md` or `.py` version). The file should have a notebook icon in Colab's file browser.

### Colab: Runtime disconnects
**Solution:** Colab's free tier disconnects after ~90 minutes of inactivity. Run a cell every 30 minutes to keep it alive. For longer sessions, use Runtime → "Run all" and let it complete in one go. You can also purchase Colab Pro for longer runtimes.

### Windows: "Python is not recognized as an internal or external command"
**Solution:** Python was not added to PATH. Reinstall Python and check the box "Add Python to environment variables". Or manually add Python to your PATH system variable.

### Mac: "zsh: command not found: python"
**Solution:** macOS doesn't ship with Python 3. Install via `brew install python` or download from python.org. Use `python3` instead of `python`.

### "pip is not recognized" (Windows)
**Solution:** Same as above — Python not on PATH. After installing Python, use `py -m pip install <package>` as a workaround.

---

## Quick Reference: Terminal Commands

| Action | Windows | macOS / Linux |
|--------|---------|---------------|
| Create venv | `python -m venv tri_env` | `python3 -m venv tri_env` |
| Activate venv | `tri_env\Scripts\activate` | `source tri_env/bin/activate` |
| Deactivate venv | `deactivate` | `deactivate` |
| Install packages | `pip install <package>` | `pip install <package>` |
| Start Jupyter | `jupyter notebook` | `jupyter notebook` |
| Run Python script | `python script.py` | `python3 script.py` |
| Check Python version | `python --version` | `python3 --version` |

---

## Need Help?

- Ask in the cohort's communication channel (Slack / WhatsApp / Discord)
- Post on the GitHub repository Issues page
- Reach out to your session instructor or TA
- Check the TRI AI documentation at [opencode.ai](https://opencode.ai)
