# TFG

Project for misinformation analysis in text.

## Structure

- `Claims/`: claim-related logic
- `Patterns/`: pattern detectors (bias, emotion, fallacies, propaganda, violence)
- `Tone/`: tone analysis
- `misinfo_value.py`: misinformation scoring entry point
- `translate.py`: translation utilities

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies (if applicable).
3. Run modules, for example:

```bash
python .\Patterns\propaganda.py
```

## GitHub upload

The repository is configured with a `.gitignore` to exclude local environment and cache files.
