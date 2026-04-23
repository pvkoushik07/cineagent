# CineAgent Setup Instructions

## Step 1 — Rename config files

These files need to be renamed because zip tools strip the dot prefix:

| Current name | Rename to | Where |
|---|---|---|
| `env_example.txt` | `.env` | root folder |
| `gitignore.txt` | `.gitignore` | root folder |
| `claude_config/` folder | `.claude` | root folder |

On Mac you can rename in terminal:
```bash
cd cineagent
mv env_example.txt .env
mv gitignore.txt .gitignore
mv claude_config .claude
```

On Windows (Command Prompt):
```
cd cineagent
ren env_example.txt .env
ren gitignore.txt .gitignore
ren claude_config .claude
```

## Step 2 — Fill in your API keys

Open `.env` in any text editor and replace the placeholder values:

```
TMDB_API_KEY=your_actual_key_here
GEMINI_API_KEY=your_actual_key_here
```

Get keys from:
- TMDB (free): https://developer.themoviedb.org -> Settings -> API
- Gemini (free): https://aistudio.google.com -> Get API Key

## Step 3 — Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Step 4 — Install Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

## Step 5 — Launch

```bash
claude
```

Then type `/status` to verify everything works.
