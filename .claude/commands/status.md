# Project Status Check

Check what has been built and what remains. Run this at the start of every
session to orient before writing any code.

## Steps

1. Read docs/RESEARCH.md and report which phases are marked complete
2. Run `pytest tests/ -v --tb=no -q` and report pass/fail counts
3. Check if data/indices/ exists and report collection sizes if it does:
   ```python
   import chromadb
   client = chromadb.PersistentClient(path="./data/indices")
   for col in client.list_collections():
       print(col.name, col.count())
   ```
4. List any TODO comments in src/ files: `grep -r "TODO" src/`
5. Report the next unchecked phase from RESEARCH.md build phases

## Output Format

Report as:
- Completed phases: [list]
- Tests: X passed, Y failed
- KB status: text_collection=N docs, image_collection=N docs (or "not built")
- Open TODOs: [list]
- Next task: [specific next step]
