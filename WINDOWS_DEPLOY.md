# Windows deployment

Use these files from the project root:

- `run_server.bat`
- `run_server.vbs`

## On the target machine

1. Copy the whole project folder.
2. Install Python 3.12 x64.
3. Create and populate the virtual environment:
   ```bat
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
4. Run `run_server.bat` once so the VietOCR model is downloaded into `app\models\` if it is not already there.
5. Create a Task Scheduler task:
   - Trigger: `At startup`
   - Action: `wscript.exe`
   - Arguments: `"C:\path\to\project\run_server.vbs"`
   - Enable `Run whether user is logged on or not`
   - Enable `Run with highest privileges`

## Notes

- The script auto-detects CUDA and uses `cuda:0` when available; otherwise it falls back to `cpu`.
- Port defaults to `8000`.
- If you already have `.venv\Scripts\python.exe`, the batch file will use it automatically.
