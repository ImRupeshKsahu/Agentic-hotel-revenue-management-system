# Hotel RMS PoC

Proof-of-concept hotel revenue management system with demand forecasting, PMS snapshots, pricing rules, and an optional LLM strategy layer.

## Setup

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Create your local environment file.

   ```powershell
   Copy-Item .env.example .env
   ```

4. Edit `.env` and add your real API key.

## Run

```powershell
streamlit run src/app.py
```

## Test

```powershell
python -m unittest discover -s tests
```

## Notes

- Historical occupancy data covers July 1, 2015 through September 6, 2017.
- `.env` is ignored by git and should not be committed.
- Generated caches, model binaries, plots, and backtest outputs are ignored so the GitHub repo stays source-focused.
