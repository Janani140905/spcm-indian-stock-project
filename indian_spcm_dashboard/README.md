# Indian SPCM Dashboard (Demo UI)

This is a standalone Streamlit dashboard scaffold intended for a quick guide presentation.
It deliberately labels all sample prices, charts, and sentiment items as illustrative demo data.
It does not claim to provide live prices or actual model predictions.

## Run on Windows PowerShell

```powershell
cd path\to\indian_spcm_dashboard
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL Streamlit prints (normally http://localhost:8501).

## Tomorrow's integration
- Replace demo `stocks` data with the existing `IndianMarketDataFetcher` output.
- Connect the existing price feature inference module.
- Connect the Phase 3 news/sentiment adapter.
- Connect predictions only when compatible frozen model and preprocessing artifacts exist.
- Preserve data timestamps and provider/feed labels.
