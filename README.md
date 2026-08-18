# Juniper DG

A Streamlit app for running the jupyter dg code online.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## Deployment

The app's entry point is `gen_app.py`.

- **Streamlit Community Cloud**: set "Main file path" to `gen_app.py` when
  creating the app at share.streamlit.io — this is a deployment setting,
  not a repo file, so it must be set there directly.
- **Other hosts (Render, Railway, Heroku, custom containers)**: the
  included `Procfile` already points to `gen_app.py`
  (`streamlit run gen_app.py --server.port=$PORT --server.headless=true`).

## Citation

TBA

## Problems or Questions?

Contact: L.I.Vazquez-Salazar (mailto:l.i.vazquez-salazar@thphys.uni-heidelberg.de)
