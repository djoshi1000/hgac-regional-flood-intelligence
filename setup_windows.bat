@echo off
py -3.11 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo Environment ready.
echo Run scripts 00_get_basin.py, 01_download_dem.py, 02_build_terrain.py
echo Then: streamlit run app.py
