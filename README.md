🏃 Szacowanie czasu półmaratonu – aplikacja StreamlitAplikacja umożliwia oszacowanie przewidywanego czasu ukończenia półmaratonu (21.1 km) na podstawie:

-płci
-wieku
-tempa na 1 km
-opcjonalnie: czasu na 5 km

🚀 Funkcjonalności

✅ AI-powered input - LLM (OpenAI) wyciąga dane z tekstu
✅ Model ML z DigitalOcean Spaces
✅ Monitoring z Langfuse
✅ Analiza porównawcza z wykresami
✅ Czasy pośrednie na dystansie
✅ Responsywny interfejs Streamlit

📦 Struktura projektu

MODUŁ_9_ZAD_DOM/
├── app.py                          # Główna aplikacja Streamlit
├── requirements.txt                # Zależności
├── .env                           # Zmienne środowiskowe
├── .env.example                   # Przykład konfiguracji
├── data/                          # Dane treningowe
│   ├── halfmarathon_wroclaw_2023__final.csv
│   └── halfmarathon_wroclaw_2024__final.csv
├── models/                        # Modele ML
│   └── trained_model.pkl
├── notebooks/                     # Analizy i pipeline
│   ├── training_pipeline.ipynb
│   └── analiza.ipynb
└── README.md                      # Dokumentacja

🛠️ Wymagania

streamlit
pandas
numpy
plotly
joblib
boto3
python-dotenv
scikit-learn
openai>=1.50.0
langfuse

🔐 Konfiguracja (.env)

# DigitalOcean Spaces
DO_SPACES_KEY=your_key
DO_SPACES_SECRET=your_secret
DO_SPACES_REGION=fra1
DO_SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com
DO_SPACES_BUCKET=mf-hm-predictor
DO_SPACES_MODEL_KEY=trained_model.pkl

# OpenAI
OPENAI_API_KEY=your_openai_key

# Langfuse
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com

🚀 Uruchomienie lokalne

git clone https://github.com/PrzemekDS/halfmarathon-app.git
cd halfmarathon-app

pip install -r requirements.txt
streamlit run app.py

📊 Architektura
Frontend: Streamlit
ML Model: Scikit-learn + fallback heurystyka
LLM: OpenAI GPT-3.5-turbo
Monitoring: Langfuse
Storage: DigitalOcean Spaces
Visualization: Plotly

👨‍💻 Autor
Przemysław Patoleta / Przemek_DS
GitHub: https://github.com/PrzemekDS

Projekt edukacyjny w ramach kursu "Od Zera do AI"
