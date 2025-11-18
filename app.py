import os
import io
import joblib
import boto3
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta, datetime
from dotenv import load_dotenv
import openai
from langfuse import Langfuse
import json
import re

# Załaduj zmienne z .env
load_dotenv()

# Inicjalizacja Langfuse - TWOJA METODA
langfuse = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        st.sidebar.success("🔍 Langfuse: Aktywny")
    except Exception as e:
        st.sidebar.error(f"🔍 Langfuse: Błąd - {e}")

# ====================== USTAWIENIA STRONY ======================
st.set_page_config(
    page_title="Szacowanie czasu półmaratonu",
    page_icon="🏃",
    layout="centered",
)

# ====================== STYLE (BEZ GRADIENTU) ======================
CUSTOM_CSS = """
<style>
  .stApp {
    background: linear-gradient(135deg, #ced4da 0%, #adb5bd 45%, #ced4da 100%);
    color: #212529;
  }
  .main-title {
    text-align: center;
    padding-bottom: 0.5rem;
    color: #212529;
    font-weight: 700;
  }
  .stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #e8590c, #fd7e14);
    color: white;
    font-weight: 600;
    padding: 0.8rem 0;
    transition: all 0.2s ease-in-out;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 16px rgba(253, 126, 20, 0.35);
    background: linear-gradient(135deg, #d9480f, #e8590c);
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====================== DO SPACES – ŁADOWANIE MODELU ======================
@st.cache_resource(show_spinner=False)
def load_model_from_spaces():
    """Pobiera model z DigitalOcean Spaces"""
    try:
        key = os.getenv("DO_SPACES_KEY")
        secret = os.getenv("DO_SPACES_SECRET")
        region = os.getenv("DO_SPACES_REGION", "fra1")
        endpoint = os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com")
        bucket = os.getenv("DO_SPACES_BUCKET")
        model_key = os.getenv("DO_SPACES_MODEL_KEY", "model.pkl")

        if not all([key, secret, bucket]):
            st.info("ℹ️ Nie ustawiono zmiennych DO Spaces – użyję heurystyki.")
            return None

        s3 = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
        )

        obj = s3.get_object(Bucket=bucket, Key=model_key)
        byts = obj["Body"].read()
        model = joblib.load(io.BytesIO(byts))
        st.success("✅ Model ML załadowany z DigitalOcean Spaces")
        return model

    except Exception as e:
        st.warning(f"⚠️ Nie udało się pobrać modelu z DO Spaces (użyję heurystyki)")
        return None

MODEL = load_model_from_spaces()

# ====================== FUNKCJE POMOCNICZE ======================
def parse_time_to_seconds(text: str) -> int | None:
    """Konwertuje czas MM:SS na sekundy"""
    if not text or not text.strip():
        return None
    parts = text.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        minutes, seconds = int(parts[0]), int(parts[1])
        if seconds >= 60 or seconds < 0 or minutes < 0:
            return None
        return minutes * 60 + seconds
    except ValueError:
        return None

def format_seconds_to_hms(seconds: float) -> str:
    """Konwertuje sekundy na HH:MM:SS"""
    sec = int(round(seconds))
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    rest_seconds = sec % 60
    return f"{hours}:{minutes:02d}:{rest_seconds:02d}"

# ====================== LLM + LANGFUSE (TWOJA METODA + MOJA LOGIKA) ======================
def extract_running_data_with_llm(user_input):
    """Ekstrakcja danych biegowych przy użyciu LLM - POŁĄCZENIE METOD"""
    if not os.getenv("OPENAI_API_KEY"):
        return None, ["Brak klucza OpenAI API"]
    
    # TWOJA METODA: Stwórz trace
    trace = None
    if langfuse:
        try:
            trace = langfuse.trace(
                name="llm_half_marathon_data_extraction",
                input={"user_input": user_input}
            )
        except Exception as e:
            st.error(f"🔍 Langfuse: Błąd trace - {e}")
    
    try:
        # MOJA LOGIKA: Lepszy prompt (CZAS vs TEMPO)
        system_prompt = """
        Jesteś asystentem do ekstrakcji danych biegowych. Wyciągnij z tekstu użytkownika:
        - płeć (M dla mężczyzny, K dla kobiety)
        - wiek (liczba 18-80)
        - czas_5km - CAŁKOWITY czas na 5 km w formacie MM:SS (np. "23:00" oznacza "23 minuty na dystansie 5 km")
        - tempo_km - TEMPO na JEDEN kilometr w formacie MM:SS (np. "5:00" oznacza "5 minut na 1 km")
        
        UWAGA - BARDZO WAŻNE:
        - "23 minuty na 5km" → to jest CZAS (czas_5km: "23:00")
        - "tempo 5:00" lub "5 minut na kilometr" → to jest TEMPO (tempo_km: "5:00")
        - Użytkownik poda ALBO czas ALBO tempo
        - Jeśli poda czas w minutach bez sekund, dodaj ":00"
        
        Przykłady:
        - "biegam 5km w 25 minut" → {"czas_5km": "25:00", "tempo_km": null}
        - "moje tempo to 5:00" → {"czas_5km": null, "tempo_km": "5:00"}
        
        UWAGA: Jeśli użytkownik poda prędkość w km/h:
        - Przelicz na tempo: tempo = 60 ÷ prędkość
        - Potem tempo × 5 = czas na 5km
        - Przykład: 12 km/h → tempo 5:00/km → czas 25:00
        Zwróć TYLKO JSON:
        {"gender": "M", "age": 35, "czas_5km": "23:00", "tempo_km": null}
        """
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        llm_response = response.choices[0].message.content.strip()
        st.info(f"🤖 Odpowiedź AI: `{llm_response}`")
        
        # Parsowanie JSON
        extracted_data = None
        try:
            json_match = re.search(r'\{[^}]+\}', llm_response)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                extracted_data = json.loads(llm_response)
        except Exception as e:
            st.error(f"❌ BŁĄD PARSOWANIA JSON: {str(e)}")
            extracted_data = extract_data_fallback(user_input)
        
        # WALIDACJA
        missing_fields = []
        
        # Płeć
        if extracted_data.get("gender"):
            gender_val = str(extracted_data["gender"]).strip().upper()
            if gender_val in ["M", "MĘŻCZYZNA", "MALE"]:
                extracted_data["gender"] = "M"
            elif gender_val in ["K", "KOBIETA", "FEMALE", "F"]:
                extracted_data["gender"] = "K"
            else:
                extracted_data["gender"] = None
                missing_fields.append("płeć")
        else:
            missing_fields.append("płeć")
        
        # Wiek
        if not extracted_data.get("age"):
            missing_fields.append("wiek")
        else:
            try:
                extracted_data["age"] = int(extracted_data["age"])
                if not (18 <= extracted_data["age"] <= 80):
                    missing_fields.append("wiek (poza zakresem)")
                    extracted_data["age"] = None
            except:
                missing_fields.append("wiek")
                extracted_data["age"] = None
        
        # MOJA LOGIKA: Konwersja TEMPO → CZAS
        czas_5km = extracted_data.get("czas_5km")
        tempo_km = extracted_data.get("tempo_km")
        
        final_time_5km = None
        
        if czas_5km:
            time_str = str(czas_5km).strip()
            if re.match(r'^\d{1,2}:\d{2}$', time_str):
                final_time_5km = time_str
            else:
                missing_fields.append("czas (zły format)")
        elif tempo_km:
            tempo_str = str(tempo_km).strip()
            if re.match(r'^\d{1,2}:\d{2}$', tempo_str):
                tempo_seconds = parse_time_to_seconds(tempo_str)
                if tempo_seconds:
                    total_seconds = tempo_seconds * 5
                    minutes = int(total_seconds // 60)
                    seconds = int(total_seconds % 60)
                    final_time_5km = f"{minutes}:{seconds:02d}"
                else:
                    missing_fields.append("tempo (błąd konwersji)")
            else:
                missing_fields.append("tempo (zły format)")
        else:
            missing_fields.append("czas lub tempo")
        
        extracted_data["time_5km"] = final_time_5km
        
        # TWOJA METODA: Generation do trace
        if trace:
            try:
                trace.generation(
                    name="llm_extraction",
                    input=user_input,
                    output=llm_response,
                    model="gpt-4o-mini",
                    metadata={
                        "tokens_used": response.usage.total_tokens,
                        "extracted_gender": extracted_data.get("gender"),
                        "extracted_age": extracted_data.get("age"),
                        "extracted_time_5km": extracted_data.get("time_5km")
                    }
                )
            except Exception as e:
                st.error(f"🔍 Langfuse: Błąd generation - {e}")
        
        # TWOJA METODA: Finalny update
        if trace:
            try:
                trace.update(output={
                    "extracted_data": extracted_data,
                    "missing_fields": missing_fields,
                    "success": len(missing_fields) == 0
                })
            except Exception as e:
                st.error(f"🔍 Langfuse: Błąd update - {e}")
        
        return extracted_data, missing_fields
        
    except Exception as e:
        error_msg = f"Błąd: {str(e)}"
        st.error(f"❌ {error_msg}")
        if trace:
            try:
                trace.event(name="error", input={"error": str(e)})
            except:
                pass
        return None, [error_msg]

def extract_data_fallback(text):
    """Fallback do ekstrakcji danych"""
    data = {"gender": None, "age": None, "czas_5km": None, "tempo_km": None}
    
    text_lower = text.lower()
    if any(word in text_lower for word in ["mężczyzna", "mezczyzna", "męski", "m ", "pan "]):
        data["gender"] = "M"
    elif any(word in text_lower for word in ["kobieta", "żeński", "k ", "pani "]):
        data["gender"] = "K"
    
    age_match = re.search(r'(\b\d{1,2})\s*(lat|latach|roku)', text)
    if age_match:
        data["age"] = int(age_match.group(1))
    
    time_match = re.search(r'(\d{1,2}):(\d{2})', text)
    if time_match:
        data["czas_5km"] = f"{time_match.group(1)}:{time_match.group(2)}"
    
    return data

# ====================== PREDYKCJA ======================
def heuristic_half_marathon_time(gender: str, age: int, five_k_seconds: float) -> float:
    """Heurystyka gdy model nie działa"""
    pace_per_km = five_k_seconds / 5.0
    base_time = pace_per_km * 21.0975
    gender_mult = 1.0 if gender == "M" else 1.05
    
    if age < 35:
        age_mult = 1.0 - min(0.10, (35 - age) * 0.0035)
    else:
        age_mult = 1.0 + min(0.32, (age - 35) * 0.0065)
    
    predicted = base_time * gender_mult * age_mult
    return max(3600, min(predicted, 5 * 3600))

def predict_time(gender: str, age: int, five_k_seconds: float) -> float:
    """Predykcja czasu półmaratonu"""
    if MODEL is not None:
        try:
            sex = 1 if gender == "M" else 0
            tempo_5km = five_k_seconds / 5.0 / 60.0
            X = np.array([[sex, age, tempo_5km]], dtype=float)
            pred_sec = float(MODEL.predict(X)[0])
            return max(3600, min(pred_sec, 5 * 3600))
        except Exception:
            st.warning(f"⚠️ Błąd modelu – heurystyka")
    return heuristic_half_marathon_time(gender, age, five_k_seconds)

def build_splits_dataframe(predicted_seconds: float) -> pd.DataFrame:
    """Czasy pośrednie"""
    checkpoints = [5, 10, 15, 21.0975]
    times = [predicted_seconds * (c / 21.0975) for c in checkpoints]
    data = {
        "Dystans (km)": [f"{c:.1f}" if c != int(c) else f"{int(c)}" for c in checkpoints],
        "Przewidywany czas": [format_seconds_to_hms(t) for t in times],
    }
    return pd.DataFrame(data)

# ====================== WYKRESY ======================
@st.cache_data(ttl=3600, show_spinner=False)
def load_race_data_from_spaces():
    """Pobiera dane zawodów"""
    try:
        key = os.getenv("DO_SPACES_KEY")
        secret = os.getenv("DO_SPACES_SECRET")
        region = os.getenv("DO_SPACES_REGION", "fra1")
        endpoint = os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com")
        bucket = os.getenv("DO_SPACES_BUCKET")
        
        if not all([key, secret, bucket]):
            return None

        s3 = boto3.client("s3", region_name=region, endpoint_url=endpoint,
                         aws_access_key_id=key, aws_secret_access_key=secret)

        dataframes = []
        for filename in ["halfmarathon_wroclaw_2023__final.csv", "halfmarathon_wroclaw_2024__final.csv"]:
            try:
                obj = s3.get_object(Bucket=bucket, Key=filename)
                df = pd.read_csv(io.BytesIO(obj["Body"].read()), sep=";")
                dataframes.append(df)
            except:
                continue

        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        return None
    except:
        return None

def parse_race_time_to_seconds(time_str):
    """Konwertuje HH:MM:SS na sekundy"""
    try:
        if pd.isna(time_str):
            return None
        parts = str(time_str).strip().split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        return None
    except:
        return None

def calculate_race_statistics(df, gender, age):
    """Statystyki z danych"""
    if df is None or df.empty:
        return None
    
    if "Płeć" in df.columns:
        df_filtered = df[df["Płeć"] == gender].copy()
    else:
        df_filtered = df.copy()
    
    df_filtered["time_seconds"] = df_filtered["Czas"].apply(parse_race_time_to_seconds)
    df_filtered = df_filtered.dropna(subset=["time_seconds"])
    
    if df_filtered.empty:
        return None
    
    times = df_filtered["time_seconds"].values
    
    return {
        "top_10": np.percentile(times, 10),
        "top_25": np.percentile(times, 25),
        "median": np.percentile(times, 50),
        "mean": np.mean(times),
        "top_75": np.percentile(times, 75),
        "total_count": len(times)
    }

def get_percentile_and_category(predicted_seconds, stats):
    """Percentyl i kategoria"""
    if stats is None:
        return None, "Nieznana"
    
    if predicted_seconds <= stats["top_10"]:
        return 95, "🏅 Elita"
    elif predicted_seconds <= stats["top_25"]:
        return 85, "🥇 Zaawansowany"
    elif predicted_seconds <= stats["median"]:
        return 65, "🥈 Średnio-zaawansowany"
    elif predicted_seconds <= stats["top_75"]:
        return 40, "🥉 Średni"
    else:
        return 20, "🏃 Początkujący"

def create_comparison_chart(predicted_seconds, stats):
    """Wykres słupkowy"""
    if stats is None:
        return None
    
    categories = ["Top 10%", "Top 25%", "Twój czas", "Średnia"]
    values = [stats["top_10"], stats["top_25"], predicted_seconds, stats["mean"]]
    colors = ["#28a745", "#17a2b8", "#fd7e14", "#6c757d"]
    text_labels = [format_seconds_to_hms(v) for v in values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=values, text=text_labels, textposition="outside",
        marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.3)', width=2)),
        hovertemplate="<b>%{x}</b><br>Czas: %{text}<extra></extra>"
    ))
    
    fig.update_layout(
        title={'text': "📊 Twój czas na tle innych", 'x': 0.5, 'xanchor': 'center',
               'font': {'size': 18, 'color': '#212529'}},
        yaxis_title="Czas (sekundy)", height=400,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, color='#212529'),
        margin=dict(t=80, b=60, l=60, r=40),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        xaxis=dict(showgrid=False)
    )
    
    return fig

# ====================== STAN SESJI ======================
if "gender" not in st.session_state:
    st.session_state.gender = "M"
if "age" not in st.session_state:
    st.session_state.age = 35
if "time5" not in st.session_state:
    st.session_state.time5 = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "splits" not in st.session_state:
    st.session_state.splits = None
if "llm_used" not in st.session_state:
    st.session_state.llm_used = False

# ====================== INTERFEJS ======================
st.markdown('<h1 class="main-title">🏃 Szacowanie czasu półmaratonu</h1>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("🤖 Wprowadzanie danych przez AI")

user_input = st.text_area(
    "Twój opis:", 
    placeholder="Jestem mężczyzną, mam 35 lat, biegam 5km w 23 minuty...",
    height=100
)

llm_submit = st.button("🎯 Analizuj tekst AI", key="llm_analyze")

if llm_submit and user_input:
    with st.spinner("AI analizuje..."):
        extracted_data, missing_fields = extract_running_data_with_llm(user_input)
    
    if extracted_data:
        st.success("✅ AI przetworzyło tekst!")
        
        if extracted_data.get("gender") in ["M", "K"]:
            st.session_state.gender = extracted_data["gender"]
            st.success(f"✅ Płeć: {extracted_data['gender']}")
        
        if extracted_data.get("age") and 18 <= extracted_data["age"] <= 80:
            st.session_state.age = extracted_data["age"]
            st.success(f"✅ Wiek: {extracted_data['age']}")
        
        if extracted_data.get("time_5km"):
            st.session_state.time5 = extracted_data["time_5km"]
            st.success(f"✅ Czas 5km: {extracted_data['time_5km']}")
        
        st.session_state.llm_used = True
        
        if missing_fields:
            st.warning(f"⚠️ Brakuje: {', '.join(missing_fields)}")
        
        st.rerun()
    else:
        st.error("❌ Błąd przetwarzania")

st.markdown("---")
st.subheader("🎯 Dane wejściowe" + (" (AI)" if st.session_state.llm_used else ""))

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Płeć", options=["M", "K"], key="gender")
    age = st.number_input("Wiek", min_value=18, max_value=80, step=1, key="age")
with col2:
    time5_text = st.text_input("Czas na 5 km (MM:SS)", placeholder="23:00", key="time5")

submit = st.button("🚀 Sprawdź wynik", use_container_width=True, key="predict", type="primary")

if submit:
    five_k_seconds = parse_time_to_seconds(time5_text)
    
    if five_k_seconds is None:
        st.error("❌ Podaj czas MM:SS")
    elif not (300 <= five_k_seconds <= 3600):
        st.error("❌ Czas 5km: 15:00-60:00")
    else:
        predicted = predict_time(gender, int(age), five_k_seconds)
        st.session_state.prediction = predicted
        st.session_state.splits = build_splits_dataframe(predicted)

# ====================== WYNIKI ======================
if st.session_state.prediction is not None:
    predicted_seconds = st.session_state.prediction
    time_str = format_seconds_to_hms(predicted_seconds)
    
    st.markdown(f"""
<div style='border: 3px solid #fd7e14; border-radius: 16px; padding: 1.5rem; background: white; margin-top: 2rem;'>
    <div style='background: linear-gradient(135deg, #e8590c, #fd7e14); color: white; padding: 0.8rem; border-radius: 10px; text-align: center; font-size: 1.4rem; font-weight: 700; margin-bottom: 1.2rem;'>
        🏆 WYNIK
    </div>
    <div style='background-color: #d4edda; border: 1px solid #c3e6cb; border-left: 3px solid #28a745; border-radius: 12px; padding: 0.95rem 1.2rem; text-align: center;'>
        ✅ Czas półmaratonu: <strong>{time_str}</strong>
        <br>Dokładność: ±5 minut (MAE)
    </div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("### 📋 Czasy pośrednie")
    st.dataframe(st.session_state.splits, use_container_width=True, hide_index=True)
    
    # WYKRESY
    with st.spinner("Ładowanie porównań..."):
        race_data = load_race_data_from_spaces()
    
    if race_data is not None:
        stats = calculate_race_statistics(race_data, gender, int(age))
        
        if stats:
            percentile, category = get_percentile_and_category(predicted_seconds, stats)
            
            st.markdown("---")
            st.markdown("### 📊 Porównanie z innymi")
            
            chart = create_comparison_chart(predicted_seconds, stats)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if percentile:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #28a745, #20c997); color: white; 
                                padding: 1.2rem; border-radius: 12px; text-align: center;'>
                        <div style='font-size: 2.5rem; font-weight: 700;'>{percentile}%</div>
                        <div style='font-size: 1rem; margin-top: 0.5rem;'>
                            Szybszy niż {percentile}% uczestników
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                            padding: 1.2rem; border-radius: 12px; text-align: center;'>
                    <div style='font-size: 2rem; font-weight: 700;'>{category}</div>
                    <div style='font-size: 1rem; margin-top: 0.5rem;'>
                        Kategoria
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"📈 {stats['total_count']} uczestników Wrocław 2023-2024")
    
    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        st.success("🔮 **Model ML**" if MODEL else "🔮 **Heurystyka**")
    with cols[1]:
        st.info("🎯 **MAE:** ~5 min")
    with cols[2]:
        st.success("🤖 **AI**" if st.session_state.llm_used else "📝 **Ręcznie**")

st.markdown("---")
st.caption("🏃 ML + AI (GPT-4o-mini) + Langfuse | MAE: ~5 min")