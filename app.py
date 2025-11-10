import os
import io
import joblib
import boto3
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from dotenv import load_dotenv
import openai
from langfuse import Langfuse
import json
import re

# Załaduj zmienne z .env
load_dotenv()

# Inicjalizacja Langfuse
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

# ====================== STYLE ======================
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
  .metric-card {
    background: white;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(33, 37, 41, 0.15);
    padding: 1.1rem;
    height: 100%;
    border-left: 4px solid #fd7e14;
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
  .info-box {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 0.95rem 1.2rem;
    margin-top: 0.5rem;
    box-shadow: 0 4px 12px rgba(108, 117, 125, 0.15);
    border-left: 3px solid #fd7e14;
  }
  .stTable { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(33, 37, 41, 0.1); }
  .stTable table { color: #212529 !important; width: 100%; }
  .stTable th { background-color: #495057 !important; color: white !important; font-weight: 600; padding: 12px 16px; }
  .stTable td { background-color: white !important; color: #212529 !important; padding: 10px 16px; border-bottom: 1px solid #dee2e6; }
  .stTable tr:nth-child(even) td { background-color: #f8f9fa !important; }
  .stTable tr:hover td { background-color: #fff3cd !important; }
  .llm-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ====================== DO SPACES – ŁADOWANIE MODELU ======================
@st.cache_resource(show_spinner=False)
def load_model_from_spaces():
    """
    Pobiera i wczyta model.pkl z DigitalOcean Spaces.
    """
    try:
        key = os.getenv("DO_SPACES_KEY")
        secret = os.getenv("DO_SPACES_SECRET")
        region = os.getenv("DO_SPACES_REGION", "fra1")
        endpoint = os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com")
        bucket = os.getenv("DO_SPACES_BUCKET")
        model_key = os.getenv("DO_SPACES_MODEL_KEY", "trained_model.pkl")

        if not all([key, secret, bucket]):
            st.info("ℹ️ Nie ustawiono pełnych zmiennych dla DO Spaces – użyję heurystyki.")
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

# ====================== LLM + LANGFUSE FUNCTIONS ======================
def extract_running_data_with_llm(user_input):
    """
    Ekstrakcja danych biegowych z tekstu użytkownika przy użyciu LLM
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None, ["Brak klucza OpenAI API"]
    
    trace = None
    if langfuse:
        try:
            trace = langfuse.trace(
                name="llm_half_marathon_data_extraction",
                input={"user_input": user_input}
            )
            st.success("🔍 Langfuse: Trace utworzony")
        except Exception as e:
            st.error(f"🔍 Langfuse: Błąd trace - {e}")
    
    try:
        # Prompt dla LLM
        system_prompt = """
        Jesteś asystentem do ekstrakcji danych biegowych. Wyciągnij z tekstu użytkownika:
        - płeć (M dla mężczyzny, K dla kobiety)
        - wiek (liczba 18-80)
        - czas na 5km (w formacie MM:SS, np. 25:30)
        
        Jeśli jakichś danych brakuje, zwróć null dla tych pól.
        Zwróć dane TYLKO w formacie JSON, bez dodatkowego tekstu.
        
        Przykład poprawnego JSON:
        {"gender": "M", "age": 35, "pace_5km": "25:30"}
        """
        
        # Wywołanie OpenAI
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        llm_response = response.choices[0].message.content.strip()
        
        # DEBUG: Pokaż odpowiedź LLM
        st.info(f"🤖 Odpowiedź AI: `{llm_response}`")
        
        # Parsowanie JSON
        try:
            # Spróbuj znaleźć JSON w odpowiedzi
            json_match = re.search(r'\{[^}]+\}', llm_response)
            if json_match:
                json_str = json_match.group()
                extracted_data = json.loads(json_str)
            else:
                extracted_data = json.loads(llm_response)
        except Exception as e:
            st.warning(f"⚠️ Błąd parsowania JSON, używam fallback: {e}")
            # Fallback - użyj regex
            extracted_data = extract_data_fallback(user_input)
        
        # Logowanie do Langfuse
        if trace:
            try:
                trace.generation(
                    name="llm_extraction",
                    input=user_input,
                    output=extracted_data,
                    model="gpt-3.5-turbo",
                    metadata={
                        "tokens_used": response.usage.total_tokens,
                    }
                )
                st.success("🔍 Langfuse: Generation zapisane")
            except Exception as e:
                st.error(f"🔍 Langfuse: Błąd generation - {e}")
        
        # Walidacja i konwersja danych
        missing_fields = []
        
        # Sprawdź i przekonwertuj płeć
        if extracted_data.get("gender"):
            gender_val = str(extracted_data["gender"]).strip().upper()
            if gender_val in ["M", "MĘŻCZYZNA", "MALE"]:
                extracted_data["gender"] = "M"
            elif gender_val in ["K", "KOBIETA", "FEMALE"]:
                extracted_data["gender"] = "K"
            else:
                extracted_data["gender"] = None
                missing_fields.append("płeć")
        else:
            missing_fields.append("płeć")
            
        # Sprawdź wiek
        if not extracted_data.get("age"):
            missing_fields.append("wiek")
        else:
            # Upewnij się że wiek jest liczbą
            try:
                extracted_data["age"] = int(extracted_data["age"])
                if extracted_data["age"] < 18 or extracted_data["age"] > 80:
                    missing_fields.append("wiek (poza zakresem 18-80)")
                    extracted_data["age"] = None
            except:
                missing_fields.append("wiek")
                extracted_data["age"] = None
                
        # Sprawdź czas 5km
        if not extracted_data.get("pace_5km"):
            missing_fields.append("czas na 5km")
        else:
            # Upewnij się że czas jest w formacie MM:SS
            time_str = str(extracted_data["pace_5km"]).strip()
            if not re.match(r'^\d{1,2}:\d{2}$', time_str):
                missing_fields.append("czas na 5km (zły format)")
                extracted_data["pace_5km"] = None
        
        # Finalny update Langfuse
        if trace:
            try:
                trace.update(output={
                    "extracted_data": extracted_data,
                    "missing_fields": missing_fields,
                    "success": len(missing_fields) == 0
                })
                st.success("🔍 Langfuse: Trace zaktualizowany")
            except Exception as e:
                st.error(f"🔍 Langfuse: Błąd update - {e}")
        
        return extracted_data, missing_fields
        
    except Exception as e:
        error_msg = f"Błąd przetwarzania: {str(e)}"
        st.error(f"❌ {error_msg}")
        if trace:
            try:
                trace.event(name="processing_error", input={"error": str(e)})
            except Exception as trace_error:
                st.error(f"🔍 Langfuse: Błąd error logging - {trace_error}")
        return None, [error_msg]

def extract_data_fallback(text):
    """Fallback do ekstrakcji danych gdy LLM zawiedzie"""
    data = {"gender": None, "age": None, "pace_5km": None}
    
    # Ekstrakcja płci
    text_lower = text.lower()
    if any(word in text_lower for word in ["mężczyzna", "mezczyzna", "męski", "m ", "pan ", "male"]):
        data["gender"] = "M"
    elif any(word in text_lower for word in ["kobieta", "żeński", "k ", "pani ", "female"]):
        data["gender"] = "K"
    
    # Ekstrakcja wieku
    age_match = re.search(r'(\b\d{1,2})\s*(lat|latach|roku|latka|lat|years|year)', text)
    if age_match:
        data["age"] = int(age_match.group(1))
    else:
        # Szukaj liczby 18-80
        numbers = re.findall(r'\b(1[89]|[2-7][0-9]|80)\b', text)
        if numbers:
            data["age"] = int(numbers[0])
    
    # Ekstrakcja czasu 5km
    time_match = re.search(r'(\d{1,2}):(\d{2})', text)
    if time_match:
        minutes = time_match.group(1)
        seconds = time_match.group(2)
        data["pace_5km"] = f"{minutes}:{seconds}"
    
    return data

# ====================== RESZTA FUNKCJI (bez zmian) ======================
# ... (tutaj wklej wszystkie pozostałe funkcje z poprzedniego kodu:
# get_comparison_data, classify_position, create_comparison_chart, 
# create_pace_progression_chart, parse_time_to_seconds, format_seconds_to_hms,
# human_readable_timedelta, heuristic_half_marathon_time, predict_time, 
# build_splits_dataframe)

# ====================== DANE PORÓWNAWCZE ======================
def get_comparison_data(gender, age, predicted_time):
    age_groups = {
        '18-25': {'M': 7200, 'K': 8100},
        '26-35': {'M': 7500, 'K': 8400},
        '36-45': {'M': 7800, 'K': 8700},
        '46-55': {'M': 8100, 'K': 9000},
        '56+':   {'M': 8400, 'K': 9300}
    }
    if age <= 25:
        age_group = '18-25'
    elif age <= 35:
        age_group = '26-35'
    elif age <= 45:
        age_group = '36-45'
    elif age <= 55:
        age_group = '46-55'
    else:
        age_group = '56+'

    avg_time = age_groups[age_group][gender]
    comparison_data = {
        'Twój wynik': predicted_time,
        'Średnia w grupie': avg_time,
        'Wynik elitarny': avg_time * 0.70,
        'Wynik dobry': avg_time * 0.85,
        'Wynik przeciętny': avg_time * 1.00,
        'Wynik początkujący': avg_time * 1.15
    }
    return comparison_data, age_group

def classify_position(predicted_seconds: float, group_avg_seconds: float) -> str:
    ratio = predicted_seconds / group_avg_seconds
    if ratio <= 0.80:
        return "🏅 Elitarny"
    elif ratio <= 0.95:
        return "💪 Dobry"
    elif ratio <= 1.10:
        return "🙂 Przeciętny"
    else:
        return "🚀 Początkujący"

def create_comparison_chart(comparison_data, age_group, gender):
    categories = list(comparison_data.keys())
    times = list(comparison_data.values())
    time_strings = [format_seconds_to_hms(t) for t in times]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=times,
        text=time_strings,
        textposition='auto',
        marker_color=['#fd7e14', '#6c757d', '#28a745', '#20c997', '#ffc107', '#dc3545']
    ))
    fig.update_layout(
        title=f'Porównanie z innymi biegaczami ({gender}, {age_group})',
        xaxis_title='Kategoria',
        yaxis_title='Czas (sekundy)',
        showlegend=False,
        template='plotly_white',
        font=dict(size=12)
    )
    fig.update_yaxes(tickformat='%H:%M:%S')
    return fig

def create_pace_progression_chart(predicted_time):
    distances = [5, 10, 15, 21.0975]
    times = [predicted_time * (d / 21.0975) for d in distances]
    paces = [t / d for d, t in zip(distances, times)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distances,
        y=paces,
        mode='lines+markers',
        name='Tempo',
        line=dict(color='#fd7e14', width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title='Progresja tempa na dystansie',
        xaxis_title='Dystans (km)',
        yaxis_title='Tempo (s/km)',
        template='plotly_white'
    )
    fig.update_yaxes(tickformat='%M:%S', title='Tempo (min/km)')
    return fig

# ====================== STAN SESJI ======================
if "gender" not in st.session_state:
    st.session_state.gender = "M"
if "age" not in st.session_state:
    st.session_state.age = 35
if "pace" not in st.session_state:
    st.session_state.pace = "05:00"
if "time5" not in st.session_state:
    st.session_state.time5 = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "splits" not in st.session_state:
    st.session_state.splits = None
if "llm_used" not in st.session_state:
    st.session_state.llm_used = False

# ====================== FUNKCJE POMOCNICZE ======================
def parse_time_to_seconds(text: str) -> int | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    parts = stripped.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        values = [int(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 2:
        minutes, seconds = values
        hours = 0
    else:
        hours, minutes, seconds = values
    if seconds >= 60 or seconds < 0 or minutes < 0 or hours < 0:
        return None
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

def format_seconds_to_hms(seconds: float) -> str:
    sec = int(round(seconds))
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    rest_seconds = sec % 60
    return f"{hours}:{minutes:02d}:{rest_seconds:02d}"

def human_readable_timedelta(seconds: float) -> str:
    td = timedelta(seconds=int(round(seconds)))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_left = total_seconds % 60
    parts = []
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes:02d}m")
    parts.append(f"{seconds_left:02d}s")
    return " ".join(parts)

# ====== HEURYSTYKA (fallback) ======
def heuristic_half_marathon_time(gender: str, age: int, pace_seconds: float, five_k_seconds: float | None) -> float:
    distance_km = 21.0975
    pace_candidates = [pace_seconds]
    if five_k_seconds is not None:
        pace_candidates.append(five_k_seconds / 5.0)
    effective_pace = sum(pace_candidates) / len(pace_candidates)
    base_time = effective_pace * distance_km
    gender_multiplier = 1.0 if gender == "M" else 1.05
    if age < 35:
        age_multiplier = 1.0 - min(0.10, (35 - age) * 0.0035)
    else:
        age_multiplier = 1.0 + min(0.32, (age - 35) * 0.0065)
    predicted_seconds = base_time * gender_multiplier * age_multiplier
    predicted_seconds = max(3600, min(predicted_seconds, 5 * 3600))
    return predicted_seconds

# ====== PREDYKCJA: model z Spaces albo heurystyka ======
def predict_time(gender: str, age: int, pace_seconds: float, five_k_seconds: float | None) -> float:
    """
    Jeśli dostępny model (MODEL != None), używa ML:
        features = [sex(M->1,K->0), age, pace_seconds]
    W innym razie – heurystyka.
    """
    if MODEL is not None:
        try:
            sex = 1 if gender == "M" else 0
            X = np.array([[sex, age, pace_seconds]], dtype=float)
            pred_sec = float(MODEL.predict(X)[0])
            # sanity clamp – na wypadek „dziwnych” wartości modelu
            pred_sec = max(3600, min(pred_sec, 5 * 3600))
            return pred_sec
        except Exception as e:
            st.warning(f"⚠️ Błąd predykcji modelem – użyję heurystyki")
    return heuristic_half_marathon_time(gender, age, pace_seconds, five_k_seconds)

def build_splits_dataframe(predicted_seconds: float) -> pd.DataFrame:
    checkpoints = [5, 10, 15, 21.0975]
    data = {
        "Dystans (km)": [f"{c:.1f}" if c != int(c) else f"{int(c)}" for c in checkpoints],
        "Przewidywany czas": [human_readable_timedelta(predicted_seconds * (c / 21.0975)) for c in checkpoints],
    }
    return pd.DataFrame(data)

# ====================== INTERFEJS UŻYTKOWNIKA ======================
st.markdown('<h1 class="main-title">🏃 Szacowanie czasu półmaratonu</h1>', unsafe_allow_html=True)
st.markdown("**Płeć + wiek + tempo → przewidywany wynik 21.1 km. Dokładność orientacyjna: ±5 min (MAE).**")

# ====================== SEKCJA LLM ======================
st.markdown("---")
st.markdown('<div class="llm-section">', unsafe_allow_html=True)
st.subheader("🤖 Wprowadzanie danych przez AI")
st.markdown("Opisz się, a AI wyciągnie potrzebne dane:")

user_input = st.text_area(
    "Twój opis:", 
    placeholder="Jestem mężczyzną, mam 35 lat, biegam 5km w 25 minut...",
    height=100
)

llm_submit = st.button("🎯 Analizuj tekst AI", key="llm_analyze", type="secondary")

if llm_submit and user_input:
    with st.spinner("AI analizuje tekst..."):
        extracted_data, missing_fields = extract_running_data_with_llm(user_input)
    
    if extracted_data:
        st.success("✅ AI pomyślnie przetworzyło tekst!")
        
        # Wypełnij formularz danymi z LLM - POPRAWIONE!
        if extracted_data.get("gender") in ["M", "K"]:
            st.session_state.gender = extracted_data["gender"]
            st.success(f"✅ Płeć ustawiona na: {extracted_data['gender']}")
        
        if extracted_data.get("age") and 18 <= extracted_data["age"] <= 80:
            st.session_state.age = extracted_data["age"]
            st.success(f"✅ Wiek ustawiony na: {extracted_data['age']}")
        
        if extracted_data.get("pace_5km"):
            st.session_state.pace = extracted_data["pace_5km"]
            st.success(f"✅ Tempo ustawione na: {extracted_data['pace_5km']}")
        
        st.session_state.llm_used = True
        
        # Pokaż wyciągnięte dane
        st.info(f"""
        **Wyciągnięte dane:**
        - **Płeć:** {extracted_data.get('gender', 'Nie znaleziono')}
        - **Wiek:** {extracted_data.get('age', 'Nie znaleziono')}
        - **Czas 5km:** {extracted_data.get('pace_5km', 'Nie znaleziono')}
        """)
        
        if missing_fields:
            st.warning(f"⚠️ Brakujące dane: {', '.join(missing_fields)}")
            
        # Odśwież stronę aby pokazać zaktualizowane dane
        st.rerun()
    else:
        st.error("❌ Nie udało się przetworzyć tekstu")

st.markdown('</div>', unsafe_allow_html=True)

# ====================== FORMULARZ WEJŚCIA ======================
st.markdown("---")
st.subheader("🎯 Dane wejściowe" + (" (wypełnione przez AI)" if st.session_state.llm_used else ""))

input_columns = st.columns(2)
with input_columns[0]:
    gender = st.selectbox("Płeć", options=["M", "K"], key="gender")
    age = st.number_input("Wiek", min_value=18, max_value=80, step=1, key="age")
with input_columns[1]:
    pace_text = st.text_input("Tempo na 1 km (MM:SS)", key="pace")
    time5_text = st.text_input("Czas na 5 km (MM:SS) — opcjonalnie", placeholder="np. 24:30", key="time5")

submit = st.button("🚀 Sprawdź wynik", use_container_width=True, key="predict", type="primary")

# ====================== PRZETWARZANIE ======================
if submit:
    pace_seconds = parse_time_to_seconds(pace_text)
    five_k_seconds = parse_time_to_seconds(time5_text) if time5_text else None
    if pace_seconds is None:
        st.error("❌ Podaj tempo w formacie MM:SS, np. 05:10.")
    elif not (150 <= pace_seconds <= 900):
        st.error("❌ Tempo powinno mieścić się między 2:30 a 15:00 min/km.")
    elif five_k_seconds is not None and not (900 <= five_k_seconds <= 3600):
        st.error("❌ Czas 5 km powinien mieścić się między 15:00 a 60:00.")
    else:
        predicted = predict_time(gender, int(age), pace_seconds, five_k_seconds)
        st.session_state.prediction = predicted
        st.session_state.splits = build_splits_dataframe(predicted)

# ====================== WYNIKI ======================
if st.session_state.prediction is not None:
    predicted_seconds = st.session_state.prediction
    avg_pace = predicted_seconds / 21.0975
    speed = 21.0975 / (predicted_seconds / 3600)

    time_str = format_seconds_to_hms(predicted_seconds)
    pace_str = f"{int(avg_pace // 60)}:{int(avg_pace % 60):02d} / km"
    speed_str = f"{speed:.2f} km/h"

    comparison_data, age_group = get_comparison_data(gender, age, predicted_seconds)
    group_avg = comparison_data["Średnia w grupie"]
    position_label = classify_position(predicted_seconds, group_avg)

    html_code = f"""
<div style='border: 3px solid #fd7e14; border-radius: 16px; padding: 1.5rem; background: white; box-shadow: 0 8px 24px rgba(253, 126, 20, 0.2); margin-top: 2rem; margin-bottom: 2rem;'>
    <div style='background: linear-gradient(135deg, #e8590c, #fd7e14); color: white; padding: 0.8rem; border-radius: 10px; text-align: center; font-size: 1.4rem; font-weight: 700; margin-bottom: 1.2rem; box-shadow: 0 4px 12px rgba(253, 126, 20, 0.3);'>
        🏆 WYNIK
    </div>
    <div style='background-color: #d4edda; border: 1px solid #c3e6cb; border-left: 3px solid #28a745; border-radius: 12px; padding: 0.95rem 1.2rem; margin-bottom: 1.0rem; box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);'>
        ✅ Szacowany czas półmaratonu: <strong>{time_str}</strong> – dokładność orientacyjna ±5 minut.
    </div>
    <div style='background-color: #fff3cd; border: 1px solid #ffeeba; border-left: 3px solid #fd7e14; border-radius: 12px; padding: 0.75rem 1.0rem; margin-bottom: 1.2rem;'>
        📍 <strong>Twoja pozycja w grupie ({gender}, {age_group}):</strong> {position_label}
    </div>
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
        <div style='background-color: #e9ecef; border-left: 3px solid #fd7e14; border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 4px 12px rgba(108, 117, 125, 0.15);'>
            <div style='font-size: 0.9rem; color: #495057; margin-bottom: 0.3rem;'>⏱️ <strong>Szacowany czas:</strong></div>
            <div style='font-size: 1.3rem; font-weight: 600; color: #212529;'>{time_str}</div>
        </div>
        <div style='background-color: #e9ecef; border-left: 3px solid #fd7e14; border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 4px 12px rgba(108, 117, 125, 0.15);'>
            <div style='font-size: 0.9rem; color: #495057; margin-bottom: 0.3rem;'>📊 <strong>Średnie tempo:</strong></div>
            <div style='font-size: 1.3rem; font-weight: 600; color: #212529;'>{pace_str}</div>
        </div>
        <div style='background-color: #e9ecef; border-left: 3px solid #fd7e14; border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 4px 12px rgba(108, 117, 125, 0.15);'>
            <div style='font-size: 0.9rem; color: #495057; margin-bottom: 0.3rem;'>⚡ <strong>Średnia prędkość:</strong></div>
            <div style='font-size: 1.3rem; font-weight: 600; color: #212529;'>{speed_str}</div>
        </div>
    </div>
</div>
"""
    st.markdown(html_code, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Analiza porównawcza")

    col1, col2 = st.columns(2)
    with col1:
        fig_comparison = create_comparison_chart(comparison_data, age_group, gender)
        st.plotly_chart(fig_comparison, use_container_width=True, key="comparison_chart")

    with col2:
        fig_pace = create_pace_progression_chart(predicted_seconds)
        st.plotly_chart(fig_pace, use_container_width=True, key="pace_chart")

    st.markdown("### 📋 Czasy pośrednie")
    st.dataframe(st.session_state.splits, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💡 Interpretacja wyników")
    st.markdown(f"**📍 Twoja pozycja w grupie:** {position_label}")
    info_cols = st.columns(2)
    with info_cols[0]:
        st.markdown("""
**🎯 Pozycja w grupie – legenda:**
- **🏅 Elitarny:** ~Top 10–15% biegaczy
- **💪 Dobry:** lepszy niż ~75–85%
- **🙂 Przeciętny:** blisko średniej w grupie
- **🚀 Początkujący:** pierwsze starty / powyżej średniej
        """)
    with info_cols[1]:
        st.markdown("""
**📊 O statystykach:**
- Dane syntetyczne na bazie polskich biegów (orientacyjnie)
- Uwzględniono podział na płeć i wiek
- Aktualizowane sezonowo
- Wynik służy wyłącznie celom informacyjnym
        """)
    
    # Informacja o źródle predykcji
    st.markdown("---")
    source_cols = st.columns(3)
    with source_cols[0]:
        if MODEL is not None:
            st.success("🔮 **Źródło predykcji:** Model ML z DigitalOcean Spaces")
        else:
            st.info("🔮 **Źródło predykcji:** Heurystyka ekspercka")
    with source_cols[1]:
        st.info("🎯 **Dokładność:** ±5 minut (MAE)")
    with source_cols[2]:
        if st.session_state.llm_used:
            st.success("🤖 **Dane wprowadzone przez AI**")
        else:
            st.info("📝 **Dane wprowadzone ręcznie**")

st.markdown("---")
st.caption("🏃 Aplikacja do szacowania czasu półmaratonu | Model ML + AI + Langfuse | Dokładność ±5 min")