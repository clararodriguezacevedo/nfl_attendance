import streamlit as st
import pandas as pd
import numpy as np
import time
import csv
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
)

df = pd.read_csv("./csv_files/engineered/nfl_games_attendance.csv")


### Setup para los modelos con el df regular

X = df.drop("game_attendance", axis=1)
y = df.game_attendance

# Identificar columnas numéricas y categóricas
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Preprocesamiento para variables numéricas (normalización) y categóricas (one-hot encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

regular_scores = {
    "efficiency": {
        "linear": 0,
        "poly": 0,
        "lasso": 0,
        "ridge": 0,
    },
    "time": {
        "linear": 0,
        "poly": 0,
        "lasso": 0,
        "ridge": 0,
    },
    "precision": {
        "linear": 0,
        "poly": 0,
        "lasso": 0,
        "ridge": 0,
    },
}

# Crear un pipeline que combine preprocesamiento y modelo de regresión lineal
linear_reg_model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

### Modelos entrenados con el df regular

linear_reg_start_time = time.time()

# Entrenar en el set de entrenamiento
linear_reg_model.fit(X_train, y_train)

linear_reg_end_time = time.time()

# Darle puntaje al modelo
linear_reg_score = linear_reg_model.score(X_test, y_test)

# Calcular tiempo transcurrido
linear_reg_time = linear_reg_end_time - linear_reg_start_time

# print(f"Linear Regression Model Time Spent: {linear_reg_time}")
# print(f"Linear Regression Model Score: {linear_reg_score:0.3f}")

regular_scores["precision"]["linear"] = linear_reg_score
regular_scores["time"]["linear"] = linear_reg_time


# Crear un pipeline que combine preprocesamiento y modelo de regresión lasso
lasso_model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", Lasso(alpha=20))]
)

lasso_start_time = time.time()

# Entrenar en el set de entrenamiento
lasso_model.fit(X_train, y_train)

lasso_end_time = time.time()

# Darle puntaje al modelo
lasso_score = lasso_model.score(X_test, y_test)

# Calcular tiempo transcurrido
lasso_time = lasso_end_time - lasso_start_time

# print(f"Lasso Regression Model Time Spent: {lasso_time}")
# print(f"Lasso Model Score: {lasso_score:0.3f}")

regular_scores["precision"]["lasso"] = lasso_score
regular_scores["time"]["lasso"] = lasso_time


# Crear un pipeline que combine preprocesamiento y modelo de regresión ridge
ridge_model = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=4))]
)

ridge_start_time = time.time()

# Entrenar en el set de entrenamiento
ridge_model.fit(X_train, y_train)

ridge_end_time = time.time()

# Darle puntaje al modelo
ridge_score = ridge_model.score(X_test, y_test)

# Calcular tiempo transcurrido
ridge_time = ridge_end_time - ridge_start_time

# print(f"Ridge Regression Model Time Spent: {ridge_time}")
# print(f"Ridge Model Score: {ridge_score:0.3f}")

regular_scores["precision"]["ridge"] = ridge_score
regular_scores["time"]["ridge"] = ridge_time

def process_input_data(sample_model, sample_week, sample_home_team, sample_away_team, sample_date, sample_time, sample_pts_win, sample_pts_loss, sample_home_team_previous_year_performance, sample_away_team_previous_year_performance, sample_home_team_current_year_performance, sample_away_team_current_year_performance, sample_home_team_current_sos, sample_away_team_current_sos, sample_weather_condition):
    print(sample_model,sample_week, sample_home_team, sample_away_team, sample_date, sample_time, sample_pts_win, sample_pts_loss, sample_home_team_previous_year_performance, sample_away_team_previous_year_performance, sample_home_team_current_year_performance, sample_away_team_current_year_performance, sample_home_team_current_sos, sample_away_team_current_sos, sample_weather_condition)
    # print(st.session_state.week,
    #       st.session_state.home_team,
    #       st.session_state.away_team,
    #       st.session_state.date,
    #       st.session_state.time,
    #       st.session_state.pts_win,
    #       st.session_state.pts_loss,
    #       st.session_state.home_team_previous_year_performance,
    #       st.session_state.away_team_previous_year_performance,
    #       st.session_state.home_team_current_year_performance,
    #       st.session_state.away_team_current_year_performance,
    #       st.session_state.home_team_current_sos,
    #       st.session_state.away_team_current_sos,
    #       st.session_state.weather_condition)

    # sample_week, sample_home_team, sample_away_team, sample_date, sample_time, sample_pts_win, sample_pts_loss, sample_home_team_previous_year_performance, sample_away_team_previous_year_performance, sample_home_team_current_year_performance, sample_away_team_current_year_performance, sample_home_team_current_sos, sample_away_team_current_sos, sample_weather_condition = st.session_state.week, st.session_state.home_team, st.session_state.away_team, st.session_state.date, st.session_state.time, st.session_state.pts_win, st.session_state.pts_loss, st.session_state.home_team_previous_year_performance, st.session_state.away_team_previous_year_performance, st.session_state.home_team_current_year_performance, st.session_state.away_team_current_year_performance, st.session_state.home_team_current_sos, st.session_state.away_team_current_sos, st.session_state.weather_condition

    # Serie de diccionarios para mappear los variables no dadas a partir de las dadas (por ejemplo, el estadio y la ciudad a partir del equipo local)
    days_week = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun",
    }

    home_team_names = {
        "Chiefs": "Kansas City Chiefs",
        "Raiders": "Oakland Raiders",
        "Falcons": "Atlanta Falcons",
        "Bills": "Buffalo Bills",
        "Steelers": "Pittsburgh Steelers",
        "Ravens": "Baltimore Ravens",
        "Eagles": "Philadelphia Eagles",
        "Lions": "Detroit Lions",
        "Jaguars": "Jacksonville Jaguars",
        "Rams": "Los Angeles Rams",
        "Panthers": "Carolina Panthers",
        "Packers": "Green Bay Packers",
        "Cowboys": "Dallas Cowboys",
        "Vikings": "Minnesota Vikings",
        "Broncos": "Denver Broncos",
        "Texans": "Houston Texans",
        "Cardinals": "Arizona Cardinals",
        "Patriots": "New England Patriots",
        "Buccaneers": "Tampa Bay Buccaneers",
        "Titans": "Tennessee Titans",
        "Dolphins": "Miami Dolphins",
        "Seahawks": "Seattle Seahawks",
        "Redskins": "Washington Redskins",
        "Commanders": "Washington Redskins",
        "Jets": "New York Jets",
        "Colts": "Indianapolis Colts",
        "Bears": "Chicago Bears",
        "Saints": "New Orleans Saints",
        "49ers": "San Francisco 49ers",
        "Browns": "Cleveland Browns",
        "Bengals": "Cincinnati Bengals",
        "Chargers": "Los Angeles Chargers",
        "Giants": "New York Giants",
    }

    stadium_names = {
        "Bears": "Soldier Field",
        "Packers": "Lambeau Field",
        "Chiefs": "Arrowhead",
        "Bills": "Highmark",
        "Saints": "Caesars Superdome",
        "Dolphins": "Hard Rock",
        "Jaguars": "EverBank",
        "Panthers": "Bank of America",
        "Redskins": "Northwest",
        "Ravens": "M&T Bank",
        "Buccaneers": "Raymond James",
        "Browns": "Huntington Bank",
        "Titans": "Nissan",
        "Bengals": "Paycor",
        "Broncos": "Empower Field at Mile High",
        "Steelers": "Acrisure",
        "Lions": "Ford Field",
        "Texans": "NRG",
        "Patriots": "Gillette",
        "Seahawks": "Lumen Field",
        "Eagles": "Lincoln Financial Field",
        "Cardinals": "State Farm",
        "Colts": "Lucas Oil",
        "Cowboys": "AT&T",
        "Jets": "MetLife",
        "Giants": "MetLife",
        "49ers": "Levi's",
        "Vikings": "US Bank",
        "Falcons": "Mercedes-Benz",
        "Raiders": "Oakland Coliseum",
        "Rams": "Los Angeles Memorial Coliseum",
        "Chargers": "Dignity Health Sports Park",
    }

    stadium_max_capacities = {
        "Bears": 66944,
        "Packers": 81441,
        "Chiefs": 76416,
        "Bills": 80290,
        "Saints": 76468,
        "Dolphins": 70000,
        "Jaguars": 82000,
        "Panthers": 74867,
        "Redskins": 67617,
        "Ravens": 70745,
        "Buccaneers": 74512,
        "Browns": 54147,
        "Titans": 69143,
        "Bengals": 67260,
        "Broncos": 76125,
        "Steelers": 68400,
        "Lions": 70000,
        "Texans": 75000,
        "Patriots": 71000,
        "Seahawks": 72000,
        "Eagles": 75000,
        "Cardinals": 72200,
        "Colts": 70000,
        "Cowboys": 85000,
        "Jets": 83367,
        "Giants": 83367,
        "49ers": 75000,
        "Vikings": 73000,
        "Falcons": 79330,
        "Raiders": 63132,
        "Rams": 93607,
        "Chargers": 27000,
    }

    stadium_regular_capacities = {
        "Bears": 62500,
        "Packers": 79704,
        "Chiefs": 76416,
        "Bills": 74000,
        "Saints": 73208,
        "Dolphins": 64767,
        "Jaguars": 67814,
        "Panthers": 74867,
        "Redskins": 67617,
        "Ravens": 70745,
        "Buccaneers": 69218,
        "Browns": 50805,
        "Titans": 69143,
        "Bengals": 65515,
        "Broncos": 76125,
        "Steelers": 68400,
        "Lions": 65000,
        "Texans": 72220,
        "Patriots": 65878,
        "Seahawks": 68740,
        "Eagles": 69879,
        "Cardinals": 63400,
        "Colts": 63000,
        "Cowboys": 80000,
        "Jets": 82500,
        "Giants": 82500,
        "49ers": 68500,
        "Vikings": 66655,
        "Falcons": 71000,
        "Raiders": 53200,
        "Rams": 77500,
        "Chargers": 27000,
    }

    cities = {
        "Patriots": "Foxborough,MA",
        "Packers": "Green_Bay,WI",
        "Chiefs": "Kansas_City,MO",
        "Cowboys": "Arlington,TX",
        "Rams": "Inglewood,CA",
        "Buccaneers": "Tampa,FL",
        "49ers": "Santa_Clara,CA",
        "Bears": "Chicago,IL",
        "Dolphins": "Miami_Gardens,FL",
        "Jets": "East_Rutherford,NJ",
        "Giants": "East_Rutherford,NJ",
        "Raiders": "Paradise,NV",
        "Broncos": "Denver,CO",
        "Seahawks": "Seattle,WA",
        "Cardinals": "Glendale,AZ",
        "Eagles": "Philadelphia,PA",
        "Bengals": "Cincinnati,OH",
        "Browns": "Cleveland,OH",
        "Steelers": "Pittsburgh,PA",
        "Ravens": "Baltimore,MD",
        "Falcons": "Atlanta,GA",
        "Saints": "New_Orleans,LA",
        "Panthers": "Charlotte,NC",
        "Vikings": "Minneapolis,MN",
        "Lions": "Detroit,MI",
        "Colts": "Indianapolis,IN",
        "Texans": "Houston,TX",
        "Titans": "Nashville,TN",
        "Jaguars": "Jacksonville,FL",
        "Bills": "Orchard_Park,NY",
        "Commanders": "Landover,MD",
        "Redskins": "Landover,MD",
        "Chargers": "Inglewood,CA",
    }

    sample_year = sample_date.year
    sample_day = days_week[sample_date.weekday()]
    sample_winner = home_team_names[sample_home_team]
    sample_stadium_name = stadium_names[sample_home_team]
    sample_stadium_max_capacity = stadium_max_capacities[sample_home_team]
    sample_stadium_regular_capacity = stadium_regular_capacities[sample_home_team]
    sample_home_city = cities[sample_home_team]

    sample_data = pd.DataFrame(
        [[sample_year, sample_week, sample_winner, sample_day, sample_date, sample_time, sample_pts_win, sample_pts_loss, sample_home_team, sample_away_team, sample_home_team_previous_year_performance, sample_away_team_previous_year_performance, sample_home_team_current_year_performance, sample_away_team_current_year_performance, sample_home_team_current_sos, sample_away_team_current_sos, sample_stadium_name,sample_stadium_max_capacity,sample_stadium_regular_capacity,sample_home_city, sample_weather_condition]],
        columns=["year","week","winner","day","date","time","pts_win","pts_loss","home_team_name","away_team_name","home_team_previous_year_performance","away_team_previous_year_performance","home_team_current_year_performance","away_team_current_year_performance","home_team_current_sos","away_team_current_sos","stadium_name","stadium_max_capacity","stadium_regular_capacity","home_city","weather_condition"])

    results = []

    # Hace 100 iteraciones de la prediccion para calcular el promedio
    for i in range(100):
        # Identificar columnas numéricas y categóricas
        numerical_features = sample_data.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = sample_data.select_dtypes(include=["object"]).columns

        # Volver a calcular sets de entrenamiento y testeo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        if sample_model == "Lasso":
            # Entrenar en el set de entrenamiento
            lasso_model.fit(X_train, y_train)

            # Predecir en el set de testeo
            lasso_sample_data_prediction = lasso_model.predict(sample_data)
            results.append(lasso_sample_data_prediction)
        elif sample_model == "Ridge":
            # Entrenar en el set de entrenamiento
            ridge_model.fit(X_train, y_train)

            # Predecir en el set de testeo
            ridge_sample_data_prediction = ridge_model.predict(sample_data)
            results.append(ridge_sample_data_prediction)
        else:
            # Entrenar en el set de entrenamiento
            linear_reg_model.fit(X_train, y_train)

            # Predecir en el set de testeo
            linear_sample_data_prediction = linear_reg_model.predict(sample_data)
            results.append(linear_sample_data_prediction)

    st.session_state.results = np.array(results)
    st.session_state.mean_result = np.mean(results)
    st.toast("Se ha hecho la prediccion!", icon="🎉")


col1, col2, col3 = st.columns(3)
col1.metric("Precision del modelo lineal", str(np.round(linear_reg_score,4)*100)+"%")
col2.metric("Precision del modelo ridge", str(np.round(ridge_score,4)*100)+"%")
col3.metric("Precision del modelo lasso", str(np.round(lasso_score,4)*100)+"%")

st.header("Ingresa datos para realizar una prediccion")
with st.form("Ingresa datos para realizar una prediccion"):
    model = st.selectbox("Modelo de regresion:", ["Lineal", "Lasso", "Ridge"])
    week = st.slider("Semana", 1, 17)
    home_team = st.text_input("Equipo Local")
    away_team = st.text_input("Equipo Visitante")
    date = st.date_input("Fecha")
    game_time = st.time_input("Horario")
    pts_win = st.slider("Puntos del equipo ganador", 0, 100, 30)
    pts_loss = st.slider("Puntos del equipo perdedor", 0, 100, 20)
    home_team_previous_year_performance = st.number_input("Performance del año anterior del equipo local", -10.0, 10.0, 0.0, 0.1, key=0)
    away_team_previous_year_performance = st.number_input("Performance del año anterior del equipo visitante", -10.0, 10.0, 0.0, 0.1, key=1)
    home_team_current_year_performance = st.number_input("Performance actual del equipo local", -10.0, 10.0, 0.0, 0.1, key=2)
    away_team_current_year_performance = st.number_input("Performance actual del equipo visitante", -10.0, 10.0, 0.0, 0.1, key=3)
    home_team_current_sos = st.number_input("SOS (Strength of schedule) del equipo local", -10.0, 10.0, 0.0, 0.1, key=4)
    away_team_current_sos = st.number_input("SOS (Strength of schedule) del equipo visitante", -10.0, 10.0, 0.0, 0.1, key=5)
    weather_condition = st.radio("Clima", ["cloudy", "clear", "snow", "rain"])

    submitted = st.form_submit_button("Realizar prediccion")

if submitted:
    process_input_data(model, week, home_team, away_team, date, game_time, pts_win, pts_loss, home_team_previous_year_performance, away_team_previous_year_performance, home_team_current_year_performance, away_team_current_year_performance, home_team_current_sos, away_team_current_sos, weather_condition)
    st.metric("Resultado de la prediccion:", str(round(st.session_state.mean_result)) + " personas en asistencia")
    st.dataframe(st.session_state.results)

#  args=(week, home_team, away_team, date, game_time, pts_win, pts_loss, home_team_previous_year_performance, away_team_previous_year_performance, home_team_current_year_performance, away_team_current_year_performance, home_team_current_sos, away_team_current_sos, weather_condition)
