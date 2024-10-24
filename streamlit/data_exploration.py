import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("./csv_files/engineered/nfl_games_attendance.csv")

tab1, tab2 = st.tabs(["Explorar la base de datos", "Graficos"])



tab1.write("Base de datos")
tab1.dataframe(df, column_order=["game_attendance", "stadium_name","stadium_max_capacity","stadium_regular_capacity","home_team_name","away_team_name", "home_city","weather_condition", "year","week","winner","day","date","time","pts_win","pts_loss","home_team_previous_year_performance","away_team_previous_year_performance","home_team_current_year_performance","away_team_current_year_performance","home_team_current_sos","away_team_current_sos"])



tab2.header("Asistencia a los partidos para cada estadio")
tab2.scatter_chart(df, y="game_attendance", x="stadium_name", x_label="Nombre de los estadios", y_label="Asistencia por partido")

tab2.header("Asistencia a los partidos por condicion clilmatica")
fig1 = px.box(df, y="game_attendance", x="weather_condition")
tab2.plotly_chart(fig1, x_label="Condicion climatica", y_label="Asistencia por partido")

tab2.header("Asistencia a los partidos por dia de la semana")
fig1 = px.box(df, y="game_attendance", x="day")
tab2.plotly_chart(fig1, x_label="Dia de la semana", y_label="Asistencia por partido")

tab2.header("Asistencia a los partidos por equipo local y equipo visitante")
fig2 = px.density_heatmap(df, x="home_team_name", y="away_team_name", z="game_attendance", histfunc="avg", width=1200, height=700)
fig2.update_layout(xaxis_title="Equipo local",
                  yaxis_title="Equipo visitante",)
fig2.update_xaxes(tickangle=90,tickmode='linear')
fig2.update_layout(xaxis= dict(type='category', categoryorder = "category ascending"))
fig2.update_layout(yaxis= dict(type='category', categoryorder = "category ascending"))
fig2.update_layout(legend_title_text="Asistencia promedio a los partidos")
tab2.plotly_chart(fig2, use_container_width=False)
