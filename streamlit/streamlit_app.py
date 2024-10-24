import streamlit as st
import pandas as pd
import numpy as np

pg = st.navigation([st.Page("data_exploration.py", title="Explorar los datos"), st.Page("predictions.py", title="Hacer predicciones")])
pg.run()
