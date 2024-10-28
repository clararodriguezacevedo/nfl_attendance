Para ejecutar los archivos:

Forkear el repo y clonar localmente con Git.
Una vez que se tienen los archivos locales, abrirlos en su IDE de preferencia.

En la carpeta csv_files/originals_from_kaggle se encuentran las tablas originales, antes de ser modificadas.
En la carpeta csv_files/engineered se encuentran las tablas modificadas.

- Los JuPyter Notebooks data_wrangling y weather modifican las tablas originales para organizarlas, agregar y eliminar variables, etc. La tabla final está almacenada en csv_files/engineered/nfl_games_attendance.csv
- El JuPyter Notebook correlation genera una matriz de correlacion a partir de la tabla de csv_files/engineered/nfl_games_attendance.csv
- El JuPyter Notebook plotting genera gráficos a partir de la tabla de csv_files/engineered/nfl_games_attendance.csv
- El JuPyter Notebooks regression contiene los modelos de regresion a partir de la tabla de csv_files/engineered/nfl_games_attendance.csv. Procesa los datos, genera las predicciones, y analiza los resultados.
Este archivo tambien contiene una celda que permite ingresar datos de un partido y realizar una prediccion con un modelo a eleccion
Los datos recopilados por la ejecución de los modelos se almacenan en la tabla csv_files/engineered/efficiency_registry.csv

La interfaz de Streamlit (que corresponde a la carpeta /streaamlit) esta hosteada en el link https://nflattendance-metodologia.streamlit.app/ 
