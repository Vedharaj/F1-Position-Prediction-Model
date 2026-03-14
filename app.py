import difflib
from io import StringIO
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# import os
# import zipfile

# os.system("kaggle datasets download -d jtrotman/formula-1-race-data")


# with zipfile.ZipFile("formula-1-race-data.zip", 'r') as zip_ref:
#     zip_ref.extractall("dataset")

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)


def read_csv_with_fallback(path):
    """Read CSV robustly across Windows/default encoding differences."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("f1_position_model.pkl")

# -----------------------------
# Load datasets
# -----------------------------
results_df = read_csv_with_fallback("dataset/results.csv")
circuits_df = read_csv_with_fallback("dataset/f1_circuits_data.csv")

# clean numeric
results_df["position"] = results_df["positionOrder"]
results_df["points"] = pd.to_numeric(results_df["points"], errors="coerce").fillna(0)

results_df = results_df.sort_values("raceId")

# -----------------------
# Dictionaries (name → id)
# -----------------------

drivers = {}
constructors = {}
circuits = {}

drivers_dict_df = read_csv_with_fallback("dataset/drivers.csv")
for _, row in drivers_dict_df.iterrows():
    drivers[f"{row['forename']} {row['surname']}"] = int(row["driverId"])

constructors_dict_df = read_csv_with_fallback("dataset/constructors.csv")
for _, row in constructors_dict_df.iterrows():
    constructors[row["constructorRef"]] = int(row["constructorId"])

circuits_dict_df = read_csv_with_fallback("dataset/circuits.csv")
for _, row in circuits_dict_df.iterrows():
    circuits[f"{row['name']}, {row['location']}, {row['country']}"] = int(row["circuitId"])

# ------------------------------------------------
# Helper: window stats
# ------------------------------------------------

def calc_stats(df, window):

    last = df.tail(window)

    pos = last["position"].mean() if not last.empty else 10
    pts = last["points"].mean() if not last.empty else 0

    return pos, pts


# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

def get_stats(driver_id, constructor_id, race_id):

    driver_hist = results_df[
        (results_df.driverId == driver_id) &
        (results_df.raceId < race_id)
    ]

    constructor_hist = results_df[
        (results_df.constructorId == constructor_id) &
        (results_df.raceId < race_id)
    ]

    # driver windows
    d3_pos, d3_pts = calc_stats(driver_hist, 3)
    d5_pos, d5_pts = calc_stats(driver_hist, 5)
    d10_pos, d10_pts = calc_stats(driver_hist, 10)

    d_all_pos = driver_hist["position"].mean() if not driver_hist.empty else 10
    d_all_pts = driver_hist["points"].mean() if not driver_hist.empty else 0

    # constructor windows
    c3_pos, c3_pts = calc_stats(constructor_hist, 3)
    c5_pos, c5_pts = calc_stats(constructor_hist, 5)
    c10_pos, c10_pts = calc_stats(constructor_hist, 10)

    c_all_pos = constructor_hist["position"].mean() if not constructor_hist.empty else 10
    c_all_pts = constructor_hist["points"].mean() if not constructor_hist.empty else 0

    return [
        d3_pos,
        d5_pos,
        d10_pos,
        d_all_pos,

        d3_pts,
        d5_pts,
        d10_pts,
        d_all_pts,

        c3_pos,
        c5_pos,
        c10_pos,
        c_all_pos,

        c3_pts,
        c5_pts,
        c10_pts,
        c_all_pts
    ]


# ------------------------------------------------
# Fuzzy match helper
# ------------------------------------------------

def find_best_match(text, dictionary):

    if text in dictionary:
        return dictionary[text]

    matches = difflib.get_close_matches(text, dictionary.keys(), n=1, cutoff=0.6)

    if matches:
        return dictionary[matches[0]]

    return 0


# ------------------------------------------------
# UI
# ------------------------------------------------

st.title("🏎️ F1 Race Position Predictor")

tab1, tab2 = st.tabs([
    "Single Prediction",
    "Bulk Prediction"
])


# =================================================
# TAB 1
# =================================================

with tab1:

    grid = st.slider("Grid Position", 1, 20)

    race_id = st.number_input(
        "Race ID",
        min_value=1,
        value=1100
    )

    driver_name = st.selectbox(
        "Driver",
        sorted(drivers.keys())
    )

    constructor_name = st.selectbox(
        "Constructor",
        sorted(constructors.keys())
    )

    circuit_name = st.selectbox(
        "Circuit",
        sorted(circuits.keys())
    )

    year = st.selectbox(
        "Year",
        list(range(2000, 2027))
    )

    driver_id = drivers[driver_name]
    constructor_id = constructors[constructor_name]
    circuit_id = circuits[circuit_name]

    # get circuit features
    circuit_row = circuits_df[circuits_df["circuitId"] == circuit_id]

    laps = circuit_row["Laps"].values[0]
    corners = circuit_row["Corners"].values[0]

    if st.button("Predict Result"):

        stats = get_stats(
            driver_id,
            constructor_id,
            race_id
        )

        features = np.array([[
            grid,
            driver_id,
            constructor_id,
            circuit_id,
            year,
            laps,
            corners,
            *stats
        ]])

        prediction = model.predict(features)

        st.success(f"🏁 Predicted Position: {int(prediction[0])}")


# =================================================
# TAB 2
# =================================================

with tab2:

    csv_text = st.text_area("Paste CSV")

    race_id_bulk = st.number_input(
        "Race ID",
        min_value=1,
        value=1100,
        key="bulk_race"
    )

    circuit_name_bulk = st.selectbox(
        "Circuit",
        sorted(circuits.keys()),
        key="bulk_circuit"
    )

    year_bulk = st.selectbox(
        "Year",
        list(range(2000, 2027)),
        key="bulk_year"
    )

    circuit_id_bulk = circuits[circuit_name_bulk]

    circuit_row = circuits_df[circuits_df["circuitId"] == circuit_id_bulk]

    laps_bulk = circuit_row["Laps"].values[0]
    corners_bulk = circuit_row["Corners"].values[0]

    if st.button("Predict From CSV"):

        if csv_text.strip():

            df = pd.read_csv(StringIO(csv_text))

            predictions = []

            for _, row in df.iterrows():

                grid_val = int(row["Grid"])
                driver_name = row["Driver"]
                team_name = row["Team"]

                driver_id = find_best_match(driver_name, drivers)
                constructor_id = find_best_match(team_name, constructors)

                stats = get_stats(
                    driver_id,
                    constructor_id,
                    race_id_bulk
                )

                features = np.array([[
                    grid_val,
                    driver_id,
                    constructor_id,
                    circuit_id_bulk,
                    year_bulk,
                    laps_bulk,
                    corners_bulk,
                    *stats
                ]])

                pred = model.predict(features)[0]

                predictions.append(pred)

            df["Predicted_Position"] = predictions

            df = df.sort_values("Predicted_Position")
            df["Predicted_Position"] = range(1, len(df)+1)

            st.dataframe(df)