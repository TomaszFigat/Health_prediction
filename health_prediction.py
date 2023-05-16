import pandas as pd
from pathlib import Path
import pathlib
import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename, 'rb'))


def main():
    st.set_page_config(page_title="Health prediction")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://st4.depositphotos.com/1325771/39154/i/600/depositphotos_391545206-stock-photo-happy-male-medical-doctor-portrait.jpg")

    df = pd.read_csv('DSP_13.csv', sep=';')
    df = df.fillna(df.mean())

    max_symptoms = df['objawy'].max()
    max_age = df['wiek'].max()
    max_comorbidities = df['choroby_wsp'].max()
    max_height = df['wzrost'].max()
    max_meds = df['leki'].max()

    with overview:
        st.title("Health prediction")

    with left:
        age = st.slider("Wiek", value=1, min_value=1, max_value=int(max_age))
        height = st.slider("Wzrost", value=180, min_value=150,
                           max_value=int(max_height))

    with right:
        syptoms = st.slider("Objawy", value=0, min_value=0,
                            max_value=int(max_symptoms))
        meds = st.slider("Leki", value=0, min_value=0, max_value=int(max_meds))
        comorbidities = st.slider(
            "Choroby współistniejące",
            value=1,
            min_value=0, max_value=int(max_comorbidities))

    data = [[syptoms, age, comorbidities, height, meds]]
    health = model.predict(data)

    print(f'health: {health[0]}')
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy osoba jest zdrowa?")
        st.subheader(("Tak" if health[0] == 0 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(
            s_confidence[0][health][0] * 100))


if __name__ == "__main__":
    main()