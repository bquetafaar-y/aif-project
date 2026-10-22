
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split


MODEL_FILENAME = "random_forest_model.joblib"



def load_and_predict(X, filename=MODEL_FILENAME):
    model = load(filename)

    prediction = model.predict(X)
    probabilities = model.predict_proba(X)

    class_index = list(model.classes_).index(1)
    probability_lnu = float(probabilities[0][class_index])

    return prediction, probability_lnu

def create_streamlit_app():
    st.title("Прогноз вибору університету")
    st.subheader("Чи обере абітурієнт, ЛНУ імені Івана Франка?")

    st.write(
        "Введи характеристики абітурієнта, "
        "щоб отримати прогноз моделі."
    )

    nmt_average_score = st.slider(
        "Середній бал НМТ",
        min_value=100.0,
        max_value=200.0,
        value=160.0,
        step=1.0,
    )

    intended_specialty = st.selectbox(
        "Бажана спеціальність",
        [
            "International Relations",
            "Economics",
            "Law",
            "Computer Science",
            "Philology",
            "Management",
            "Psychology",
            "Journalism",
        ],
    )

    funding_preference = st.selectbox(
        "Бажана форма фінансування",
        ["Budget", "Contract", "Either"],
    )

    distance_to_lviv_km = st.number_input(
        "Відстань до Львова (км)",
        min_value=1.0,
        max_value=800.0,
        value=50.0,
        step=1.0,
    )

    tuition_affordability = st.selectbox(
        "Доступність оплати навчання",
        ["High", "Medium", "Low"],
    )

    reputation_importance_1_5 = st.slider(
        "Важливість репутації університету (1–5)",
        min_value=1,
        max_value=5,
        value=4,
    )

    lviv_study_preference = st.selectbox(
        "Бажання навчатися у Львові",
        ["High", "Medium", "Low"],
    )

    if st.button("Спрогнозувати вибір"):
        applicant = pd.DataFrame(
            [{
                "nmt_average_score": nmt_average_score,
                "intended_specialty": intended_specialty,
                "funding_preference": funding_preference,
                "distance_to_lviv_km": distance_to_lviv_km,
                "tuition_affordability": tuition_affordability,
                "reputation_importance_1_5": reputation_importance_1_5,
                "lviv_study_preference": lviv_study_preference,
            }]
        )

        # Узгоджуємо dummy-колонки з тими, на яких навчалася модель
        X_reference = load("X.joblib")
        feature_columns = X_reference.columns

        applicant = pd.get_dummies(applicant)
        applicant = applicant.reindex(
            columns=feature_columns,
            fill_value=0,
        )

        prediction, probability_lnu = load_and_predict(applicant)

        st.divider()
        st.subheader("Результат прогнозу")

        if int(prediction[0]) == 1:
            st.success("Модель прогнозує, що абітурієнт обере ЛНУ.")
        else:
            st.info("Модель прогнозує, що абітурієнт не обере ЛНУ.")

        st.metric(
            "Оцінена ймовірність вибору ЛНУ",
            f"{probability_lnu:.1%}",
        )

        st.progress(probability_lnu)

        st.caption(
            "Модель навчена на синтетичних даних, тому її прогноз "
            "не є достовірною оцінкою реальної поведінки абітурієнтів."
        )

if __name__ == "__main__":
    create_streamlit_app()