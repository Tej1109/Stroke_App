import streamlit as st

st.set_page_config(page_title="Reduce Stroke Risk", page_icon="💡", layout="centered")

st.title("💡 Evidence-Based Ways to Reduce Stroke Risk")
st.write("Here are medically recommended lifestyle improvements.")

st.header("1️⃣ Healthy Diet")
st.image("images/healthy_food.jpg", use_column_width=True)
st.write("- Eat fruits, vegetables, nuts\n- Choose whole grains\n- Reduce salt intake")

st.header("2️⃣ Exercise Regularly")
st.image("images/exercise.jpg", use_column_width=True)
st.write("- 30 minutes of activity daily\n- Walking, jogging, swimming")

st.header("3️⃣ Control Blood Pressure")
st.image("images/bp_check.jpg", use_column_width=True)
st.write("- Check BP weekly\n- Reduce sodium\n- Follow doctor’s treatment")

st.header("4️⃣ Quit Smoking")
st.image("images/quit_smoking.jpg", use_column_width=True)
st.write("- Smoking increases stroke risk by 2–4×\n- Quitting rapidly reduces risk")

st.markdown("---")
st.page_link("app.py", label="⬅ Back to Prediction")
