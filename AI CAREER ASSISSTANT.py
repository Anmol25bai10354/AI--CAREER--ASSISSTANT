import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# -------------------------------
# PAGE TITLE
# -------------------------------
st.title(" AI Digital Career Assistant")

st.write("Answer a few questions to get your ideal career recommendation!")

# -------------------------------
# DATASET
# -------------------------------
data = {
    'Python':        [1, 0, 0, 1, 1, 0, 0, 1],
    'Communication': [1, 1, 0, 0, 1, 1, 0, 1],
    'Design':        [0, 1, 1, 0, 0, 1, 1, 1],
    'Math':          [1, 0, 1, 1, 1, 0, 0, 1],
    'Career': [
        'Data Analyst',
        'HR Manager',
        'UI/UX Designer',
        'ML Engineer',
        'Data Scientist',
        'Marketing Manager',
        'Graphic Designer',
        'AI Engineer'
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# MODEL TRAINING
# -------------------------------
X = df[['Python', 'Communication', 'Design', 'Math']]
y = df['Career']

model = DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(X, y)

# -------------------------------
# USER INPUT (UI)
# -------------------------------
st.subheader("📝 Your Skills")

python = st.selectbox("Do you know Python?", [0, 1])
communication = st.selectbox("Good communication skills?", [0, 1])
design = st.selectbox("Interested in design?", [0, 1])
math = st.selectbox("Good at math?", [0, 1])

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("🎯 Get Career Recommendation"):
    
    user_data = pd.DataFrame([[python, communication, design, math]],
    columns=['Python','Communication','Design','Math'])

    prediction = model.predict(user_data)

    st.success(f"🎯 Recommended Career: {prediction[0]}")

    # -------------------------------
    # CAREER ADVICE
    # -------------------------------
    career_advice = {
        'Data Analyst': "Learn Excel, SQL, Data Visualization",
        'ML Engineer': "Learn ML algorithms, Python, Deep Learning",
        'Data Scientist': "Statistics, Python, Machine Learning",
        'AI Engineer': "AI models, Python, Neural Networks",
        'UI/UX Designer': "Figma, Design Principles",
        'Graphic Designer': "Photoshop, Creativity",
        'HR Manager': "Communication, Management skills",
        'Marketing Manager': "Digital Marketing, SEO"
    }

    st.info(f"📚 Skills to learn: {career_advice.get(prediction[0])}")

    # -------------------------------
    # GRAPH
    # -------------------------------
    skills = ['Python', 'Communication', 'Design', 'Math']
    values = [python, communication, design, math]

    fig, ax = plt.subplots(1, 2, figsize=(10,4))

    ax[0].bar(skills, values)
    ax[0].set_title("Bar Chart")

    ax[1].pie(values, labels=skills, autopct='%1.1f%%', startangle=90)
    ax[1].set_title("Pie Chart")

    st.pyplot(fig)
