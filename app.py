import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.title('Employee Salary Prediction App')

uploaded_file = st.file_uploader("Upload your Employee Data CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Label Encoding
    le_gender = LabelEncoder()
    le_edu = LabelEncoder()
    le_job = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Education Level'] = le_edu.fit_transform(df['Education Level'])
    df['Job Title'] = le_job.fit_transform(df['Job Title'])

    X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
    y = df['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.success('Model trained successfully!')

    st.sidebar.header('Enter Employee Details for Prediction')
    age = st.sidebar.slider('Age', int(df['Age'].min()), int(df['Age'].max()))
    gender = st.sidebar.selectbox('Gender', le_gender.classes_)
    education = st.sidebar.selectbox('Education Level', le_edu.classes_)
    job = st.sidebar.selectbox('Job Title', le_job.classes_)
    experience = st.sidebar.slider('Years of Experience', int(df['Years of Experience'].min()), int(df['Years of Experience'].max()))

    if st.button('Predict Salary'):
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [le_gender.transform([gender])[0]],
            'Education Level': [le_edu.transform([education])[0]],
            'Job Title': [le_job.transform([job])[0]],
            'Years of Experience': [experience]
        })

        prediction = model.predict(input_data)
        st.success(f'Predicted Salary: ${prediction[0]:,.2f}')
else:
    st.info('Please upload a CSV file to proceed.')
