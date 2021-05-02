import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import numpy as np
from random import sample

# Create a title and sub-title
st.write("""
# Stroke Prediction
Detect if someone has stroke using machine learning
""")

# Open and display an image
image = Image.open('brain.png')
st.image(image, caption='Love Machine Learning', use_column_width=True)

# columns -
# gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status
# 0: never smoked, 1: formerly smoked, 2: smokes, 3: unknown
# Male: 0, Female: 1
# work_type- 0: never worked, 1: government, 2: self-employed, 3: private
# Residence_type- 0: urban, 1: rural
# ever-married - 0: No, 1: Yes


# Get the data
origin_df = pd.read_csv('stroke.csv')
df = pd.read_csv('export_stroke.csv')
st.subheader('Data Information')

# show the data as a table
st.dataframe(origin_df)

# show statistics of the data
st.write(origin_df.describe())

# show the data as a chart
# chart = st.bar_chart(origin_df)

# # Split the data into independent 'X' and dependent 'Y'
# X = df.iloc[:, 1:11].values
# Y = df.iloc[:, -1].values
#
# # Split the data set into 75% training and 25 % testing
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


def create_splits(df):
    train_df, valid_df = train_test_split(df, test_size=0.2, stratify=df['stroke'], random_state=0)
    s_inds = train_df[train_df['stroke'] == 1].index.tolist()
    ns_inds = train_df[train_df['stroke'] == 0].index.tolist()

    ns_sample = sample(ns_inds, len(s_inds))
    train_data = train_df.loc[s_inds + ns_sample]

    vs_inds = valid_df[valid_df['stroke'] == 1].index.tolist()
    vns_inds = valid_df[valid_df['stroke'] == 0].index.tolist()
    vns_sample = sample(vns_inds, len(vs_inds) * 4)
    val_data = valid_df.loc[vs_inds + vns_sample]
    x_train = train_data.iloc[:, 1:11].values
    y_train = train_data.iloc[:, -1].values

    x_test = val_data.iloc[:, 1:11].values
    y_test = val_data.iloc[:, -1].values

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = create_splits(df)


def get_user_input():
    gender = st.sidebar.radio('gender', ['male', 'female'])
    age = st.sidebar.slider('age', 17, 100, 50)
    st.sidebar.text("1: Yes, 0: No")
    hypertension = st.sidebar.radio('hypertension', [1, 0])
    heart_disease = st.sidebar.radio('heart disease', [1, 0])
    ever_married = st.sidebar.radio('ever married', [1, 0])
    work_type = st.sidebar.radio('work type', ['never worked', 'government job', 'self-employed', 'private'])
    residence_type = st.sidebar.radio('Residence type', ['urban', 'rural'])
    avg_glucose_level = st.sidebar.slider('average glucose level', 50, 300, 100)
    bmi = st.sidebar.slider('bmi', 11, 98, 30)
    smoking_status = st.sidebar.radio('smoking status', ['never smoked', 'formerly smoked', 'smokes', 'unknown'])

    if gender == 'male':
        gender = 0
    else:
        gender = 1

    if work_type == 'never worked':
        work_type = 0
    elif work_type == 'government job':
        work_type = 1
    elif work_type == 'self-employed':
        work_type = 2
    else:
        work_type = 3

    if residence_type == 'urban':
        residence_type = 0
    else:
        residence_type = 1

    if smoking_status == 'never smoked':
        smoking_status = 0
    elif smoking_status == 'formerly smoked':
        smoking_status = 1
    elif smoking_status == 'smokes':
        smoking_status = 2
    else:
        smoking_status = 3

    user_data = {'gender': gender,
                 'age': age,
                 'hypertension': hypertension,
                 'heart_disease': heart_disease,
                 'ever_married': ever_married,
                 'work_type': work_type,
                 'Residence_type': residence_type,
                 'avg_glucose_level': avg_glucose_level,
                 'bmi': bmi,
                 'smoking_status': smoking_status}

    features = pd.DataFrame(user_data, index=[0])
    return features


# store user input into a variable
user_input = get_user_input()
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

# show the models metrics
st.subheader('Models Accuracy')
accuracy_score = accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100
st.write(str(accuracy_score) + '%')

# prediction based on user input
prediction = RandomForestClassifier.predict(user_input)

if prediction == 0:
    result = "You are not likely to have stroke"
else:
    result = "You are likely to have stroke"

# Display the classification result
st.subheader('Prediction Result:')
st.write(result)
