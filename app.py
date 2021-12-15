import streamlit as st
import settings

st.sidebar.title('Machine Learning Code Generator')

st.sidebar.title('Data')
features = st.sidebar.text_input('Feature Columns', )
target = st.sidebar.text_input('Target Column ', )


model = st.sidebar.selectbox('Select your model', ['Linear Regression',
                                                   'Logistic Regression', 'Decision Tree'])

############ Scaling ############
minmax_scaling_status = st.sidebar.checkbox('Min Max Scaling')
standard_scaler_status = st.sidebar.checkbox('Standard Scaler')
visualize_missing_status = st.sidebar.checkbox('Visualize Missing')

if visualize_missing_status:
    visualize_missing_text = {'import': 'import missingno as msno', 'code':
                              'msno.matrix(data)'}
else:
    visualize_missing_text = {'import': '', 'code': ''}


############ NaN Values ############
st.sidebar.header('NaN Values')
nan_value_selection = st.sidebar.radio('NaN Values', ['Drop', 'Fill'])

if nan_value_selection:
    if nan_value_selection == 'Drop':
        nan_values_text = 'data.dropna()'

    elif nan_value_selection == 'Fill':
        fill_selection = st.sidebar.selectbox('Select', ['Mean', 'Median'])
        if fill_selection == 'Mean':
            nan_values_text = f"data.fillna(df.mean())"
        elif fill_selection == 'Median':
            nan_values_text = f"data.fillna(df.median())"

    elif nan_value_selection == 'Fill':
        fill_selection = st.sidebar.selectbox('Select', ['Mean', 'Median'])
        nan_values_text = f"data.fillna(method='{fill_selection.lower()}')"

############ Encoding ############
encoding = st.sidebar.radio(
    'Encoding Type', ['None', 'One Hot Encoding', 'Label Encoding'])

if encoding:
    if encoding == 'Label Encoding':
        encoding_text = """label_encoder = LabelEncoder()
data["column_name"] = label_encoder.fit_transform(data["column_name"])"""

    elif encoding == 'One Hot Encoding':
        encoding_text = """pd.get_dummies()"""

    else:
        encoding_text = ''

train_size = st.sidebar.slider(
    'Train Size Percentage:', min_value=1, max_value=99)

############ Model Evaluation ############

if 'regression' in settings.model_types[model]['type']:
    evaluation_types = st.sidebar.selectbox(
        'Evaluation Type:', ['Mean Absolute Error', 'Mean Squared Error'])

elif 'classification' in settings.model_types[model]['type']:
    evaluation_types = st.sidebar.selectbox(
        'Evaluation Type:', ['Accuracy Score', 'F1 Score', 'Roc Auc Score'])


st.title('🥷 ML Code Generator')


text = f"""

# Importings
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
{visualize_missing_text['import']}
----
{settings.model_types[model]['import']}
from sklearn.model_selection import train_test_split
from sklearn.metrics import {settings.evaluations_dict[evaluation_types]}
----
data = pd.read_csv('./data/path')
{visualize_missing_text['code']}
{nan_values_text}
----
X, y = data.iloc[{features}], data.loc[['{target}']]
----
# Model Pre-processing
{encoding_text}
----
# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={train_size})
----
# Model Fitting
model = {settings.model_types[model]['model_name']}
model.fit(X_train, y_train)
----
# Prediction
preds = model.predict(X_test)
----
# Evaluate
score = {settings.evaluations_dict[evaluation_types]}(preds, y_test)
print(score)
"""
code_file = st.code(text.replace('\n\n', '\n').replace('----', ''))

if st.download_button('🐍 Download.py', text, 'code_data.py'):
    st.success('Codes are saved to your local directory!')
    st.balloons()
