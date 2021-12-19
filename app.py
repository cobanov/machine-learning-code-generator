import streamlit as st
import settings

st.sidebar.title("Machine Learning Code Generator")

st.sidebar.title("Model")

model = st.sidebar.selectbox(
    "Select your model", ["Linear Regression", "Logistic Regression", "Decision Tree"]
)

st.sidebar.title("Pre-Processing")

############ NaN Values ############
nan_value_selection = st.sidebar.radio("NaN Values", ["Drop", "Fill"])

if nan_value_selection:
    if nan_value_selection == "Drop":
        nan_values_text = "data.dropna(inplace=True)"

    elif nan_value_selection == "Fill":
        fill_selection = st.sidebar.selectbox("Select", ["Mean", "Median"])
        if fill_selection == "Mean":
            nan_values_text = f"data.fillna(df.mean(), inplace=True)"
        elif fill_selection == "Median":
            nan_values_text = f"data.fillna(df.median(), inplace=True)"

    elif nan_value_selection == "Fill":
        fill_selection = st.sidebar.selectbox("Select", ["Mean", "Median"])
        nan_values_text = f"data.fillna(method='{fill_selection.lower()}')"
####################################

############ Encoding ############
encoding = st.sidebar.radio(
    "Encoding Type", ["None", "One Hot Encoding", "Label Encoding"]
)

if encoding:
    if encoding == "Label Encoding":
        encode_columns = st.sidebar.text_input(
            "Columns to Encode", placeholder="comma, seperated, column, names"
        )
        encoding_text = f"""label_encoder = LabelEncoder()
data[['{encode_columns}']] = data[['{encode_columns}']].apply(label_encoder.fit_transform )"""

    elif encoding == "One Hot Encoding":
        encoding_text = """data = pd.get_dummies(data)"""

    else:
        encoding_text = ""


############ Column Names ############
st.sidebar.title("Dataset")
features = st.sidebar.text_input("Feature Columns", placeholder="1:10")
target = st.sidebar.text_input("Target Column ", placeholder="target_name")
######################################


############ Test Size  ############
test_size = st.sidebar.slider(
    "Test Size Percentage:", min_value=0.05, max_value=0.95, step=0.05
)
######################################

############ Scaling ############
minmax_scaling_status = st.sidebar.checkbox("Min Max Scaling")
standard_scaler_status = st.sidebar.checkbox("Standard Scaler")
visualize_missing_status = st.sidebar.checkbox("Visualize Missing")

if visualize_missing_status:
    visualize_missing_text = {
        "import": "import missingno as msno",
        "code": "msno.matrix(data)",
    }
else:
    visualize_missing_text = {"import": "", "code": ""}
####################################


st.sidebar.title("Evaluation")


############ Model Evaluation ############

if "regression" in settings.model_types[model]["type"]:
    evaluation_types = st.sidebar.selectbox(
        "Evaluation Type:", ["Mean Absolute Error", "Mean Squared Error"]
    )

elif "classification" in settings.model_types[model]["type"]:
    evaluation_types = st.sidebar.selectbox(
        "Evaluation Type:", ["Accuracy Score", "F1 Score", "Roc Auc Score"]
    )

confusion_matrix_status = st.sidebar.checkbox("Confusion Matrix")

if confusion_matrix_status:
    confusion_matrix_import = "from sklearn.metrics import confusion_matrix"
    confusion_matrix_text = """sns.heatmap(confusion_matrix(y_test, preds), annot=True, square=True)
plt.show()"""

else:
    confusion_matrix_text = ""
    confusion_matrix_import = ""

st.title("🥷 ML Code Generator")


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
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
----
data = pd.read_csv('./data/path')
{visualize_missing_text['code']}
{nan_values_text}
----
# Model Pre-processing
{encoding_text}
----
X, y = data.iloc[:, {features}], data.loc[:, ['{target}']]
print('X Shape: ', X.shape)
print('Y Shape: ', y.shape)
----
# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size})
----
# Model Fitting
model = {settings.model_types[model]['model_name']}
model.fit(X_train, y_train)
----
# Prediction
preds = model.predict(X_test)
----
# Evaluate
score = {settings.evaluations_dict[evaluation_types]}(y_test, preds)
print('Score:', score)
----
{confusion_matrix_text}
"""
text_beautified = text.replace("\n\n", "\n").replace("----", "")
code_file = st.code(text_beautified)

if st.download_button("🐍 Download.py", text_beautified, "code_template.py"):
    st.success("Codes are saved to your local directory!")
    st.balloons()
