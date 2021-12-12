import streamlit as st

st.sidebar.title('Machine Learning Code Generator')
st.sidebar.header('Model')
st.sidebar.text('Select your model!')
model = st.sidebar.selectbox('Select', ['Linear Regression',
                                        'Logistic Regression', 'Decision Tree'])

st.sidebar.checkbox('Min Max Scaling')
st.sidebar.checkbox('Standard Scaler')

# NaN Values Sidebar
st.sidebar.header('NaN Values')
nan_value_selection = st.sidebar.radio('NaN Values', ['Drop', 'Fill'])

# if nan_value_selection == 'Fill':
#     fill_selection = st.sidebar.selectbox('Select', ['Mean', 'Median'])

if nan_value_selection:
    if nan_value_selection == 'Drop':
        nan_values_text = 'data.dropna()'

    elif nan_value_selection == 'Fill':
        fill_selection = st.sidebar.selectbox('Select', ['Mean', 'Median'])
        nan_values_text = f"data.fillna(method='{fill_selection.lower()}')"


encoding = st.sidebar.radio('Radio', ['One Hot Encoding', 'Label Encoding'])

train_size = st.sidebar.slider(
    'Train Size Percentage:', min_value=1, max_value=99)

model_types = {'Linear Regression':
               {'model_name': 'LinearRegression()',
                'import': 'from sklearn.linear_model import LinearRegression'},
               'Logistic Regression':
               {'model_name': 'LogisticRegression()',
                'import': 'from sklearn.linear_model import LogisticRegression'},
               'Decision Tree':
               {'model_name': 'DecisionTreeClassifier()',
                'import': 'from sklearn.tree import DecisionTreeClassifier'}}

if encoding:
    if encoding == 'Label Encoding':
        encoding_text = """label_encoder = LabelEncoder()
data["column_name"] = label_encoder.fit_transform(data["column_name"])"""

    elif encoding == 'One Hot Encoding':
        encoding_text = """pd.get_dummies()"""
    else:
        encoding_text = ''


st.code(f"""


==================
Please provide data 

data = pd.read_csv()
==================

## Importings 
{model_types[model]['import']}

## Model Pre-processing
{nan_values_text}
{encoding_text}

## Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={train_size})

## Model Fitting

model = {model_types[model]['model_name']}
model.fit(X_train, y_train)

## Evaluate

""")
