model_types = {'Linear Regression':
               {'model_name': 'LinearRegression()',
                'import': 'from sklearn.linear_model import LinearRegression',
                'type': ['regression']},
               'Logistic Regression':
               {'model_name': 'LogisticRegression()',
                'import': 'from sklearn.linear_model import LogisticRegression',
                'type': ['classification']},
               'Decision Tree':
               {'model_name': 'DecisionTreeClassifier()',
                'import': 'from sklearn.tree import DecisionTreeClassifier',
                'type': ['classification']}}

evaluations_dict = {'Mean Absolute Error': 'mean_absolute_error',
                    'Mean Squared Error': 'mean_squared_error',
                    'Roc Auc Score': 'roc_auc_score',
                    'F1 Score': 'f1_score',
                    'Accuracy Score': 'accuracy_score'}

