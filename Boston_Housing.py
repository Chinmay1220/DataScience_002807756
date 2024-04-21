import streamlit as st
import pandas as pd


from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor


# Streamlit app title and description
st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

# Load Boston dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
st.sidebar.header('Specify Input Parameters')

# Function to get user input
def user_input_features():
    input_features = {}
    for feature_name in X.columns:
        input_features[feature_name] = st.sidebar.slider(feature_name, X[feature_name].min(), X[feature_name].max(), X[feature_name].mean())
    return pd.DataFrame(input_features, index=[0])

# Get user input
df = user_input_features()

# Display specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Build model
model = RandomForestRegressor()
model.fit(X, Y)

# Make prediction
prediction = model.predict(df)

# Display prediction
st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explain model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Display feature importance
st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
