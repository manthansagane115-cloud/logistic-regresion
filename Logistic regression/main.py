import streamlit as st
import pandas as pd
import pickle
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'titanic_model.pkl')

# Load the trained model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: 'titanic_model.pkl' not found at {model_path}. Please ensure the model file is in the same directory as the app.")
    st.stop()

# Streamlit App Title
st.title('Titanic Survival Predictor')
st.write('Enter the passenger details to predict survival.')

# Input fields for user data
pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])
age = st.sidebar.slider('Age', 0.42, 80.0, 25.0)
sibsp = st.sidebar.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
parch = st.sidebar.slider('Number of Parents/Children Aboard', 0, 6, 0)
fare = st.sidebar.slider('Fare', 0.0, 512.3292, 30.0)
embarked = st.sidebar.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Preprocess input data to match model training format
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a DataFrame from input
    data = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    }
    input_df = pd.DataFrame(data)

    # Encode 'Sex' (male: 1, female: 0, as per training notebook)
    # Re-initialize LabelEncoder for consistency or use a predefined mapping
    input_df['Sex'] = input_df['Sex'].map({'male': 1, 'female': 0})
    
    # One-hot encode 'Embarked' (drop_first=True, as per training notebook)
    # Ensure all possible columns (Embarked_Q, Embarked_S) are present, even if 0
    input_df = pd.get_dummies(input_df, columns=['Embarked'], drop_first=True)

    # Ensure all columns expected by the model are present and in the correct order
    # Based on `coef_df` from `0vvC99lDFy0e`
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    return input_df

# Make prediction when button is clicked
if st.sidebar.button('Predict Survival'):
    processed_input = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)
    
    try:
        prediction_proba = model.predict_proba(processed_input)[:, 1][0]
        prediction = (prediction_proba > 0.5).astype(int)

        st.subheader('Prediction Result:')
        if prediction == 1:
            st.success(f"The passenger is likely to survive with a probability of {prediction_proba:.2f}")
        else:
            st.error(f"The passenger is likely to not survive with a probability of {1 - prediction_proba:.2f}")
        
        st.write(f"Survival Probability: {prediction_proba:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.write("\n---\n")
st.write("**How to run this application:**")
st.write("1. Ensure both `main.py` and `titanic_model.pkl` are in the same directory.")
st.write("2. Open a terminal or command prompt.")
st.write("3. Navigate to the directory where you saved the files.")
st.write("4. Install required packages: `pip install streamlit pandas scikit-learn`")
st.write("5. Run the command: `streamlit run main.py`")
st.write("6. Your browser will automatically open the Streamlit application.")
