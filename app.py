import streamlit as st
import numpy as np
import pickle
import random

def hard_voting(models, input_data):
    # Collect predictions from each model
    predictions = [model.predict(input_data).flatten() for model in models]
    # Perform a majority vote
    vote_counts = np.sum(predictions, axis=0)
    # Majority vote for binary classification (0 or 1)
    majority_vote = 1 if vote_counts[0] > len(models) / 2 else 0
    return majority_vote

# Load your trained model
model1 = pickle.load(open('RandomForest.pkl', 'rb'))
model2 = pickle.load(open('decisiontree.pkl', 'rb'))
model3 = pickle.load(open('GaussianNb.pkl', 'rb'))
model4 = pickle.load(open('knn.pkl', 'rb'))
model5 = pickle.load(open('XGBoost.pkl', 'rb'))
if 'gender' not in st.session_state:
    st.session_state['gender'] = "Male"
if 'hemoglobin' not in st.session_state:
    st.session_state['hemoglobin'] = ""
if 'mch' not in st.session_state:
    st.session_state['mch'] = ""
if 'mchc' not in st.session_state:
    st.session_state['mchc'] = ""
if 'mcv' not in st.session_state:
    st.session_state['mcv'] = ""
# Custom CSS for styling
st.markdown(
    """
     <style>
    .big-font {
        font-size:40px !important;
        color: #4CAF50;
        font-weight: bold;
    }
    .app-header {
        text-align: center;
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .prediction-result-positive {
        font-size: 30px;
        color: #dc3545; /* Red for positive/anemia */
        font-weight: bold;
        margin-top: 10px;
    }
    .prediction-result-negative {
        font-size: 30px;
        color: #28a745; /* Green for negative/no anemia */
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# App Header
st.markdown('<div class="app-header"><h1 class="big-font">Anemia Prediction App</h1></div>', unsafe_allow_html=True)

def generate_random_input():
    st.session_state['gender'] = random.choice(["Male", "Female"])
    st.session_state['hemoglobin'] = str(round(random.uniform(5.0, 20.5), 1)) # got from df.describe()
    st.session_state['mch'] = str(round(random.uniform(15.0, 33.0), 1))        # got from df.describe()
    st.session_state['mchc'] = str(round(random.uniform(27.0, 36.0), 1))       # got from df.describe()
    st.session_state['mcv'] = str(round(random.uniform(65.0, 105.0), 1))       # got from df.describe()
# User inputs
st.markdown("""<br>""", unsafe_allow_html=True)
if st.button("Generate Random Input"):
        generate_random_input()
with st.form("user_input_form"):
    # gender = st.selectbox("Select Gender", ("Male", "Female"), index=("Male", "Female").index(st.session_state['gender']))
    gender = st.radio("Select Gender", ["Male", "Female"],index=("Male", "Female").index(st.session_state['gender']), horizontal=True)
    hemoglobin = st.text_input("Enter Hemoglobin level (g/dL)", value=st.session_state['hemoglobin'])
    mch = st.text_input("Enter MCH (pg)", value=st.session_state['mch'])
    mchc = st.text_input("Enter MCHC (g/dL)", value=st.session_state['mchc'])
    mcv = st.text_input("Enter MCV (fL)", value=st.session_state['mcv'])
    submitted = st.form_submit_button("Predict")
    en_result= 0
    if submitted:
        # Parse text input to float
        hemoglobin = float(hemoglobin) if hemoglobin else 0
        mch = float(mch) if mch else 0
        mchc = float(mchc) if mchc else 0
        mcv = float(mcv) if mcv else 0
        gender_binary = 1 if gender == "Female" else 0
        input_data = np.array([gender_binary, hemoglobin, mch, mchc, mcv]).reshape(1, -1)
        models = [model1,model2, model3, model4, model5]
        model_name= ['Random Forest','Decision Tree', 'Gaussian Naive Bayes','K-Neighbours','XGBoost']
        for i, model in enumerate(models):
            prediction = model.predict(input_data)
            result = "You have anemia" if prediction[0] == 1 else "You do not have anemia"
            if prediction[0] == 1:  # Assuming 1 indicates anemia
                result = "You have anemia"
                st.markdown(f'prediction from {model_name[i]} classifier:<p class="prediction-result-positive">{result}</p>', unsafe_allow_html=True)
            else:
                result = "You do not have anemia"
                st.markdown(f'prediction from {model_name[i]}:<p class="prediction-result-negative">{result}</p>', unsafe_allow_html=True)
            # en_result += prediction[0]
        
        final_prediction = hard_voting(models, input_data)
        en_result = "You have anemia" if final_prediction == 1 else "You do not have anemia"
    
    # Display the result from hard voting ensemble
        if final_prediction == 1:  # Assuming 1 indicates anemia
            st.markdown(f'prediction from Ensemble of all models above (hard voting): <p class="prediction-result-positive">{en_result}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'prediction from Ensemble of all models above (hard voting): <p class="prediction-result-negative">{en_result}</p>', unsafe_allow_html=True)
#to start the app, run 'streamlit run app.py' command on cmd