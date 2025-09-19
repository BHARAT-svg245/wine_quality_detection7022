import streamlit as st
import numpy as np
import joblib

model=joblib.load("Wine_quality_prediction.joblib")
#Title the app
st.title("Wine_Quality_detector")
st.write("Enter the inputs:")
#Input from the user
#Id,fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality
f_acidity=st.number_input("Enter acidity:",min_value=4.0,max_value=16.0,value=7.8)
v_acidity=st.number_input("enter v_acid",min_value=0.11,max_value=1.60,value=0.56)
c_acid=st.number_input("enter c_acid:",min_value=0.0,max_value=1.0,value=0.57)
r_sugar=st.number_input("Enter r_sugar:",min_value=0.90,max_value=16.0,value=10.5)
cl=st.number_input("enter cl:",min_value=0.0120,max_value=0.615,value=0.540)
f_So2=st.number_input("enter f_so2:",min_value=1.0,max_value=68.0,value=53.0)
t_So2=st.number_input("enter c_acid:",min_value=5.0,max_value=290.0,value=210.0)
density=st.number_input("enter density:",min_value=0.98,max_value=1.50,value=1.001)
ph=st.number_input("enter ph:",min_value=2.5,max_value=5.0,value=3.0)
sulphate=st.number_input("enter sulphate:",min_value=0.30,max_value=2.0,value=1.45)
alc=st.number_input("enter alc:",min_value=8.0,max_value=15.0,value=11.2)

#Button action
if st.button("Predict Wine"):
    feature=np.array([[f_acidity,v_acidity,c_acid,r_sugar,cl,f_So2,t_So2,density,ph,sulphate,alc]])
    prediction=model.predict(feature)
    st.success(f"{prediction[0]}")
