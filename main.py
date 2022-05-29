import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import streamlit.components.v1 as components

st. set_page_config(layout="wide")

st.header("Overview of trends in Automative Industry and Car Price Prediction")
st.text(" ")
st.text(" ")


col1,col2,col3 = st.columns([3,4,3])

data = pd.read_csv("https://raw.githubusercontent.com/kumarnandini/Automative-Industry-Data-Analysis-and-Car-Price-Prediction/main/car_price.csv")

with col1:
    st.text("The automotive industry is shifting gears.")
    st.text("Global disruption, technological advances, ")
    st.text("and changing consumer behaviors are altering")
    st.text("the auto industry on many levels all at once.")
    st.text("The traditional business model of designing,")
    st.text("manufacturing, selling, servicing, and ")
    st.text("financing vehicles continues.")
    st.text(" ")
    # st.text("Yet at the same time, the automotive industry")
    # st.text("is racing toward a new world, driven by ")
    # st.text("sustainability and changing consumer behavior")
    # st.text(", encompassing electric vehicles, connected ")
    # st.text("cars, mobility fleet sharing, onboard sensors,")
    # st.text(" new business models, and always-on connectedness.")


with col2:
    for i in data[['fueltype', 
        'carbody', 'drivewheel','peakrpm', 'enginetype', 'cylindernumber', 'fuelsystem']]:
        fig=plt.figure(figsize=(10,4))
        sns.countplot(data[i], data = data, palette='rocket_r')
        plt.xticks(rotation = 90)
        st.pyplot(fig)
        # plt.show()icefire_rrocket_r


with col1:
    symbol = st.number_input("Enter symboling")
    wheelbase = st.number_input("Enter wheelbase")
    carlength = st.number_input("Enter car length")
    carwidth = st.number_input("Enter car width")
    carheight = st.number_input("Enter car height")
    curbweight = st.number_input("Enter curb weight")
    enginesize = st.number_input("Enter engine size")
    boreratio = st.number_input("Enter bore ratio")
    stroke = st.number_input("Enter stroke")
    compratio = st.number_input("Enter compression ratio")
    horsepower = st.number_input("Enter horsepower")
    peakrpm = st.number_input("Enter peak rpm")
    citympg = st.number_input("Enter city mpg")
    highwaympg = st.number_input("Enter highway mpg")

features = np.array([[symbol,wheelbase,carlength,carwidth,carheight,curbweight,enginesize,boreratio,stroke,compratio,horsepower,peakrpm,citympg,highwaympg]])

data['symboling'] = data['symboling'].astype('object')
import re
p = re.compile(r'\w+-?\w+')
datanames = data['CarName'].apply(lambda x: re.findall(p, x)[0])

data['data_company'] = data['CarName'].apply(lambda x: re.findall(p, x)[0])

# volkswagen
data.loc[(data['data_company'] == "vw") | 
         (data['data_company'] == "vokswagen")
         , 'data_company'] = 'volkswagen'

# porsche
data.loc[data['data_company'] == "porcshce", 'data_company'] = 'porsche'

# toyota
data.loc[data['data_company'] == "toyouta", 'data_company'] = 'toyota'

# nissan
data.loc[data['data_company'] == "Nissan", 'data_company'] = 'nissan'

# mazda
data.loc[data['data_company'] == "maxda", 'data_company'] = 'mazda'
data['data_company'].astype('category').value_counts()

predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)

prediction = model.predict(features)

with col1:
    if(st.button("Submit")):
        st.text("Estimated car price in dollars:")
        st.text("{}".format(prediction))

with col3:
    fig3 = plt.figure()
    sns.distplot(data['price'], color='black')
    st.pyplot(fig3)

    fig4 = plt.figure()
    from sklearn.preprocessing import PowerTransformer
    p = PowerTransformer(method = 'box-cox')
    data['price'] = p.fit_transform(data[['price']])
    sns.distplot(data.price, color='blue')
    st.pyplot(fig4)

    fig2 = plt.figure()
    sns.distplot(data['wheelbase'], color='indigo')
    st.pyplot(fig2)

    fig5 = plt.figure()
    sns.distplot(data['curbweight'],color='black')
    st.pyplot(fig5)

    fig6 = plt.figure()
    sns.distplot(data['stroke'])
    st.pyplot(fig6)

    fig7 = plt.figure()
    sns.distplot(data['compressionratio'], color='brown')
    st.pyplot(fig7)

st.components.v1.iframe("https://app.powerbi.com/reportEmbed?reportId=e0998bfe-7920-42ab-9529-02ea9a11a692&autoAuth=true&ctid=ff65bb2a-d8a6-4a70-bfb2-79b1a8746349&config=eyJjbHVzdGVyVXJsIjoiaHR0cHM6Ly93YWJpLWluZGlhLWNlbnRyYWwtYS1wcmltYXJ5LXJlZGlyZWN0LmFuYWx5c2lzLndpbmRvd3MubmV0LyJ9", height=1000)

