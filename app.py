import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('Automated Time series Forescasting')

"This data app uses Facebook's open=souce prophet library."

#step-1:Import Data

df = st.file_uploader('Import the time series csv file here.column name should be ds and y')

if df is not None:
    data = pd.read_csv(df)
    data['ds'] = pd.to_datetime(data['ds'],errors='coerce')

    st.write(data)

    max_date= data['ds'].max()
    #st.write(max_data) yo hataudai herni


#step 2: Select Forecasr Horizon

"Keep in mind that forecasts become less accurate"

periods_input =st.number_input('h(ow many periods would you like to forecast into the future?',
min_value=1, max_value=365)

if df is not None:
    m= Prophet()
    m.fit(data)#step 2: Select Forecasr Horizon

"Keep in mind that forecasts become less accurate"

periods_input =st.number_input('how many periods would you like to forecast into the future?',
min_value=1, max_value=365)

if df is not None:
    m= Prophet()
    m.fit(data)

#step 3: Visualize Forecast Data

if df is not None:
    future = m.make_future_dataframe(periods= periods_input)

    forecast = m.predict(future)
    fcst = forecast[['ds','yhat','yhat_lower','yhat_upper']]

    fcst_filtered = fcst[fcst['ds']>max_date]
    st.write(fcst_filtered)

"The next something something "

fig1 = m.plot(forecast)
st.write(fig1)

fig2= m.plot_components(forecast)
st.write(fig2)


#step 4: Download the Forecast Data

if df is not None:
        csv_exp=fcst_filtered.to_csv(index=False)
        # Encode the csv_exp variable to base64
        csv_exp_base64 = base64.b64encode(csv_exp.encode())
        # Decode the base64 bytes to a string
        csv_exp_base64 = csv_exp_base64.decode()
        href=f'<a href="data:file/csv ;base64,{csv_exp_base64}">Download CSV File </a>(right- click and save as &lt;forecast_name&gt;.csv)'
        st.markdown(href,unsafe_allow_html=True)



