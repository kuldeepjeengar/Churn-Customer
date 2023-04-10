# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:53:51 2023

@author: hp
"""


import streamlit as st


def TELECOM_CHURN():

    import numpy as np
    import pickle
    import streamlit as sts
    
    st.sidebar.markdown("Page 1")
    
    st.title("![Alt Text](https://imgur.com/s9R8Tt1.jpg)")
    

    st.sidebar.header('User Input Parameters')

    #loading the model
    loaded_model = pickle.load(open('Final_draft.sav','rb'))
        

    def churn_prediction(input_data):
        
        input_data_1 = np.asarray(input_data)
        #st.write(input_data_1.shape)
        input_data_1_reshaped = input_data_1.reshape(1,-1)
        #st.write(input_data_1_reshaped.shape)
        
        #checking the prediction
        prediction_1 = loaded_model.predict(input_data_1_reshaped)
        
        if(prediction_1[0] == 0):
            return ('This person is not going to churn')
        else:
            return ('This person is going to Churn')
        
        
        
    #def main():
    st.title('Telecom Churn Prediction')


    str_voice_plan     = st.sidebar.radio('Voice Mail Plan', ['Yes','No'])

    if str_voice_plan == 'Yes':
        voice_plan     = 1
    else:
        voice_plan     = 0
        
    #st.write(voice_mail_plan)
    account_length    = st.sidebar.number_input('Account Length',min_value=0)
    voice_messages     = st.sidebar.number_input('Voice Mail Messages',min_value=0)
    day_mins        = st.sidebar.number_input('Day Minutes')
    eve_mins        = st.sidebar.number_input('Evening Minutes')
    night_mins           = st.sidebar.number_input('Night Minutes')
    intl_mins  = st.sidebar.number_input('International Minutes')
    customer_calls  = st.sidebar.number_input('Customer Service Calls', min_value=0)
    str_intl_plan      = st.sidebar.radio('International Plan', ['Yes','No'])

    if str_intl_plan == 'Yes':
        intl_plan     = 1
    else:
        intl_plan     = 0

    #st.write(international_plan)
   
    intl_calls     = st.sidebar.number_input('International Calls', min_value=0)
    intl_charge     = st.sidebar.number_input('International Charge', min_value=0)
    day_calls    = st.sidebar.number_input('Day Calls', min_value=0)
    night_calls     = st.sidebar.number_input('Night Calls', min_value=0)
    eve_calls     = st.sidebar.number_input('Evening Calls', min_value=0)
    Total_Charge    = st.sidebar.number_input('Total Charges',min_value=0)
    
    # code for prediction
    churn_status = ''

    #creating submit button
    if st.button('Predict Churn Status'):
        churn_status= churn_prediction([account_length, voice_plan,voice_messages,eve_mins,night_mins,intl_mins,day_mins,customer_calls,intl_plan
                                        ,intl_calls,intl_charge,day_calls,eve_calls,night_calls,Total_Charge])


    st.success(churn_status)    

def VISUALISATION():
    import streamlit as st
    import pandas as pd 
    import matplotlib.pyplot as plt 
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns 
    
    st.title("DATA VISUALISATION ")
    st.sidebar.markdown("Page 2 ")
    st.title("![Alt Text](https://media1.tenor.com/images/b56165c4e1024f208ce84f63fa41befe/tenor.gif?itemid=14794342)")
    st.set_option('deprecation.showPyplotGlobalUse', False)


    def main():
	    """Semi Automated ML App with Streamlit """

	    activities = ["EDA","Plots"]	
	    choice = st.sidebar.selectbox("Select Activities",activities)

	    if choice == 'EDA':
		    st.subheader("Exploratory Data Analysis")

		    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		    if data is not None:
			     df = pd.read_csv(data)
			     st.dataframe(df.head())
			

			     if st.checkbox("Show Shape"):
				     st.write(df.shape)

			     if st.checkbox("Show Columns"):
				     all_columns = df.columns.to_list()
				     st.write(all_columns)

			     if st.checkbox("Summary"):
				     st.write(df.describe())

			     if st.checkbox("Show Selected Columns"):
				     selected_columns = st.multiselect("Select Columns",all_columns)
				     new_df = df[selected_columns]
				     st.dataframe(new_df)

			     if st.checkbox("Show Value Counts"):
				     st.write(df.iloc[:,-1].value_counts())

			     if st.checkbox("Correlation Plot(Matplotlib)"):
				     plt.matshow(df.corr())
				     st.pyplot()


			     if st.checkbox("Correlation Plot(Seaborn)"):
				     st.write(sns.heatmap(df.corr(),annot=True))
				     st.pyplot()


			     if st.checkbox("Pie Plot"):
				     all_columns = df.columns.to_list()
				     column_to_plot = st.selectbox("Select 1 Column",all_columns)
				     pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				     st.write(pie_plot)
				     st.pyplot()

	    elif choice == 'Plots':
		    st.subheader("Data Visualization")
		    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		    if data is not None:
			    df = pd.read_csv(data)
			    st.dataframe(df.head())

			    if st.checkbox("Show Value Counts"):
				    st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				    st.pyplot()
		
			    # Customizable Plot

			    all_columns_names = df.columns.tolist()
			    type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			    if st.button("Generate Plot"):
				    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				    if type_of_plot == 'area':
					    cust_data = df[selected_columns_names]
					    st.area_chart(cust_data)

				    elif type_of_plot == 'bar':
					    cust_data = df[selected_columns_names]
					    st.bar_chart(cust_data)

				    elif type_of_plot == 'line':
					    cust_data = df[selected_columns_names]
					    st.line_chart(cust_data)
                   
				# Custom Plot 
				    elif type_of_plot:
					    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					    st.write(cust_plot)
					    st.pyplot()

    if __name__ == '__main__':
	    main()


page_names_to_funcs = {
    "Telecom churn": TELECOM_CHURN,
    "Data Visualisation": VISUALISATION,

}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()