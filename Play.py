import pandas as pd
import sqlite3
import os

df = pd.read_csv("telco.csv")

df.columns = df.columns.str.replace(' ', '_')

customer_df = df[
    ['Customer_ID', 'Gender', 'Age', 'Under_30', 'Senior_Citizen', 'Married', 'Dependents', 'Number_of_Dependents',
     'Country', 'State', 'City', 'Zip_Code', 'Latitude', 'Longitude', 'Population']]
service_df = df[
    ['Customer_ID', 'Phone_Service', 'Multiple_Lines', 'Internet_Service', 'Internet_Type', 'Online_Security',
     'Online_Backup', 'Device_Protection_Plan', 'Premium_Tech_Support', 'Streaming_TV', 'Streaming_Movies',
     'Streaming_Music', 'Unlimited_Data']]
billing_df = df[
    ['Customer_ID', 'Tenure_in_Months', 'Offer', 'Avg_Monthly_Long_Distance_Charges', 'Avg_Monthly_GB_Download',
     'Contract', 'Paperless_Billing', 'Payment_Method', 'Monthly_Charge', 'Total_Charges', 'Total_Refunds',
     'Total_Extra_Data_Charges', 'Total_Long_Distance_Charges', 'Total_Revenue']]
referral_df = df[['Customer_ID', 'Referred_a_Friend', 'Number_of_Referrals']]
churn_df = df[['Customer_ID', 'Quarter', 'Satisfaction_Score', 'Customer_Status', 'Churn_Label', 'Churn_Score', 'CLTV',
               'Churn_Category', 'Churn_Reason']]

conn = sqlite3.connect('telco.db')

customer_df.to_sql('Customer', conn, if_exists='replace', index=False)
service_df.to_sql('Service', conn, if_exists='replace', index=False)
billing_df.to_sql('Billing', conn, if_exists='replace', index=False)
referral_df.to_sql('Referral', conn, if_exists='replace', index=False)
churn_df.to_sql('Churn', conn, if_exists='replace', index=False)

conn.close()


# %%
def query_db(query):
    conn = sqlite3.connect('telco.db')
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


# %%

# %%
DB_DESCRIPTION = """You have access to the following tables and columns in a sqllite3 database:

Customer Table
Customer_ID: A unique ID that identifies each customer.
Gender: The customer’s gender: Male, Female.
Age: The customer’s current age, in years, at the time the fiscal quarter ended.
Under_30: Indicates if the customer is under 30: Yes, No.
Senior_Citizen: Indicates if the customer is 65 or older: Yes, No.
Married: Indicates if the customer is married: Yes, No.
Dependents: Indicates if the customer lives with any dependents: Yes, No.
Number_of_Dependents: Indicates the number of dependents that live with the customer.
Country: The country of the customer’s primary residence. Example: United States.
State: The state of the customer’s primary residence.
City: The city of the customer’s primary residence.
Zip_Code: The zip code of the customer’s primary residence.
Latitude: The latitude of the customer’s primary residence.
Longitude: The longitude of the customer’s primary residence.
Population: A current population estimate for the entire Zip Code area.

Service Table
Customer_ID: A unique ID that identifies each customer (Foreign Key).
Phone_Service: Indicates if the customer subscribes to home phone service with the company: Yes, No.
Multiple_Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No.
Internet_Service: Indicates if the customer subscribes to Internet service with the company: Yes, No.
Internet_Type: Indicates the type of Internet service: DSL, Fiber Optic, Cable, None.
Online_Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No.
Online_Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No.
Device_Protection Plan: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No.
Premium_Tech_Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No.
Streaming_TV: Indicates if the customer uses their Internet service to stream television programming from a third party provider: Yes, No.
Streaming_Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No.
Streaming_Music: Indicates if the customer uses their Internet service to stream music from a third party provider: Yes, No.
Unlimited_Data: Indicates if the customer has paid an additional monthly fee to have unlimited data downloads/uploads: Yes, No.

Billing Table
Customer_ID: A unique ID that identifies each customer (Foreign Key).
Tenure_in_Months: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.
Offer: Identifies the last marketing offer that the customer accepted, if applicable. Values include None, Offer A, Offer B, Offer C, Offer D, and Offer E.
Avg_Monthly_Long_Distance_Charges: Indicates the customer’s average long distance charges, calculated to the end of the quarter specified above.
Avg_Monthly_GB_Download: Indicates the customer’s average download volume in gigabytes, calculated to the end of the quarter specified above.
Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.
Paperless_Billing: Indicates if the customer has chosen paperless billing: Yes, No.
Payment_Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check.
Monthly_Charge: Indicates the customer’s current total monthly charge for all their services from the company.
Total_Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.
Total_Refunds: Indicates the customer’s total refunds, calculated to the end of the quarter specified above.
Total_Extra_Data_Charges: Indicates the customer’s total charges for extra data downloads above those specified in their plan, by the end of the quarter specified above.
Total_Long_Distance_Charges: Indicates the customer’s total charges for long distance above those specified in their plan, by the end of the quarter specified above.
Total_Revenue: The total revenue generated from the customer.

Referral Table
Customer_ID: A unique ID that identifies each customer (Foreign Key).
Referred_a_Friend: Indicates if the customer has ever referred a friend or family member to this company: Yes, No.
Number_of_Referrals: Indicates the number of referrals to date that the customer has made.

Churn Table
Customer_ID: A unique ID that identifies each customer (Foreign Key).
Quarter: The fiscal quarter that the data has been derived from (e.g. Q3).
Satisfaction_Score: A customer’s overall satisfaction rating of the company from 1 (Very Unsatisfied) to 5 (Very Satisfied).
Customer_Status: Indicates the status of the customer at the end of the quarter: Churned, Stayed, Joined.
Churn_Label: Yes = the customer left the company this quarter. No = the customer remained with the company.
Churn_Score: A value from 0-100 that is calculated using the predictive tool IBM SPSS Modeler. The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.
CLTV: Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.
Churn_Category: A high-level category for the customer’s reason for churning: Attitude, Competitor, Dissatisfaction, Other, Price.
Churn_Reason: A customer’s specific reason for leaving the company. Directly related to Churn Category.
"""
# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

can_answer_router_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a database reading bot that can answer users' questions using information from a database. \n

    {data_description} \n\n

    Given the user's question, decide whether the question can be answered using the information in the database. \n\n

    Return a JSON with two keys, 'reasoning' and 'can_answer', and no preamble or explanation.
    Return one of the following JSON:

    {{"reasoning": "I can find the average revenue of customers with tenure over 24 months by averaging the Total Revenue column in the Billing table filtered by Tenure in Months > 24", "can_answer":true}}
    {{"reasoning": "I can find customers who signed up during the last 12 month using the Tenure in Months column in the Billing table", "can_answer":true}}
    {{"reasoning": "I can't answer how many customers churned last year because the Churn table doesn't contain a year", "can_answer":false}}


    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} \n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["data_description", "question"],
)

messages = [
    {"role": "system", "content":
        """You are a database reading bot that can answer users' questions using information from a database. \n {data_description} \n\n

        Given the user's question, decide whether the question can be answered using the information in the database. \n\n

        Return a JSON with two keys, 'reasoning' and 'can_answer', and no preamble or explanation.
        Return one of the following JSON:

        {{"reasoning": "I can find the average revenue of customers with tenure over 24 months by averaging the Total Revenue column in the Billing table filtered by Tenure in Months > 24", "can_answer":true}}
        {{"reasoning": "I can find customers who signed up during the last 12 month using the Tenure in Months column in the Billing table", "can_answer":true}}
        {{"reasoning": "I can't answer how many customers churned last year because the Churn table doesn't contain a year", "can_answer":false}}

        """},
    {"role": "user", "content": "Question: {question} \n"},
]


import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# outputs = pipeline(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1])

#####
temp = 0
if temp:
    messages[0]["content"] = messages[0]["content"].format(data_description=DB_DESCRIPTION)
    messages[1]["content"] = messages[1]["content"].format(
        question="Count customers by zip code. Return the 5 most common zip codes")

    outputs = pipeline(messages, max_new_tokens=1024)
    print(outputs[0]["generated_text"][-1])


else:
    # messages[0]["content"] = messages[0]["content"].format(data_description=DB_DESCRIPTION)
    # messages[1]["content"] = messages[1]["content"].format(question="Count customers by zip code. Return the 5 most common zip codes")

    can_answer_router = can_answer_router_prompt | pipeline | JsonOutputParser()
    can_answer_router.invoke({"question": "Count customers by zip code. Return the 5 most common zip codes",
                              "data_description": DB_DESCRIPTION})