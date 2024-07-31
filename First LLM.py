# Databricks notebook source
# Install necessary libraries
%pip install --upgrade langchain
%pip install langchain
%pip install openai
%pip install pandas
%pip install openpyxl
%pip install langchain-community
%pip install -U langchain-openai
%pip install faiss-gpu
%pip install -U langchain-openai

# COMMAND ----------

import os
import pandas as pd
from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# COMMAND ----------

os.environ['OPENAI_API_KEY'] = 'sk-proj-yad5CHJqxIH7h2hcQipPT3BlbkFJ5ASordaoCVqxeOeY60lv'  # Replace with your actual API key

# COMMAND ----------

df = spark.read.csv("/FileStore/Calendar.csv", header=True, inferSchema=True)

# COMMAND ----------

pd_df = df.toPandas()

# COMMAND ----------

def create_qa_chain(df):
  # Load the CSV data into a pandas DataFrame
  #df = pd.read_csv(file_path)
  df.rename(columns={'Month_num': 'text'}, inplace=True)

  # Convert DataFrame to a list of documents
  loader = DataFrameLoader(df)
  documents = loader.load()

  # Create embeddings
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(documents, embeddings)

  # Create an LLM
  llm = OpenAI(temperature=0.7)

  # Create a QA chain
  qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
  return qa

# COMMAND ----------


qa_chain = create_qa_chain(pd_df)

# COMMAND ----------

query = "What is sum of Month_num ?"

# COMMAND ----------

response = qa_chain(query)

# COMMAND ----------

print(response)

