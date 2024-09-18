from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri="mongodb+srv://ag8712117:dHw69YcUjLQ2hIVv@cluster0.i3wz8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client=MongoClient(uri)

DATABASE_NAME="SENSORDB"
COLLECTION_NAME="WAFERFAULT"

df=pd.read_csv("E:\\Ankit\\Ml Modal\\Sensor_Fault_Detection\\notebooks\\wafer_23012020_041211.csv")

df=df.drop("Unnamed: 0",axis=1)

#converting the data into json file 
json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
print 



