#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# logging.basicConfig(filename='api.log', level=logging.INFO)

# Load the ML model using pickle
with open('Model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Model/modelcolumns.pkl', 'rb') as columns_file:
    columns_used = pickle.load(columns_file)

    print(columns_used)


# Define new preprocessing functions
def balance_diff(data):
    orig_change = data['newbalanceOrig'] - data['oldbalanceOrg']
    orig_change = orig_change.astype(int)
    data['orig_txn_diff'] = np.where(orig_change < 0, round(data['amount'] + orig_change, 2),
                                    round(data['amount'] - orig_change, 2))
    data['orig_txn_diff'] = data['orig_txn_diff'].astype(int)
    data['orig_diff'] = np.where(data['orig_txn_diff'] != 0, 1, 0)

    dest_change = data['newbalanceDest'] - data['oldbalanceDest']
    dest_change = dest_change.astype(int)
    data['dest_txn_diff'] = np.where(dest_change < 0, round(data['amount'] + dest_change, 2),
                                    round(data['amount'] - dest_change, 2))
    data['dest_txn_diff'] = data['dest_txn_diff'].astype(int)
    data['dest_diff'] = np.where(data['dest_txn_diff'] != 0, 1, 0)

    data.drop(['orig_txn_diff', 'dest_txn_diff'], axis=1, inplace=True)

def surge_indicator(data):
    data['surge'] = np.where(data['amount'] > 450000, 1, 0)

def frequency_receiver(data):
    data['freq_dest'] = data['nameDest'].map(data['nameDest'].value_counts())
    data['freq_dest'] = data['freq_dest'].apply(lambda x: 1 if x > 20 else 0)

def merchant(data):
    values = ['M']
    conditions = list(map(data['nameDest'].str.contains, values))
    data['merchant'] = np.select(conditions, '1', '0')


def map_type_numeric(data):
    type_mapping = {'PAYMENT': 1, 'TRANSFER': 2, 'CASH_IN': 3, 'CASH_OUT': 4, 'DEBIT': 5}
    data['type_numeric'] = data['type'].map(type_mapping)
    data.drop(['type'], axis=1, inplace=True)

def tokenize_and_pad(data):
    tokenizer_org = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_org.fit_on_texts(data['nameOrig'])
    tokenizer_dest = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_dest.fit_on_texts(data['nameDest'])

    customers_org = tokenizer_org.texts_to_sequences(data['nameOrig'])
    customers_dest = tokenizer_dest.texts_to_sequences(data['nameDest'])

    data['customers_org'] = tf.keras.preprocessing.sequence.pad_sequences(customers_org, maxlen=1)
    data['customers_dest'] = tf.keras.preprocessing.sequence.pad_sequences(customers_dest, maxlen=1)

# Combine all preprocessing steps

def preprocess_data(data):
    print(data)
    balance_diff(data)
    print(data)
    surge_indicator(data)
    print(data)
    frequency_receiver(data)
    print(data)
    merchant(data)
    print(data)
    map_type_numeric(data)
    print(data)
    tokenize_and_pad(data)
    print(data)

    # Drop unnecessary columns
    data = data.drop(['nameOrig', 'nameDest','merchant'], axis=1)

    return data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Remove bankOrig and bankDest from the data
        if 'bankOrig' in data:
            del data['bankOrig']
        if 'bankDest' in data:
            del data['bankDest']

        # Perform preprocessing
        data_for_prediction = preprocess_data(pd.DataFrame([data]))

        # Make prediction
        prediction = model.predict(data_for_prediction)

        return jsonify({'prediction': int(prediction[0])})  # Assuming prediction is a single value (0 or 1)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)