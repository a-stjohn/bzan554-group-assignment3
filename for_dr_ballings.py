import pandas as pd
import numpy as np
import tensorflow as tf


# load in data
data = pd.read_table(
    '/mnt/c/Users/amsj1/OneDrive - University of Tennessee/2nd_year/BZAN554_deep_learning/bzan554-group-assignment2/pricing_final.csv',
    delimiter=',',
    dtype={'sku': int,
           'price': float,
           'quantity': float,
           'order': int,
           'duration': float,
           'category': int})

# Compute number of categories
sku_nbr_unique = np.max(data['sku']) + 1
category_nbr_unique = np.max(data['category']) + 1

# # Inspect the FFNN models
# # had to put in excel and delimit on ',' and ']' to extract out what we needed
# ffnn_models = pd.read_csv(
#     'SavedNonLSTMModels/SavedForwardModels.csv',
#     header=None
# )
# # get the info of the model with the min loss
# min_model = ffnn_models[ffnn_models[12] == ffnn_models[12].min()]

# Get the mean and standard deviation of numerical features
def mean_and_std(var_name):
    return (np.mean(data[var_name]) , np.std(data[var_name]))

price_mean, price_std = mean_and_std('price')
quantity_mean, quantity_std = mean_and_std('quantity')
duration_mean, duration_std = mean_and_std('duration')
order_mean, order_std = mean_and_std('order')

#Numeric features
X_num = np.array(data[['price', 'duration', 'order']]).reshape(len(data), 3)
X_num = X_num - np.array([price_mean, duration_mean, order_mean])
X_num = X_num / np.array([price_std, duration_std, order_std])
# X_num = tf.convert_to_tensor(X_num) # this caused an error while train, test, splitting

y = data['quantity'] - np.array(quantity_mean)
y = y / np.array(quantity_std)

# make train, test, split
from sklearn.model_selection import train_test_split
X_num_train, X_num_test = train_test_split(
    X_num,
    test_size=0.3,
    random_state=420
)
X_num_train = tf.convert_to_tensor(X_num_train)
X_num_test = tf.convert_to_tensor(X_num_test)

X_sku_train, X_sku_test = train_test_split(
    data['sku'],
    test_size=0.3,
    random_state=420
)
X_cat_train, X_cat_test = train_test_split(
    data['category'],
    test_size=0.3,
    random_state=420
)
y_train, y_test = train_test_split(
    y,
    test_size=0.3,
    random_state=420
)

#####################################################
# LSTM Models
#####################################################

num_inputs = tf.keras.layers.Input(
    shape=(None, 3),
    name = 'X_num_inputs')

sku_input = tf.keras.layers.Input(
    shape=(1,),
    name = 'X_sku')
#output_dim should be more than this, say 30,000. Could also tune this.
sku_embedding = tf.keras.layers.Embedding(
    input_dim=sku_nbr_unique,
    output_dim=1000,
    input_length=3,
    name = 'emb_sku')(sku_input)
# sku_embedding_flat = tf.keras.layers.Flatten(
#     name='flatten_sku')(sku_embedding)

cat_input = tf.keras.layers.Input(
    shape=(1,),
    name = 'X_cat')
cat_embedding = tf.keras.layers.Embedding(
    input_dim=category_nbr_unique,
    output_dim=10,
    input_length=1,
    name = 'emb_cat')(cat_input)
# cat_embedding_flat = tf.keras.layers.Flatten(
#     name='flatten_cat')(cat_embedding)

inputs_concat = tf.keras.layers.Concatenate(
    name = 'concatenation')([cat_embedding, sku_embedding, num_inputs])

rrn1 = tf.keras.layers.LSTM(
    100,
    activation='elu',
    return_sequences=True,
    name='rrn1')(inputs_concat)
rrn2 = tf.keras.layers.LSTM(
    50,
    activation = 'elu',
    return_sequences=True)(rrn1)
rrn3 = tf.keras.layers.LSTM(
    20,
    activation = 'elu',
    return_sequences = True)(rrn2) #NOTE return_sequences = True
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(rrn3)

model = tf.keras.Model(inputs = inputs_concat, outputs = outputs)