# Group assignment 3

# Use the data, inputs and outputs as specified in group assignment 2.

# Given everything we have learned, compare the performance of a feed-forward neural network and a LSTM.

# Answer:
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

#####################################################
# FFNN best model
#####################################################

# # Inspect the FFNN models
# # had to put in excel and delimit on ',' and ']' to extract out what we needed
# ffnn_models = pd.read_csv(
#     'SavedNonLSTMModels/SavedForwardModels.csv',
#     header=None
# )
# # get the info of the model with the min loss
# min_model = ffnn_models[ffnn_models[12] == ffnn_models[12].min()]

# rerun the best FFNN model to compare with
def build_model(nbr_hidden_layers = 1,
                nbr_hidden_neurons_per_layer = [30],
                hidden_activation_function_per_layer = ['relu'],
                optimizer = 'Adam',
                learning_rate = 0.001,
                batch_size = 1):

        #destroy any previous models
        #https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
        tf.keras.backend.clear_session()
        
        #define architecture: layers, neurons, activation functions
        #price, duration, order
        num_inputs = tf.keras.layers.Input(
            shape=(3,), 
            name = 'X_num_inputs')
        
        sku_input = tf.keras.layers.Input(
            shape=(1,), 
            name = 'X_sku')
        #output_dim should be more than this, say 30,000. Could also tune this.
        sku_embedding = tf.keras.layers.Embedding(
            input_dim=sku_nbr_unique,
            output_dim=1000,
            input_length=1,
            name = 'emb_sku')(sku_input)
        sku_embedding_flat = tf.keras.layers.Flatten(
            name='flatten_sku')(sku_embedding)
        
        cat_input = tf.keras.layers.Input(
            shape=(1,), name = 'X_cat' )
        cat_embedding = tf.keras.layers.Embedding(
            input_dim=category_nbr_unique, 
            output_dim=10, 
            input_length=1,
            name = 'emb_cat')(cat_input)

        cat_embedding_flat = tf.keras.layers.Flatten(
            name='flatten_cat')(cat_embedding)

        inputs_concat = tf.keras.layers.Concatenate(
            name = 'concatenation')([cat_embedding_flat, sku_embedding_flat, num_inputs])
        
        for layer in range(nbr_hidden_layers):
            if layer == 0:
                x = tf.keras.layers.Dense(units=nbr_hidden_neurons_per_layer[layer],
                                      activation=hidden_activation_function_per_layer[layer],
                                      name = 'hidden' + str(layer))(inputs_concat)
            else:
                x = tf.keras.layers.Dense(units=nbr_hidden_neurons_per_layer[layer],
                                      activation=hidden_activation_function_per_layer[layer],
                                      name = 'hidden' + str(layer))(x)
                
        outputs = tf.keras.layers.Dense(units=1,
                                      activation="linear",
                                      name = 'output')(x)
        
        model = tf.keras.Model(inputs = [num_inputs,sku_input,cat_input], 
                       outputs = outputs)

        #define optimizer and learning rate
        if optimizer == 'SGD':
            opt = tf.keras.optimizers.SGD(learning_rate = learning_rate)
        elif optimizer == 'SGDwithLRscheduling':
            initial_learning_rate = learning_rate
            decay_steps = 10000
            decay_rate = 1/10
            learning_schedule= tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps,
                decay_rate)
            opt = tf.keras.optimizers.SGD(learning_rate=learning_schedule)
        elif optimizer == 'RMSprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                              rho=0.9,
                                              momentum=0.0,
                                              epsilon=1e-07)
        elif optimizer == 'Adam':            
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                           beta_1=0.9,
                                           beta_2=0.999,
                                           epsilon=1e-07)

        model.compile(loss = 'mse', optimizer = opt)

        #return model and batch_size
        return [model,batch_size]

model, batch = build_model()

# Get the mean and standard deviation of numerical features
def mean_and_std(var_name):
    return (np.mean(data[var_name]) , np.std(data[var_name]))

price_mean, price_std = mean_and_std('price')    
quantity_mean, quantity_std = mean_and_std('quantity')
duration_mean, duration_std = mean_and_std('duration')
order_mean, order_std = mean_and_std('order')

#Numeric features
X_num = np.array(data[['price', 'duration', 'order']]).reshape(
    len(data), 3)
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

# Best FFNN model fitting (original best was chunks of data with bs of 1)
model.fit(
    x=[X_num_train[:1000], X_sku_train[:1000], X_cat_train[:1000]],
    y=y_train[:1000],
    batch_size=5,
    epochs=10,
    callbacks=[tf.keras.callbacks.CSVLogger('FFNN_training.log')]
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
    input_length=1,
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

#####################################################
# Aaron's model area
#####################################################

#Create model
model = tf.keras.Model(inputs = inputs_concat, outputs = outputs)

#Compile model
model.compile(
    loss = 'mse',
    optimizer = tf.keras.optimizers.Adagrad(learning_rate = 0.001)
)
#Fit model
model.fit(
    x=[X_num_train[:1000], X_sku_train[:1000], X_cat_train[:1000]],
    y=y_train[:1000],
    batch_size=5,
    epochs=10,
    callbacks=[tf.keras.callbacks.CSVLogger('logs/aaron_AdaGrad_training.log')]
)

model.save('saved_models/aaron_AdaGrad')

#####################################################
# Dan's model area
#####################################################

inputs = tf.keras.layers.Input(shape = (None,3)) #no need to specify length of sequence (set first dimension to None)
rrn1 = tf.keras.layers.LSTM(
    100,
    activation='elu',
    return_sequences=True)(inputs)
rrn2 = tf.keras.layers.LSTM(
    50,
    activation = 'elu',
    return_sequences=True)(rrn1)
rrn3 = tf.keras.layers.LSTM(
    20,
    activation = 'elu',
    return_sequences = True)(rrn2) #NOTE return_sequences = True
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(rrn3)

#Create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Compile model
model.compile(
    loss = 'mse',
    optimizer = tf.keras.optimizers.Nadam(learning_rate = 0.001))

#Fit model
model.fit(
    x=[X_num_train[:1000], X_sku_train[:1000], X_cat_train[:1000]],
    y=y_train[:1000],
    batch_size=5,
    epochs=10,
    callbacks=[tf.keras.callbacks.CSVLogger('logs/dan_Nadam_training.log')]
)

model.save('saved_models/dan_Nadam')

#####################################################
# Becky's Model area - Adam
#####################################################
inputs = tf.keras.layers.Input(shape = (None,1)) #no need to specify length of sequence (set first dimension to None)
rrn1 = tf.keras.layers.LSTM(
    100,
    activation='elu',
    return_sequences=True)(inputs)
rrn2 = tf.keras.layers.LSTM(
    50,
    activation = 'elu',
    return_sequences=True)(rrn1)
rrn3 = tf.keras.layers.LSTM(
    20,
    activation = 'elu',
    return_sequences = True)(rrn2) #NOTE return_sequences = True
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(rrn3)

#Create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Compile model
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, rho=0.9, momentum=0.0, epsilon=1e-07))

#Fit model
model.fit(x=[X_num_train[:1000],X_sku_train[:1000],X_cat_train[:1000]],y=y_train[:1000], batch_size=5, epochs=10,callbacks=[tf.keras.callbacks.CSVLogger('logs/becky_Adam.log')])

#Save model
model.save('saved_models/becky_Adam')

#####################################################
# Ethan's Model area - RMSprop
#####################################################
inputs = tf.keras.layers.Input(shape = (None,1)) #no need to specify length of sequence (set first dimension to None)
rrn1 = tf.keras.layers.LSTM(
    100,
    activation='elu',
    return_sequences=True)(inputs)
rrn2 = tf.keras.layers.LSTM(
    50,
    activation = 'elu',
    return_sequences=True)(rrn1)
rrn3 = tf.keras.layers.LSTM(
    20,
    activation = 'elu',
    return_sequences = True)(rrn2) #NOTE return_sequences = True
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(rrn3)

#Create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Compile model
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001, rho=0.9, momentum=0.0, epsilon=1e-07))

#Fit model
model.fit(x=[X_num_train[:1000],X_sku_train[:1000],X_cat_train[:1000]],y=y_train[:1000], batch_size=5, epochs=10,callbacks=[tf.keras.callbacks.CSVLogger('logs/ethanRMSprop.log')])

model.save('saved_models/Ethan_RMSprop')