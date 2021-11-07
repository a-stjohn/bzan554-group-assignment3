import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

########################
# load data for FFNN
########################

data_FFNN = pd.read_table(
    '/mnt/c/Users/amsj1/OneDrive - University of Tennessee/2nd_year/BZAN554_deep_learning/bzan554-group-assignment2/pricing_final.csv',
    delimiter=',',
    dtype={'sku': int,
           'price': float,
           'quantity': float,
           'order': int,
           'duration': float,
           'category': int})

# Compute number of categories
sku_nbr_unique = np.max(data_FFNN['sku']) + 1
category_nbr_unique = np.max(data_FFNN['category']) + 1

############################
# FFNN best model
############################

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
    return (np.mean(data_FFNN[var_name]) , np.std(data_FFNN[var_name]))

price_mean, price_std = mean_and_std('price')    
quantity_mean, quantity_std = mean_and_std('quantity')
duration_mean, duration_std = mean_and_std('duration')
order_mean, order_std = mean_and_std('order')

#Numeric features
X_num = np.array(data_FFNN[['price', 'duration', 'order']]).reshape(
    len(data_FFNN), 3)
X_num = X_num - np.array([price_mean, duration_mean, order_mean])
X_num = X_num / np.array([price_std, duration_std, order_std])
# X_num = tf.convert_to_tensor(X_num) # this caused an error while train, test, splitting

y = data_FFNN['quantity'] - np.array(quantity_mean)
y = y / np.array(quantity_std)

# Best FFNN model fitting (original best was chunks of data with bs of 1)
model.fit(
    x=[X_num[:1000], data_FFNN['sku'][:1000], data_FFNN['category'][:1000]],
    y=y[:1000],
    batch_size=5,
    epochs=20,
    callbacks=[tf.keras.callbacks.CSVLogger('FFNN_training.log')]
)


########################
# load data for LSTM
########################
data = pd.read_csv('https://ballings.co/hidden/data_pricing_prepared.csv')

X = []
y = []
for sku in data.sku.unique():
    X.append(data[data['sku']==sku][['price','order','duration']].to_numpy())
    y.append(data[data['sku']==sku][['quantity']].to_numpy())
X = np.array(X)
y = np.array(y)
#X: 'price','order','duration',
#y: 'quantity'

X.shape
y.shape
#X contains 26 timeseries of length 7 with 3 variables: 'price','order','duration'
#y contains 26 timeseries of length 7 with 1 variable: 'quantity'

nbrseries = X.shape[0]
nbrtimesteps = X.shape[1]
nbrvariables = X.shape[2]


################################
# LSTM Models
################################

# Architecture
inputs = tf.keras.layers.Input(shape=(None, nbrvariables))
rrn1 = tf.keras.layers.LSTM(nbrvariables+10, return_sequences = True, activation = 'elu')(inputs)
rrn2 = tf.keras.layers.LSTM(nbrvariables+8, return_sequences = True, activation = 'elu')(rrn1)
rrn3 = tf.keras.layers.LSTM(nbrvariables+3, return_sequences = True, activation = 'elu')(rrn2)
out = tf.keras.layers.Dense(units = 1, activation = 'linear')(rrn3)

################################
# Aaron's Adagrad model area
################################

tf.keras.backend.clear_session()

model = tf.keras.Model(inputs = inputs, outputs = out)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adagrad(
    learning_rate=0.001))

epochs = 100
adagrad_lossbyepoch = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(nbrseries):
        modinfo = model.fit(x=X[[i]],y=y[[i]], batch_size=1, epochs=1, verbose=0)
        loss = modinfo.history['loss'][0]
        avg_loss = avg_loss + (1/(i+1))*(loss - avg_loss)
    adagrad_lossbyepoch.append(avg_loss)

################################
# Dan's Nadam model area
################################

tf.keras.backend.clear_session()

model = tf.keras.Model(inputs = inputs, outputs = out)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Nadam(
    learning_rate=0.001))

epochs = 100
nadam_lossbyepoch = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(nbrseries):
        modinfo = model.fit(x=X[[i]],y=y[[i]], batch_size=1, epochs=1, verbose=0)
        loss = modinfo.history['loss'][0]
        avg_loss = avg_loss + (1/(i+1))*(loss - avg_loss)
    nadam_lossbyepoch.append(avg_loss)

################################
# Ethan's RMSprop model area
################################

tf.keras.backend.clear_session()

model = tf.keras.Model(inputs = inputs, outputs = out)
model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07))

epochs = 100
rmsprop_lossbyepoch = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(nbrseries):
        modinfo = model.fit(x=X[[i]],y=y[[i]], batch_size=1, epochs=1, verbose=0)
        loss = modinfo.history['loss'][0]
        avg_loss = avg_loss + (1/(i+1))*(loss - avg_loss)
    rmsprop_lossbyepoch.append(avg_loss)

################################
# Becky's Adam model area
################################

tf.keras.backend.clear_session()

model = tf.keras.Model(inputs = inputs, outputs = out)
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07))

epochs = 100
adam_lossbyepoch = []
for epoch in range(epochs):
    avg_loss = 0
    for i in range(nbrseries):
        modinfo = model.fit(x=X[[i]],y=y[[i]], batch_size=1, epochs=1, verbose=0)
        loss = modinfo.history['loss'][0]
        avg_loss = avg_loss + (1/(i+1))*(loss - avg_loss)
    adam_lossbyepoch.append(avg_loss)



#####################
# make pretty graphs
#####################

# read in the loss from the FFNN model
ffnn_loss = pd.read_csv('FFNN_training.log')

plt.plot(ffnn_loss['loss'], label = 'FFNN Loss')
plt.plot(adagrad_lossbyepoch, label = 'AdaGrad Loss')
plt.plot(nadam_lossbyepoch, label = 'Nadam Loss')
plt.plot(rmsprop_lossbyepoch, label = 'RMSprop Loss')
plt.plot(adam_lossbyepoch, label = 'Adam Loss')

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Comparison of Loss Between Models')
plt.show()

model.summary()