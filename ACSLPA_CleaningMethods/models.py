from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.layers import LeakyReLU

def ae_1(input_dim, **kwargs):
	input_layer = Input(shape=(input_dim,))
	encoder= Dense(3, activation='relu')(input_layer)
	decoder = Dense(input_dim, activation='linear')(encoder)
	return Model(input_layer, decoder)

def ae_1_l1(input_dim, l1_factor=10e-5):
	input_layer = Input(shape=(input_dim,))
	encoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(input_layer)
	decoder = Dense(input_dim, activation='linear')(encoder)
	return Model(input_layer, decoder)

def ae_2(input_dim, **kwargs):
	input_layer = Input(shape=(input_dim,))
	encoder= Dense(5, activation='relu')(input_layer)
	encoder= Dense(3, activation='relu')(encoder)
	decoder= Dense(5, activation='relu')(encoder)
	decoder = Dense(input_dim, activation='linear')(decoder)
	return Model(input_layer, decoder)

def ae_2_l1(input_dim, l1_factor=10e-5):
	input_layer = Input(shape=(input_dim,))
	encoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(input_layer)
	encoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(encoder)
	decoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(encoder)
	decoder = Dense(input_dim, activation='linear')(decoder)
	return Model(input_layer, decoder)
	
def ae_3(input_dim, **kwargs):
	input_layer = Input(shape=(input_dim,))
	encoder= Dense(6, activation='relu')(input_layer)
	encoder= Dense(5, activation='relu')(encoder)
	encoder= Dense(3, activation='relu')(encoder)
	decoder= Dense(5, activation='relu')(encoder)
	decoder= Dense(6, activation='relu')(decoder)
	decoder = Dense(input_dim, activation='linear')(decoder)
	return Model(input_layer, decoder)

def ae_3_l1(input_dim, l1_factor=10e-5):
	input_layer = Input(shape=(input_dim,))
	encoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(input_layer)
	encoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(encoder)
	encoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(encoder)
	decoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(encoder)
	decoder= Dense(7, activation='relu', activity_regularizer=regularizers.l1(l1_factor))(decoder)
	decoder = Dense(input_dim, activation='linear')(decoder)
	return Model(input_layer, decoder)
