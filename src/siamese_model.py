import pandas as pd
import argparse
import numpy as np
from keras.preprocessing import sequence, text
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, merge, Input 
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

EMBEDDING_LEN = 300

class Dataset():
	tokenizer = None
	data = None
	word_to_idx = None

	def __init__(self, datapath):
		self.tokenizer = text.Tokenizer()
		self.data = pd.read_csv(datapath, sep="\t")
		self.tokenizer.fit_on_texts(list(self.data.question1) + list(self.data.question2))
		self.word_to_idx = self.tokenizer.word_index

	def create_dataset(self, train_data_split):
		total_data_instances = len(self.data)
		# Shuffle the data indexes
		perm = np.random.permutation(self.data.index)
		train_end_idx = int(train_data_split*total_data_instances)
		# Create train and test based on input split
		train_data = self.data.iloc[perm[0:train_end_idx]]
		test_data = self.data.iloc[perm[train_end_idx:total_data_instances]]
		print "Number of train data instances read", len(train_data)
		print "Number of test data instances read", len(test_data)
		return train_data, test_data

	def process_dataframe(self, inpdata, max_len_sentence):
		X1 = self.tokenizer.texts_to_sequences(inpdata.question1)
		X1 = sequence.pad_sequences(X1, maxlen=max_len_sentence)
		X2 = self.tokenizer.texts_to_sequences(inpdata.question2)
		X2 = sequence.pad_sequences(X1, maxlen=max_len_sentence)
		Y = list(inpdata.is_duplicate)
		return X1,X2,Y

	def create_embedding_matrix(self, embeddings_path):
		embeddings = {}
		with open(embeddings_path) as f:
			for line in f:
				values = line.split()
				embedding = np.asarray(values[1:], dtype='float32')
				embeddings[values[0]] = embedding
		embedding_matrix = np.zeros((len(self.word_to_idx) + 1, EMBEDDING_LEN))
		for key in self.word_to_idx:
			if key in embeddings:
				embedding_matrix[self.word_to_idx[key]] = embeddings[key]
		return embedding_matrix

class SiameseModel():
	def __init__(self, num_vocab, embedding_matrix, max_len):
		lstm = Sequential()
		lstm.add(Embedding(input_dim=num_vocab, output_dim=EMBEDDING_LEN, \
			weights=[embedding_matrix], input_length=max_len, trainable=False))
		lstm.add(LSTM(128, dropout_W=0.5, dropout_U=0.5))
		lstm.add(Dense(1, activation='sigmoid'))

		l_input = Input(shape=(max_len,))
		r_input = Input(shape=(max_len,))

		l_output = lstm(l_input)
		r_output = lstm(r_input)

		merged_output = merge([l_output, r_output], mode='concat')

		fcl = Dense(100, activation='relu', W_regularizer=l2(0.0001), \
			b_regularizer=l2(0.0001))(merged_output)
		fcl_drop = Dropout(0.4)(fcl)
		prediction = Dense(1, activation='sigmoid')(fcl_drop)

		model = Model(input=[l_input, r_input], output=prediction)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		return model
		

### Main function which trains the model, tests the model and report metrics
def main(params):
	datapath = params["datapath"]
	train_data_split = params["train_data_split"]
	max_len_sentence = params["max_len_sentence"]
	embeddings_path = params["embeddings_path"]

	Ds = Dataset(datapath)
	train_data, test_data = Ds.create_dataset(train_data_split)
	X1_train, X2_train, Y_train = Ds.process_dataframe(train_data, max_len_sentence)
	# Storage reduction
	train_data = None
	print "Obtained processed training data"
	embedding_matrix = Ds.create_embedding_matrix(embeddings_path)
	print "Obtained embeddings"
	num_vocab = len(Ds.word_index) + 1
	model = SiameseModel(num_vocab, embedding_matrix, max_len_sentence)
	print "Built Model"
	X_dict = dict()
	X_dict['l_input'] = X1_train
	X_dict['r_input'] = X2_train
	print "Training now..."
	model.fit(x=X_dict, y=Y_train, batch_size=10, nb_epoch=10, \
                 verbose=1, validation_split=0.2, shuffle=True)
	# model.predict()

	

if __name__=='__main__':
	### Read user inputs
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", dest="datapath", type=str, default="../data/sample_data.tsv")
	parser.add_argument("--train_data_split", dest="train_data_split", type=float, default=0.8)
	parser.add_argument("--max_len_sentence", dest="max_len_sentence", type=int, default=40)
	parser.add_argument("--embeddings_path", dest="embeddings_path", type=str, default="../../../Data/glove.840B.300d.txt")
	params = vars(parser.parse_args())
	main(params)