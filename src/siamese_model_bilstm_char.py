import pandas as pd
import argparse
import numpy as np
from keras.preprocessing import sequence, text
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, merge, Input, Bidirectional
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support as score

EMBEDDING_LEN = 300
CHAR_LEN = 100

class Dataset():
	tokenizer = None
	data = None
	word_to_idx = None

	def __init__(self, datapath):
		self.tokenizer = text.Tokenizer()
		self.data = pd.read_csv(datapath, sep="\t")
		self.data.question1 = self.data.question1.astype(str)
		self.data.question2 = self.data.question2.astype(str)
		self.tokenizer.fit_on_texts(list(self.data.question1) + list(self.data.question2))
		self.word_to_idx = self.tokenizer.word_index
		self.c_tokenizer = text.Tokenizer(char_level=True)
		self.c_tokenizer.fit_on_texts(list(self.data.question1) + list(self.data.question2))
		self.char_to_idx = self.c_tokenizer.word_index

	def create_dataset(self, train_data_split):
		total_data_instances = len(self.data)
		# Shuffle the data indexes
		np.random.seed(42)
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
		X2 = sequence.pad_sequences(X2, maxlen=max_len_sentence)
		X1_c = self.c_tokenizer.texts_to_sequences(inpdata.question1)
		X1_c = sequence.pad_sequences(X1_c, maxlen=max_len_sentence*6)
		X2_c = self.c_tokenizer.texts_to_sequences(inpdata.question2)
		X2_c = sequence.pad_sequences(X2_c, maxlen=max_len_sentence*6)
		Y = inpdata.is_duplicate
		return X1,X2,X1_c,X2_c,Y

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
			else:
				embedding_matrix[self.word_to_idx[key]] = np.random.randint(-10000, high=10000, size=(300,))/10000.0
		return embedding_matrix

class SiameseModel():
	def build_model(self, num_vocab, embedding_matrix, num_char_vocab, \
					char_embedding_matrix, max_len_sentence):
		embed_word = Embedding(input_dim=num_vocab, output_dim=EMBEDDING_LEN, \
			weights=[embedding_matrix], input_length=max_len, trainable=False)

		embed_c = Embedding(input_dim=num_char_vocab, output_dim=CHAR_LEN, \
			weights=[char_embedding_matrix], input_length=max_len*6, trainable=True)
		embed_char = Bidirectional(LSTM(100))(embed_c)

		l_input = Input(shape=(max_len,))
		r_input = Input(shape=(max_len,))
		l_c_input = Input(shape=(max_len*6,))
		r_c_input = Input(shape=(max_len*6,))

		l_input_processed = embed_word(l_input)
		r_input_processed = embed_word(r_input)

		l_c_input_processed = embed_char(l_c_input)
		r_c_input_processed = embed_char(r_c_input)

		l_input_merged = merge([l_input_processed, l_c_input_processed], mode='concat')
		r_input_merged = merge([r_input_processed, r_c_input_processed], mode='concat')

		#TODO - Check dropout
		lstm_b = Bidirectional(LSTM(256, dropout_W=0.2, dropout_U=0.2))
		l_output = lstm_b(l_input_merged)
		r_output = lstm_b(r_input_merged)

		merged_output = merge([l_output, r_output], mode='concat')

		fcl = Dense(100, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001))(merged_output)
		fcl = Dense(50, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001))(fcl)
		#fcl_drop = Dropout(0.4)(fcl)
		fcl = Dense(25, activation='relu', W_regularizer=l2(0.0001), b_regularizer=l2(0.0001))(fcl)
		prediction = Dense(1, activation='sigmoid')(fcl)

		model = Model(input=[l_input, r_input, l_c_input, r_c_input], output=prediction)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		return model
		

### Main function which trains the model, tests the model and report metrics
def main(params):
	datapath = params["datapath"]
	train_data_split = params["train_data_split"]
	max_len_sentence = params["max_len_sentence"]
	embeddings_path = params["embeddings_path"]
	model_path = params["model_path"]

	Ds = Dataset(datapath)
	embedding_matrix = Ds.create_embedding_matrix(embeddings_path)
	char_embedding_matrix = np.random.randint(-10000, high=10000, size=(len(Ds.char_to_idx) + 1, CHAR_LEN))/10000.0
	print "Obtained embeddings"
	train_data, test_data = Ds.create_dataset(train_data_split)
	X1_train, X2_train, X1_c, X2_c, Y_train = Ds.process_dataframe(train_data, max_len_sentence)
	# Storage reduction
	train_data = None
	print "Obtained processed training data"
	num_vocab = len(Ds.word_to_idx) + 1
	num_char_vocab = len(Ds.char_to_idx) + 1

	Sm = SiameseModel()
	model = Sm.build_model(num_vocab, embedding_matrix,num_char_vocab, \
					char_embedding_matrix, max_len_sentence)
	print "Built Model"
	print "Training now..."
	filepath=model_path + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x=[X1_train, X2_train, X1_c, X2_c], y=Y_train, batch_size=128, epochs=50, verbose=1, validation_split=0.2, shuffle=True, callbacks=callbacks_list)

	X1_test, X2_test, X1_ct, X2_ct, Y_test = Ds.process_dataframe(test_data, max_len_sentence)
	pred = model.predict([X1_test, X2_test, X1_ct, X2_ct], batch_size=32, verbose=0)
	precision, recall, fscore, support = score(Y_test, pred.round(), labels=[0, 1])

	print "Metrics on test dataset"
	print('precision: {}'.format(precision))
	print('recall: {}'.format(recall))
	print('fscore: {}'.format(fscore))
	print('support: {}'.format(support))

if __name__=='__main__':
	### Read user inputs
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", dest="datapath", type=str, default="../../Data/quora_duplicate_questions.tsv")
	# parser.add_argument("--datapath", dest="datapath", type=str, default="../data/sample_data.tsv")
	parser.add_argument("--train_data_split", dest="train_data_split", type=float, default=0.8)
	parser.add_argument("--max_len_sentence", dest="max_len_sentence", type=int, default=40)
	parser.add_argument("--embeddings_path", dest="embeddings_path", type=str, default="../../Data/glove.840B.300d.txt")
	parser.add_argument("--model_path", dest="model_path", type=str, default="../models/")
	params = vars(parser.parse_args())
	main(params)