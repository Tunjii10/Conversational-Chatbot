#import libraries
import numpy as np
import tensorflow as tf
import random
import os
import pickle
import yaml
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical

#set seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)
tf.random.set_seed(SEED)


class chatbot:
	
	def __init__(self, dir_path, enc_dim, lstm_dim, optimizer, loss, activation, batch_size, epoch):
		
		#initialize parameters
		self.dir_path = dir_path
		self.enc_dim = enc_dim
		self.lstm_dim = lstm_dim
		self.activation = activation
		self.optimizer = optimizer
		self.loss = loss
		self.batch_size = batch_size
		self.epoch = epoch
		
		
	def preprocess(self):
	
		#extract data from data files into bot and human instances
		dir_list = os.listdir(self.dir_path + os.sep)
		human_txt = []
		bot_txt = []
		for filepath in dir_list:
			stream = open( self.dir_path + os.sep + filepath , 'rb')
			file = yaml.safe_load(stream)
			conversations = file['conversations']
			
			for sentences in conversations:
			  #capture multiple bot answers in one cell
				if len( sentences ) > 2 :
					human_txt.append(sentences[0])
					answer = ''
					replies = sentences[ 1 : ]
					for rep in replies:
						answer += ' ' + rep
					bot_txt.append(answer)
				elif len( sentences )> 1:
					human_txt.append(sentences[0])
					bot_txt.append(sentences[1])
			
			#change bot responses in list that are not string to string
			answers_ = []
			for i in range(len(bot_txt)):
				if type(bot_txt[i] ) == str:
					#append to answers_ list if str
					answers_.append(bot_txt[i])
				else:
					#else convert to str and append to answers_ list 
					bot_txt[i] = str(bot_txt[i])
					answers_.append( bot_txt[i] )

		#add <start> and <end> strings to bot txt
		bot_txt = []
		for i in range( len( answers_ ) ) :
			bot_txt.append( '<START> ' + answers_[i] + ' <END>' )		
		
		
		return human_txt, bot_txt
		
		
	#tokenize words and create vocab dictionary
	def vocab_creator(self,human_txt,bot_txt):
		tokenizer = preprocessing.text.Tokenizer()
		tokenizer.fit_on_texts(human_txt + bot_txt)
		VOCAB_SIZE = len( tokenizer.word_index )+1
		dictionary = tokenizer.word_index
		#save dictionary
		pickle.dump(dictionary, open("./saved_model/dictionary.p", "wb"))
		
		return VOCAB_SIZE,dictionary,tokenizer
		
	#turn sentences to sequences and pad sequence
	def sequence_maker(self, VOCAB_SIZE, tokenizer, human_txt, bot_txt):
		# encoder_input_data
		tokenized_human_txt = tokenizer.texts_to_sequences(human_txt)
		maxlen_human_txt = max([len(x) for x in tokenized_human_txt])
		encoder_input_data = pad_sequences(tokenized_human_txt , maxlen=maxlen_human_txt, padding='post')
		print(encoder_input_data.shape)

		# decoder_input_data
		tokenized_bot_txt = tokenizer.texts_to_sequences(bot_txt)
		maxlen_bot_txt = max([len(x) for x in tokenized_bot_txt])
		decoder_input_data = pad_sequences(tokenized_bot_txt , maxlen=maxlen_bot_txt, padding='post')
		print(decoder_input_data.shape)

		# decoder_output_data
		for i in range(len(tokenized_bot_txt)) :
		  #remove start tag word for decoder_output data
		  tokenized_bot_txt[i] = tokenized_bot_txt[i][1:]
		padded_bot_txt = pad_sequences(tokenized_bot_txt , maxlen=maxlen_bot_txt, padding='post')
		#one hot encode
		decoder_output_data = to_categorical( padded_bot_txt , VOCAB_SIZE )
		print( decoder_output_data.shape )
		
		
		return encoder_input_data, decoder_input_data, decoder_output_data,maxlen_human_txt,maxlen_bot_txt

	#model architecture
	def seq2seq_model(self, maxlen_human_txt, maxlen_bot_txt, VOCAB_SIZE):
		encoder_inputs = Input(shape=( maxlen_human_txt , ))
		encoder_embedding = Embedding( VOCAB_SIZE, self.enc_dim , mask_zero=True ) (encoder_inputs)
		encoder_outputs , state_h , state_c = LSTM( self.lstm_dim , return_state=True )( encoder_embedding )
		encoder_states = [ state_h , state_c ]

		decoder_inputs = Input(shape=( maxlen_bot_txt ,  ))
		decoder_embedding = Embedding( VOCAB_SIZE, self.enc_dim , mask_zero=True) (decoder_inputs)
		decoder_lstm = LSTM( self.lstm_dim , return_state=True , return_sequences=True )
		decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
		decoder_dense = Dense( VOCAB_SIZE , activation=self.activation ) 
		output = decoder_dense ( decoder_outputs )

		model = Model([encoder_inputs, decoder_inputs], output )
		model.compile(optimizer=self.optimizer, loss=self.loss)

		model.summary()
		return model		
	  
	#fit and save model
	def fit(self, encoder_input_data, decoder_input_data, decoder_output_data, model):
		model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=self.batch_size, epochs=self.epoch ) 
		model.save( './saved_model/model.h5' )
		
		return self
	

		
	#train chat bot
	def trainbot(self):
		human_txt,bot_txt = self.preprocess()
		VOCAB_SIZE,dictionary,tokenizer = self.vocab_creator(human_txt,bot_txt)
		encoder_input_data, decoder_input_data, decoder_output_data,maxlen_human_txt,maxlen_bot_txt = self.sequence_maker( VOCAB_SIZE, tokenizer,human_txt,bot_txt)
		model= self.seq2seq_model(maxlen_human_txt,maxlen_bot_txt, VOCAB_SIZE)
		self.fit( encoder_input_data, decoder_input_data, decoder_output_data, model)
	
		

	#prediction model architecture
	def inference_models(self):
		
		training_model = load_model('./saved_model/model.h5')
		encoder_inputs = training_model.inputs[0]
		encoder_outputs, state_h_enc, state_c_enc = training_model.layers[4].output
		encoder_states = [state_h_enc, state_c_enc]
		encoder_model = Model(encoder_inputs, encoder_states)
		decoder_state_input_h = Input(shape=( self.lstm_dim ,))
		decoder_state_input_c = Input(shape=( self.lstm_dim ,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		
		decoder_inputs = training_model.inputs[1]
		decoder_embedding = training_model.layers[3].output
		decoder_lstm = training_model.layers[5]
		decoder_dense = training_model.layers[6]
		
		decoder_outputs, state_h, state_c = decoder_lstm(
			decoder_embedding, initial_state=decoder_states_inputs)
		decoder_states = [state_h, state_c]
		decoder_outputs = decoder_dense(decoder_outputs)
		decoder_model = Model(
			[decoder_inputs] + decoder_states_inputs,
			[decoder_outputs] + decoder_states)
		
		return encoder_model , decoder_model
	
	
	#preprocess user input ie convert to token
	def str_to_tokens( self, sentence : str ):
		dictionary = pickle.load(open("./saved_model/dictionary.p", "rb"))
		words = sentence.replace('"','')
		words = words.replace(',','')
		words = words.strip('!?')
		words = words.lower().split()
		tokens_list = []
		for word in words:
			tokens_list.append(dictionary[word]) 
		return pad_sequences( [tokens_list] , maxlen=22 , padding='post')
		

	#prediction model process /chatbot output
	def predict(self):
		dictionary = pickle.load(open("./saved_model/dictionary.p", "rb"))
		enc_model , dec_model = self.inference_models()
		for i in range(5):
			states_values = enc_model.predict( self.str_to_tokens( input( 'Enter question : ' ) ) )
			empty_target_seq = np.zeros( ( 1 , 1 ) )
			empty_target_seq[0, 0] = dictionary['start']
			stop_condition = False
			decoded_translation = ''
			while not stop_condition :
				dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
				sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
				sampled_word = None
				for word , index in dictionary.items() :
					if sampled_word_index == index :
						decoded_translation += ' {}'.format( word )
						sampled_word = word
				
				if sampled_word == 'end' or len(decoded_translation.split()) > 74:
					stop_condition = True
					
				empty_target_seq = np.zeros( ( 1 , 1 ) )  
				empty_target_seq[ 0 , 0 ] = sampled_word_index
				states_values = [ h , c ] 
		 
			print( decoded_translation )
			
		
		