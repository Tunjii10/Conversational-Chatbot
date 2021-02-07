import bot 

pipeline = bot.chatbot(dir_path = './data/chatbot_nlp/data',
					enc_dim = 200,
					lstm_dim = 300,
					optimizer = 'RMSprop',
					loss = 'categorical_crossentropy', 
					activation = 'softmax',
					batch_size = 32,
					epoch = 150
					)
					
if __name__ == '__main__':
	
	#train model
	pipeline.trainbot()
	
	#chat with bot
	pipeline.predict()
	