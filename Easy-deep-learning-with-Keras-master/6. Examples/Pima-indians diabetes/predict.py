from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from numpy import loadtxt


dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))




# make probability predictions with the model
predictions = loaded_model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]

# # make class predictions with the model
# # predictions = model.predict_classes(X)

for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))