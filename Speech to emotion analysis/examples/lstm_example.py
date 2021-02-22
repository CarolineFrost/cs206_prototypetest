"""
This example demonstrates how to use `LSTM` model from
`speechemotionrecognition` package
"""

from keras.utils import np_utils
from tensorflow import keras
from pydub import AudioSegment
import os

from common import extract_data
from speechemotionrecognition.dnn import LSTM
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


# split an audio file into segments and apply the model segment by segment
def section_by_section_analysis(model, audio_file, to_flatten):
    class_labels= ["Neutral", "Angry", "Happy", "Sad"]
    entireAudio = AudioSegment.from_wav(audio_file)
    chunkSize = 2000
    numChunks = len(entireAudio) / chunkSize
    temporaryFileName = 'tempFile.wav'
    print(len(entireAudio))
    print(numChunks)
    previousEndTime = 0
    for endTime in range(chunkSize, len(entireAudio)+chunkSize, chunkSize):
        if (endTime > len(entireAudio)):
            lastBit = (len(entireAudio) - previousEndTime)*-1
            newAudio = entireAudio[lastBit:]
        else:
            newAudio = entireAudio[previousEndTime:endTime]
        newAudio.export(temporaryFileName, format="wav") #Exports to a wav file in the current path.
        predicted_value = model.predict_one(
            get_feature_vector_from_mfcc(temporaryFileName, flatten=to_flatten))
        print('section:',previousEndTime,'-',endTime,'(ms). predicted value:', predicted_value, '[',class_labels[predicted_value], ']')
        os.remove(temporaryFileName)
        previousEndTime = endTime

    

def lstm_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    #train the model
#    print('Starting LSTM')
#    model = LSTM(input_shape=x_train[0].shape,
#                 num_classes=num_labels)
#    print('x shape', x_train[0].shape)
#    print('num labels',num_labels)
#
#    model.train(x_train, y_train, x_test, y_test_train, n_epochs=50)
#    model.evaluate(x_test, y_test)
#    model.save_model()


    newmodel = LSTM(input_shape=x_train[0].shape,num_classes=num_labels)
    newmodel.load_model(to_load="./LSTM_best_model.h5")
    newmodel.train(x_train, y_train, x_test, y_test_train, n_epochs=0)

    audio_file = "../dataset/union-interview.wav"
    
    section_by_section_analysis(model=newmodel, audio_file=audio_file, to_flatten=to_flatten)

if __name__ == '__main__':
    lstm_example()
