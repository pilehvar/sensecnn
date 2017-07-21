from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
import prepare_dataset as dataset
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import losses
from keras import optimizers
import h5py
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class model():

    params = None

    def __init__(self, args):
        params = dict()
        self.embedding_size = args.embsize
        self.maxlen = args.maxlen
        self.run_id = args.run_id
        if len(self.run_id) > 0:
            self.run_id = "." + self.run_id

        self.dataset = args.dataset
        self.dataset_id = args.dataset_id
        self.num_classes = args.num_classes
        self.static_embeddings =  args.static
        self.use_embeddings = args.emb
        self.emb_path = args.embpath

        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        if not os.path.exists("models/"):
            os.makedirs("models/")

        if not os.path.exists("logs-pred/"):
            os.makedirs("logs-pred/")


        print(params)


    def print_log(self, id, predictions, y_test):
        with open('logs-pred/'+id+self.run_id+'.txt','w') as ofile:
            for i in range(len(predictions)):
                    a = " ".join(format(x, "10.3f") for x in y_test[i])
                    b = " ".join(format(x, "10.3f") for x in predictions[i])
                    ofile.write(a+"\t"+b+"\n")


    def seq_to_matrix(seq, emb_W):
        X = np.zeros((len(seq), emb_W.shape[1]))
        print("Shape of X:", X.shape)
        for i, w in enumerate(seq):
            X[i] = emb_W[w]
        return X


    def get_embeddings(self, path, data_w2index):
        embed = open(path)
        done = 0
        covered = dict()
        W = np.zeros((len(data_w2index)+2, self.embedding_size))
        print("W:",W.shape)
        for line in embed:
            comps = line.split(' ')
            if comps[0] in data_w2index:
                done += 1
                covered[data_w2index[comps[0]]] = comps[0]
                for i in range(self.embedding_size):
                    W[data_w2index[comps[0]], i] = comps[i + 1]


        missed = dict()
        for w in data_w2index:
            k = data_w2index[w]
            if k not in covered:
                missed[k] = w

        print("Missed",len(missed),"words of", len(data_w2index))
        print("The top 100 missing words:\n",sorted(missed.items())[0:100])

        return W


    def evaluate(self):
        settings = { 'dict':'data/'+self.dataset+'/'+ self.dataset_id + '.dict.pkl',
                     'data':'data/'+self.dataset+'/'+ self.dataset_id +'.pkl',
                     'max_features':50000,
                     'filter_length':8,
                     'pool_length':4,
                     'nb_filter':128,
                     'lstm_output_size':32,
                     'batch_size':32,
                     'nb_epoch':20,
                     'folds':5
                    }


        print('Loading data...')

        all_data = dataset.load_data(n_words=settings['max_features'], path=settings['data'],
                       valid_portion=0.0)

        traind, validd, testd = all_data
        print("Train:",len(traind),len(traind[0]),len(traind[1]))
        print("Test:",len(testd), len(testd[0]), len(testd[1]))

        scores = []
        if len(testd[1]) == 0:  # if no test data is dedicated
            print("No test set: running cross-validation.")
            for i in range(settings['folds']):
                print("[Running fold"+str(i+1)+"]")
                # get the data for this fold
                this_fold_data = dataset.fold_data(traind, settings['folds'], i+1)
                this_fold_id = self.dataset + '.' + self.dataset_id + self.run_id + ".p" + str(i+1)

                this_score = self.classify(this_fold_data, settings, this_fold_id)
                scores.append(this_score)
        else:
            this_id = self.dataset + '.' + self.dataset_id + self.run_id
            this_score = self.classify(all_data, settings, this_id)
            scores.append(this_score)

        for score in scores:
            print(score)


    def classify(self, data, settings, log_id):

        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = data

        data_w2index, data_index2w = dataset.get_word_index(path=settings['dict'])
        print("Index size:", len(data_w2index))

        for x in X_train:
            sentence = ''
            for w in x:
                if w == 1:
                    sentence += "oov "
                else:
                    sentence += data_index2w[w]+" "

        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)
        print("Training size:",len(y_train))
        y_train = np_utils.to_categorical(y_train, self.num_classes)
        y_test = np_utils.to_categorical(y_test, self.num_classes)

        print("Test size:", len(y_test))
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)

        print(self.emb_path)

        print('Build model...')

        print("Size of index:",len(data_w2index))

        model = Sequential()

        if self.use_embeddings:
            W = self.get_embeddings(self.emb_path, data_w2index)
            print("Size of W:",W.shape[0],W.shape[1])
            model.add(Embedding(output_dim=W.shape[1], input_dim=W.shape[0], weights=[W], trainable=(not self.static)))
        else:
            model.add(Embedding(settings['max_features'], self.embedding_size, input_length=self.maxlen))

        model.add(Dropout(0.5))
        model.add(Conv1D(filters=settings['nb_filter'],
                                kernel_size=settings['filter_length'],
                                padding='valid',
                                activation='relu',
                                strides=1))
        model.add(MaxPooling1D(pool_size=settings['pool_length']))
        model.add(LSTM(settings['lstm_output_size']))

        model.add(Dense(self.num_classes))
        model.add(Activation('sigmoid'))
        model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adamax(), metrics=['accuracy'])

        log_name = "models/"+log_id+'.hdf5'
        checkpointer = ModelCheckpoint(filepath=log_name, verbose=1, save_best_only=True, monitor='val_acc', mode='max')
        earlystopper = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')

        print('Train...')
        model.fit(X_train, y_train, batch_size=settings['batch_size'], epochs=settings['nb_epoch'],
                  validation_split=0.1, callbacks=[checkpointer,earlystopper])

        model.load_weights(log_name)
        acc = model.evaluate(X_test, y_test, batch_size=settings['batch_size'])

        predictions = model.predict(X_test, batch_size=settings['batch_size'])
        self.print_log(log_id, predictions, y_test)

        print('Test accuracy:', acc[1])
        return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate on document classification task.')

    parser.add_argument('dataset', help='processed dataset name')
    parser.add_argument('dataset_id', help='identifier for the dataset')
    parser.add_argument('num_classes', type=int, help='number of classes')

    parser.add_argument('--run_id', default='', help='identifier of this run')
    parser.add_argument('--emb', type=bool, default=False, help='if initialized with embeddings')
    parser.add_argument('--embpath', help='path to embeddings; must be set if emb is true')
    parser.add_argument('--embsize', metavar='d', default=300, type=int, help='size of input embeddings')
    parser.add_argument('--maxlen', metavar='l', default=1000, type=int, help='maximum length of input sequence')
    parser.add_argument('--static', type=bool, default=False, help='static embeddings')

    args = parser.parse_args()
    new_model = model(args)
    new_model.evaluate()

