
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from random import shuffle
from time import time

def LoadData ( ) :
    global TRAIN_DATA, TEST_DATA, SEQ_LENGTH, ID_TO_EMOTION, EMOTION_TO_ID
    global ID_TO_WORD, WORD_TO_ID, VOCAB

    with open('DATA/TRAIN.json', 'r') as file :
        TRAIN_DATA = json.load(file)
    with open('DATA/TEST.json', 'r') as file :
        TEST_DATA = json.load(file)

    SEQ_LENGTH = 0
    for val in TRAIN_DATA.values() :
        for seq in val :
            if len(seq) > SEQ_LENGTH :
                SEQ_LENGTH = len(seq)
    for val in TEST_DATA.values() :
        for seq in val :
            if len(seq) > SEQ_LENGTH :
                SEQ_LENGTH = len(seq)
    
    ID_TO_EMOTION = dict(enumerate(list(TRAIN_DATA.keys())))
    EMOTION_TO_ID = { v : k for k , v in ID_TO_EMOTION.items() }

    VOCAB = set()
    for sentences in TRAIN_DATA.values() :
        for sentence in sentences :
            VOCAB.update(sentence)
    for sentences in TEST_DATA.values() :
        for sentence in sentences :
            VOCAB.update(sentence)

    ID_TO_WORD = list(VOCAB)
    ID_TO_WORD.sort()
    ID_TO_WORD.insert(0, '')
    WORD_TO_ID = { word : idd for idd, word in enumerate(ID_TO_WORD) }

def GetSequenceVector ( sequence ) :
    seq_vector = list()
    for word in sequence :
        seq_vector.append( WORD_TO_ID[word] )
    padding = SEQ_LENGTH - len(sequence)
    seq_vector.extend( [0] * padding )
    return seq_vector

def GenerateSequenceLabelPairs ( data ) :
    inputs = list()
    labels = list()
    for emotion, sequences in data.items() :
        emotion_id = EMOTION_TO_ID[emotion]
        for sequence in sequences :
            seq_vector = GetSequenceVector(sequence)
            inputs.append(seq_vector)
            labels.append(emotion_id)
    return list(zip(inputs, labels))

def MakeBatches ( batch_size , dataset ) :
    shuffle(dataset)
    batches = list()
    for start in range(0, len(dataset), batch_size) :
        end = start + batch_size
        batch, labels = list(zip(*dataset[start:end]))
        labels = torch.Tensor(labels).long()
        batch = torch.Tensor(batch).long()
        batches.append((batch, labels))
    return batches

def SaveProgressAndModel ( loss_progress , acc_progress , folder = 'progress' ) :
    file = open(folder + '/LP', 'rb')
    x = pickle.load(file)
    file.close()
    file = open(folder + '/LP', 'wb')
    pickle.dump(x + loss_progress, file)
    file.close()
    file = open(folder + '/ACC', 'rb')
    x = pickle.load(file)
    file.close()
    file = open(folder + '/ACC', 'wb')
    pickle.dump(x + acc_progress, file)
    file.close()
    torch.save(MODEL, folder + '/MODEL')
    
def ClearProgress ( folder = 'progress' ) :
    file = open(folder + '/LP', 'wb')
    pickle.dump([], file)
    file.close()
    file = open(folder + '/ACC', 'wb')
    pickle.dump([], file)
    file.close()
    
def LoadModel ( folder = 'progress' ) :
    global MODEL
    MODEL = torch.load(folder + '/MODEL')

class EmotionClassifier ( nn.Module ) :
    def __init__ ( self , input_size, hidden_size, output_size, num_layers ) :
        super(EmotionClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(len(VOCAB), input_size)
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                            num_layers = num_layers, dropout = 0.5, batch_first = True)
        self.dense = nn.Linear(hidden_size, output_size)
    
    def forward ( self , inputs ) :
        orig_inputs = inputs
        inputs = self.embed(inputs)
        
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size)
        output, _ = self.lstm(inputs, (h0, c0))
        
        batch_size = inputs.shape[0]
        masking_indices = [ np.argmax(orig_inputs[x] == 0) - 1 for x in range(batch_size) ]
        output = torch.stack( [ output[idd, k, :] for idd, k in enumerate(masking_indices) ] )
        
        categs = self.dense(output)
        return categs

def TrainModel ( batch_size , total_epochs , dataset ) :
    L = len(dataset)
    loss_progress, acc_progress = [], []

    for epoch in range(total_epochs) :
        print('\n EPOCH {} STARTED '.format(epoch+1))
        total_loss = 0.0
        total_correct = 0
        batches = MakeBatches(batch_size, dataset)
        epoch_start = time()
        
        for step, (batch, labels) in enumerate(batches) :
            step_start = time()
            label_dists = MODEL(batch)
            loss = criterion(label_dists, labels)
            
            loss_progress.append(loss.item())
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = torch.argmax(label_dists, dim = 1)
            corr_count = (pred == labels).sum()
            total_correct += corr_count
            acc = corr_count / batch.shape[0]
            
            acc_progress.append(acc.item())
            ti = time() - step_start
            print('    STEP : {:3d} | LOSS : {:.6f} | ACCURACY : {:.4f} | DUR : {:.4f}'.format(step+1, loss, acc, ti))
        
        acc = total_correct / L
        loss = total_loss / len(batches)
        ti = time() - epoch_start
        print(' EPOCH LOSS : {:.6f} | ACCURACY : {:.4f} | DUR : {:.4f}'.format(loss, acc, ti))

    SaveProgressAndModel(loss_progress, acc_progress)

def GetAccuracyAndConfusionMatrix ( dataset ) :
    total_correct = 0
    total = 0
    batches = MakeBatches(1, dataset)
    N = len(batches)
    n = len(ID_TO_EMOTION)
    conf_mat = np.zeros((n, n))
    
    for step, (batch, label) in enumerate(batches) :
        try : label_dists = MODEL(batch)
        except :
            print('{:3d} / {:3d} : FAULT IN THE DATASET FORMAT'.format(step+1, N))
            continue
        
        pred = torch.argmax(label_dists, dim = 1).item()
        
        conf_mat[label][pred] += 1
        corr_count = (pred == label) * 1
        total_correct += corr_count
        total += len(batch)
        
    acc = total_correct / total
    return acc, conf_mat

def PredictEmotion ( sentence ) :
    sentence = sentence.lower()
    permitted_syms = set('abcdefghijklmnopqrstuvwxyz ')
    invalid_syms = set(sentence) - permitted_syms
    for sym in invalid_syms :
        sentence = sentence.replace(sym, ' ')
    
    words = sentence.split(' ')
    words = [ w for w in words if w != '' ]
    for word in words :
        if not word in ID_TO_WORD :
            print('[SENTENCE CONTAINS A WORD <{}> NOT PRESENT IN THE VOCABULARY]'.format(word))
            return
    
    encoded = []
    for word in words :
        encoded.append(WORD_TO_ID[word])
    
    label_dists = MODEL(torch.tensor([encoded]).long())
    pred = torch.argmax(label_dists, dim = 1).item()
    return ID_TO_EMOTION[pred]

if __name__ == '__main__' :
    INPUT_LENGTH = 128
    BATCH_SIZE = 64
    TOTAL_EPOCHS = 40

    LEARNING_RATE = 0.01
    # learning rate decay scheme used while training :-
    #   learning rate for 10 epochs -- 0.01
    #   learning rate after 10 epochs -- 0.001
    #   learning rate after 17 epochs -- 0.0001
    #   learning rate after 31 epochs -- 0.00001
    
    LoadData()
    TRAIN_DATASET = GenerateSequenceLabelPairs(TRAIN_DATA)
    TEST_DATASET = GenerateSequenceLabelPairs(TEST_DATA)
    
    MODEL = EmotionClassifier(INPUT_LENGTH, 256, len(ID_TO_EMOTION), 2)
    # LoadModel() # to resume training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(MODEL.parameters(), lr = LEARNING_RATE)
    
    MODEL.train()
    TrainModel(BATCH_SIZE, TOTAL_EPOCHS, TRAIN_DATASET)
    
    MODEL.eval()
    # GetAccuracyAndConfusionMatrix(TRAIN_DATASET)
    # GetAccuracyAndConfusionMatrix(TEST_DATASET)
    # PredictEmotion('What is the date today?')

