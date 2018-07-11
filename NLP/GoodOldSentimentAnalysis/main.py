from torchnlp.datasets import imdb_dataset
from text import text_to_word_sequence
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm
from sequence import pad_sequences
from sklearn.utils import shuffle
import torch.nn as nn
import torch
import numpy as np

init = False

# Hyperparameters
SEQUENCE_LENGTH = 250
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EMBEDDING_SIZE = 100 # input_size
NUM_EPOCHS = 10
NUM_CLASSES = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.003

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if init == True:
    sentiment = {
        'pos' : 1,
        'neg' : 0,
    }

    train_texts = [text_to_word_sequence(data['text']) for data in  tqdm(imdb_dataset(train=True)) ]
    train_labels = [ sentiment[data['sentiment']] for data in imdb_dataset(train=True)]


    test_texts = [ text_to_word_sequence(data['text']) for data in tqdm(imdb_dataset(test=True))]
    test_labels = [ sentiment[data['sentiment']] for data in imdb_dataset(test=True) ]

# test = imdb_dataset(test=True)

    all_texts = np.concatenate(( train_texts, test_texts )).tolist()

    vocabulary = Dictionary(documents=all_texts)
    vocabulary.save('imdb_vocabulary')

    train_x = np.asarray([ np.asarray( vocabulary.doc2idx(doc), dtype=np.int32 ) + 1 for doc in tqdm(train_texts) ])
    train_y = np.asarray(train_labels, dtype=np.int32)

    test_x = np.asarray([ np.asarray( vocabulary.doc2idx(doc), dtype=np.int32 ) + 1  for doc in tqdm(test_texts) ])
    test_y = np.asarray( test_labels, dtype=np.int32)

    np.save('train_x.npy', train_x)
    np.save('train_y', train_y)
    np.save('test_x.npy', test_x)
    np.save('test_y.npy', test_y)


else:
    vocabulary = Dictionary.load('imdb_vocabulary')
    train_x = np.load('train_x.npy')
    train_y = np.load('train_y.npy')
    test_x = np.load('test_x.npy')
    test_y = np.load('test_y.npy')

X_train = pad_sequences(train_x, maxlen=250)
X_test = pad_sequences(test_x, maxlen=250)

x_train, y_train = shuffle(X_train, train_y, random_state=42)
x_test = X_test
y_test = test_y

x_train = torch.from_numpy(x_train).type(torch.long)
y_train = torch.from_numpy(y_train).type(torch.long)
x_test = torch.from_numpy(x_test).type(torch.long)
y_test = torch.from_numpy(y_test).type(torch.long)


assert x_train.shape[0] == y_train.shape[0]
assert y_train.shape[0] == y_test.shape[0]

def get_data(x, y, batch_size=64):
    data_size = x.shape[0]
    while True:
        for i in range(0, data_size, batch_size):
            yield(
                x[ i : min( i + batch_size, data_size )  ],
                y[i : min(i + batch_size, data_size)]
            )


train_generator = get_data(x_train, y_train, batch_size=64)


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        self.embedding = nn.Embedding(len(vocabulary) + 1, embedding_dim=EMBEDDING_SIZE, padding_idx=0)
    
    def forward(self, x):
        # Apply embedding
        x = self.embedding(x)

        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN(EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

# print(model( torch.from_numpy( train_x[:20]).to(device)  ))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

# Train the model
total_step = x_train.shape[0] // BATCH_SIZE

# for epoch in tqdm(range(NUM_EPOCHS)):
#     for i, step in enumerate(range(total_step)):
#         texts, labels = next(train_generator)

#         texts = texts.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(texts)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                 .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))


# torch.save(model, 'imdbmodel')

model = torch.load('imdbmodel')

# print(model)

TEST_BATCH_SIZE = 2000

test_generator = get_data(x_test, y_test, batch_size=TEST_BATCH_SIZE)

test_step = x_test.shape[0] // TEST_BATCH_SIZE 

with torch.no_grad():
    correct = 0
    total = 0

    for step in range(test_step):
        texts, labels = next(test_generator)

        texts = texts.to(device)
        labels = labels.to(device)

        outputs = model(texts)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test texts: {} %'.format( texts.shape[0], 100 * correct / total)) 


    # for images, labels in test_loader:
    #     images = images.reshape(-1, sequence_length, input_size).to(device)
    #     labels = labels.to(device)
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 
