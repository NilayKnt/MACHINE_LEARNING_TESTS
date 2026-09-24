

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

df = pd.read_csv("heart.csv")

# Kategorik sütunları One-Hot Encoding ile dönüştürme
categorical_cols = ["cp", "restecg", "slope", "thal", "ca"]
# Not: Veri setindeki sütun isimlerine göre bu listeyi güncelleyebilirsin
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop("target", axis=1).values
y = df["target"].values

SEED=42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
'''X = df.drop("target", axis=1).values
y = df["target"].values'''

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=SEED, stratify=y)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64,activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(8 ,activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])

model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy")])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history=model.fit(X_train,y_train,validation_split=0.2, epochs=1,batch_size=4, verbose=2,callbacks=[early_stopping])

device =torch.device("cuda"if  torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

full_ds = TensorDataset(X_train_tensor,y_train_tensor)
train_ds,val_ds = random_split(full_ds,[int(0.8*len(full_ds)),len(full_ds)-int(0.8*len(full_ds))])

train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=32)

class ANN(nn.Module):
  def __init__ (self,n_features):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_features,16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16,1),
        nn.Dropout(0.2),
    )

  def forward(self,x):
    return self.net(x)

model = ANN(X_train.shape[1]).to(device)
print(model)

criterion = nn.BCEWithLogitsLoss() # sigmoid + binary cross entropy
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def accuracy(logits,targets):
  preds = (torch.sigmoid(logits)>0.5).float()
  return (preds==targets).float().mean()

EPOCHS =150

for epoch in range(1,EPOCHS+1):
  model.train()
  train_loss = 0
  train_acc = 0
  for batch_idx,(data,targets) in enumerate(train_loader):
    data,targets = data.to(device),targets.to(device)
    optimizer.zero_grad()
    logits = model(data)
    loss= criterion(logits,targets)
    loss.backward()
    optimizer.step()
    train_loss+=loss.item()*data.size(0)
    train_acc+=accuracy(logits,targets)

model.eval()
val_loss,val_correct = 0.0,0.0
with torch.no_grad():
  for data,targets in val_loader:
    data,targets = data.to(device),targets.to(device)
    logits = model(data)
    val_loss += criterion(logits,targets).item()*data.size(0)
    val_correct += accuracy(logits,targets)*data.size(0)
val_loss /= len(val_loader.dataset)
val_acc =val_correct/len(val_loader.dataset)
print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

model.eval()
with torch.no_grad():
  test_logits = model(X_test_tensor.to(device))
  test_loss = criterion(test_logits,y_test_tensor.to(device))
  test_acc = accuracy(test_logits,y_test_tensor.to(device))
print(f"\n Test kaybı: {test_loss:.4f} | Test doğruluğu {test_acc:.4f}")
