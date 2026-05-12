# Developing a Neural Network Regression Model 

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
A company has collected a dataset containing various input features and corresponding numerical output values related to a specific problem (such as sales, price, or demand prediction). The relationship between the input variables and output is complex and cannot be accurately modeled using simple statistical methods.

To solve this problem, the company wants to develop a neural network-based regression model that can learn patterns from the existing data. The model should be trained using historical data so that it can understand how input features influence the output values.

Once trained, the model will be used to predict continuous numerical outputs for new, unseen data points. This will help the company make better decisions based on accurate predictions.

The goal is to minimize prediction error and improve the model’s performance using appropriate training techniques such as backpropagation and optimization algorithms

## Neural Network Model
<img width="1126" height="856" alt="585527108-0b9368da-34ee-4f20-aa49-5de57c9e1793" src="https://github.com/user-attachments/assets/0ef3745c-20fa-4d6d-9a19-5df9b20d9fd4" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

### Name: T H KARTHIK KRISHNA

### Register Number: 212223240067

## PROGRAM 
```
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/drive/MyDrive/DL EXP-1 - Sheet1.csv')
X = dataset1[['input']].values
y = dataset1[['output']].values

dataset1.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name: AJITH A
# Register Number: 212224230012
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)

# Name: AJITH A
# Register Number: 212224230012
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
# train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
# This cell was commented out to avoid the NameError until the function is defined.

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

### Dataset Information
<img width="262" height="415" alt="585528916-ab6ec503-f9d5-4931-bbb6-c27a43dc9520" src="https://github.com/user-attachments/assets/7d4282e2-2119-4e8d-8258-8f558f9d9b3a" />



### OUTPUT
<img width="202" height="245" alt="585528176-b4da268c-9f89-49a0-b7f4-2e713686286b" src="https://github.com/user-attachments/assets/48d700d8-d9c1-4167-b433-2c88979d4534" />
<img width="334" height="192" alt="585529794-1c4a06fe-67b8-4ebb-ae1d-b4ed38405aec" src="https://github.com/user-attachments/assets/3617ffb1-01ba-4614-95b9-d82f879cf351" />
<img width="263" height="36" alt="585529746-9581a18a-2130-4348-8612-d0c5c30625ab" src="https://github.com/user-attachments/assets/aca6dec9-8e34-4ca3-8483-a85606e76944" />



### Training Loss Vs Iteration Plot
<img width="689" height="469" alt="585530876-c09b96a3-bbb7-4b17-82d5-9afca3f32e0d" src="https://github.com/user-attachments/assets/b7b0f27a-990c-49b1-acda-f8dde0ab4d90" />



### New Sample Data Prediction

<img width="344" height="33" alt="585531055-ac2e2235-2987-417f-a00a-daf936a9948d" src="https://github.com/user-attachments/assets/406dac93-8e61-4966-820e-650095fb87f5" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.


