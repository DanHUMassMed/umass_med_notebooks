from time import time
import platform
import math
import os 

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.utils.data as utils_data
from sklearn.model_selection import KFold

from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

BASE_DIR='./output_classical_life_stage'


# Define your dataset for learning
class LearningDataset(utils_data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        x_item = self.x_data[index]
        y_item = self.y_data[index]
        return x_item, y_item

    def __len__(self):
        return len(self.x_data)

def get_dataloaders(x_train, y_train, x_validation, y_validation, 
                    train_batches = 1, train_shuffle = False,
                    validation_batches = 1, validation_shuffle = False):
    
        # Define your DataLoader for the training set
        train_batch_size = len(x_train) // train_batches
        train_dataset = LearningDataset(x_train, y_train)    
        train_loader = utils_data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=train_shuffle
        )

        # Define your DataLoader for the validation set
        validation_batch_size = len(x_validation) // validation_batches
        validation_dataset = LearningDataset(x_validation, y_validation)
        validation_loader = utils_data.DataLoader(
            validation_dataset, batch_size=validation_batch_size, shuffle=validation_shuffle
        )
        return train_loader, validation_loader


def train_validate(model, criterion, optimizer, train_loader, validation_loader, n_epochs=200):
    training_loss_lst = []
    training_accuracy_lst = []
    validation_loss_lst = []
    validation_accuracy_lst = []
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    
    for epoch in range(n_epochs):
        sum_train_loss = 0.0
        sum_train_accuracy = 0.0
        sum_validation_loss = 0.0
        sum_validation_accuracy = 0.0

        # Training loop
        model.train()
        for train_batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            train_prediction = model(x_train)
            train_loss = criterion(train_prediction, y_train)
            train_loss.backward()
            optimizer.step()
            sum_train_loss += train_loss.item()
            sum_train_accuracy += (torch.argmax(train_prediction, 1) == torch.argmax(y_train, 1)).float().mean()

        # Validation loop
        model.eval()
        with torch.no_grad():
            for validation_batch_idx, (x_validation, y_validation) in enumerate(validation_loader):
                x_validation = x_validation.to(device)
                y_validation = y_validation.to(device)
                validation_prediction = model(x_validation)
                validation_loss = criterion(validation_prediction, y_validation)
                sum_validation_loss += validation_loss.item()
                sum_validation_accuracy += (torch.argmax(validation_prediction, 1) == torch.argmax(y_validation, 1)).float().mean()
                
        
        modulus = 1 if (n_epochs // 10) == 0 else (n_epochs // 10)
        if epoch % modulus == 0:
            training_loss_lst.append(sum_train_loss / (train_batch_idx+1))
            sum_train_accuracy = sum_train_accuracy.cpu().detach().numpy()
            training_accuracy_lst.append(sum_train_accuracy / (train_batch_idx+1))
            validation_loss_lst.append(sum_validation_loss / (validation_batch_idx+1))
            sum_validation_accuracy = sum_validation_accuracy.cpu().detach().numpy()
            validation_accuracy_lst.append(sum_validation_accuracy / (validation_batch_idx+1))
            print(f'Epoch1 {epoch + 1}: Train Loss = {sum_train_loss}, Val Loss = {sum_validation_loss}, Val Accuracy = {sum_validation_accuracy}')
            

    
    model = model.to("cpu")    
            
    return training_loss_lst, training_accuracy_lst, validation_loss_lst, validation_accuracy_lst


def cross_validation_training(x_data, y_data, model, criterion, optimizer, k=5, shuffle=False, n_epochs=200):
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    
    # Set up cross-validation
    kf = KFold(n_splits=k, shuffle=shuffle)
    # Train and evaluate the model for each fold
    for fold, (train_indices, validation_indices) in enumerate(kf.split(x_data)):
        print(f'{fold=}')
        # Split the data into training and validation sets for this fold
        x_train = x_data[train_indices]
        y_train = y_data[train_indices]
        x_validation = x_data[validation_indices]
        y_validation = y_data[validation_indices]

        train_loader, validation_loader = get_dataloaders(
                    x_train, y_train, x_validation, y_validation, 
                    train_shuffle = True,
                    validation_shuffle = True)
        
        scores = train_validate(model, criterion, optimizer, train_loader, validation_loader, n_epochs)
        training_loss_lst, training_accuracy_lst, validation_loss_lst, validation_accuracy_lst = scores
        
        # Add the training and validation loss values to the lists
        train_losses.append(training_loss_lst[:-1][0])
        train_accuracies.append(training_accuracy_lst[:-1][0])
        validation_losses.append(validation_loss_lst[:-1][0])
        validation_accuracies.append(validation_accuracy_lst[:-1][0])
    return train_losses, train_accuracies, validation_losses, validation_accuracies

def plot_learning_curve(train_sizes, train_scores_lst, test_scores_lst, y_title="Score",y_as_percentage=True):
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("% of training examples")
    plt.ylabel(y_title)
    
    train_scores = np.array(train_scores_lst)
    test_scores = np.array(test_scores_lst)
    
    if y_as_percentage:
        print("WE ARE HERE")
        plt.ylim(top=110)
        train_scores = train_scores * 100
        test_scores = test_scores * 100
        
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, '^--', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt, train_scores_mean, test_scores_mean
        

def plot_learning_curve2(train_loss, validation_loss):
    # Plot the learning curve
    plt.plot(train_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def confusion_matrix_plot(fig, estimator, X, y, file_prefix,target_names):
    index = file_prefix.rindex("/")
    title = file_prefix[index + 1:].replace("_"," ").title()
    ax = fig.add_subplot(111)

    disp = plot_confusion_matrix(estimator, X, y,
                                 display_labels=sorted(target_names),
                                 cmap=plt.cm.Blues,
                                 #normalize='pred',
                                 values_format='n',
                                 ax=ax)
    disp.ax_.set_title("Confusion Matrix for\n{}".format(title))
    plt.savefig('{}_cm.png'.format(file_prefix))
    plt.clf()

    return plt

# Get a percentage of the data to train on
def get_percent_of_data(x_data, y_data, percentage):
    num_elements = int(len(x_data) * (percentage/100))
    randomstate = np.random.default_rng( 0 )
    element_indexes = randomstate.choice(len(x_data), num_elements, replace=False, shuffle=False )
    return x_data[element_indexes], y_data[element_indexes]


def create_results_df(model,x_data,y_data, one_hot_encoder):
    total_correct=0

    position_lst = []
    predicted_lst = []
    actual_lst = []
    predicted_decoded_lst = []
    actual_decoded_lst = []
    for i in range(len(x_data)-1):
        x_batch = x_data[i:i+1]
        predicted = model(x_batch)
        predicted_decoded = one_hot_encoder.inverse_transform(predicted.detach().numpy()) 
        actual_decoded = one_hot_encoder.inverse_transform(y_data[i:i+1].detach().numpy()) 
        #print(f'{X_batch} {predicted} {y_batch[i:i+1]}')
        p_a = torch.argmax(predicted, 1)[0]
        t_a = torch.argmax(y_data[i:i+1],1)[0]

        position_lst.append(i)
        predicted_lst.append(p_a.item())
        actual_lst.append(t_a.item())

        predicted_decoded_lst.append(predicted_decoded)
        actual_decoded_lst.append(actual_decoded)
        #print(f'{i:<3} {p_a} {t_a} {p_a==t_a}')
        if p_a==t_a:
            total_correct +=1
    print(f'Score {total_correct/len(x_data)}')
    results = pd.DataFrame({'position':position_lst,'predicted_cd':predicted_decoded_lst,
                            'actual_cd':actual_decoded_lst,'predicted':predicted_lst,'actual':actual_lst})
    return results

