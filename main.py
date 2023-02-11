import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
from tkinter import *


################## global dataset
excel_data_df = pd.read_csv(r'penguins.csv')
feature = excel_data_df.columns.tolist()
species = excel_data_df['species'].tolist()
first = excel_data_df.iloc[0:50]
second = excel_data_df.iloc[50:100]
third=  excel_data_df.iloc[100:150]
################## global data set

#################  read data from files
def read_file():
    data_df = pd.read_csv('penguins.csv')
    ################ class1 classify
    class_1 = data_df.iloc[0:50]
    class_1 = class_1.apply(lambda x: x.fillna(x.value_counts().index[0]))
    ################ class1 classify
    ################ class2 classify
    class_2 = data_df.iloc[50:100]
    class_2 = class_2.apply(lambda x: x.fillna(x.value_counts().index[0]))
    ################ class2 classify
    ################ class3 classify
    class_3 = data_df.iloc[100:150]
    class_3 = class_3.apply(lambda x: x.fillna(x.value_counts().index[0]))
    ################ class3 classify
    data = pd.concat([class_1, class_2, class_3], axis=0)
    ############# data preprocessing
    le = LabelEncoder()
    gender_encoded = le.fit_transform(data['gender'])
    data['gender'] = gender_encoded
    ############# data preprocessing
    ############# division data
    speciesCol = data[['species']]
    data = data.drop(columns=['species'])
    for column in data.columns:
        data[column] = data[column] / data[column].abs().max()
    data = pd.concat([speciesCol, data],axis=1)
    ############# division data
    return data
#################  read data from files

################ data filtering
#################### data filter for graph 1
def data_filter(classes, features):
    data = read_file()
    class_index = data[(data['species'] != classes[0]) & (data['species'] != classes[1])].index
    data.drop(class_index, inplace=True)
    coulmns = data.columns.tolist()
    coulmns.remove('species')
    ################## feature selection
    coulmns.remove(features[0])
    coulmns.remove(features[1])
    ################## feature selection
    data.drop(labels=coulmns, inplace=True, axis=1)
    le = LabelEncoder()
    species_encoded = le.fit_transform(data['species'])
    data['species'] = species_encoded
    ################## division classes
    first_class = data.iloc[0:50]
    second_class = data.iloc[50:100]
    ################## division classes
    return first_class, second_class
#################### data filter for graph 1

################ data filtering

########################### data split
def data_split(classes, features, learn, epo, bias):
    first_class, second_class = data_filter(classes, features)
    ##################### class selection
    class1train, class1test = train_test_split(first_class, test_size=0.4)
    class2train, class2test = train_test_split(second_class, test_size=0.4)
    ##################### class selection
    #################### trian & test data
    train = class1train.append(class2train)
    train['species'] = train['species'].replace([0], -1)
    test = class1test.append(class2test)
    test['species'] = test['species'].replace([0], -1)
    train = train.sample(frac=1)
    test = test.sample(frac=1)
    #################### trian & test data
    #################### feature train and test
    x_trian = train.drop(['species'], axis=1)
    x_test = test.drop(['species'], axis=1)
    #################### feature train and test
    weights = singleLayerPrecept_train(x_trian, train['species'], learn, epo, bias,first_class,second_class,features)
    result_test = singleLayerPrecepr_test(x_test, test['species'].shape, weights)
    confusion_matrix(result_test, test['species'].tolist())
########################### data split



############## algorithm
###################################################  single layer algorithm train
def singleLayerPrecept_train(input_data, actualData, learningRate, epochs, bias,first_class,second_class,features):
    weights = np.random.randn(2)
    x_features = np.array(input_data)
    actualData = np.array(actualData)
    y_predict = -1
    ############################################# bias check box
    if bias == 1:
        bias = np.random.random_sample()
    else:
        bias = 0
    ############################################# bias check box
    for e in range(epochs):  # outer loop
        for i in range(60):  # inner loop
            rand_index = np.random.randint(0, 59) # index random
            v = np.matmul(np.transpose(weights), x_features[rand_index]) # net value
            v = v + bias # net value
            ############# signum activation func
            condition2 = (v >= 0)
            condition1 = (v < 0)
            if condition1:
                y_predict = -1
            elif condition2:
                y_predict = 1
            ############# signum activation func
            ##################  weights update
            error = actualData[rand_index] - y_predict
            if error != 0:
                weights = weights + learningRate * (error) * x_features[rand_index]
            ##################  weights update
    #########################figure
    figure(features, first_class, second_class,bias,weights)
    #########################figure
    return weights


###################################################  single layer algorithm train

##################################   single layer algorithm test
def singleLayerPrecepr_test(input_data, label_shape, weights):
    x_feature = np.array(input_data)
    y = []
    for i in range(40):
        y.append(0)
    ########## net value
    for index in range(40):
        net = np.matmul(np.transpose(weights), x_feature[index])
        ############## condition part
        condition1 = (net < 0)
        condition2 = (net >= 0)
        if condition1:
            y[index] = -1
        elif condition2:
            y[index] = 1
        ############## condition part
    return y


##################################   single layer algorithm test
############## algorithm

########### confusion matrix
def confusion_matrix(result, labels):
    ########### variables part
    true_possitive = 0
    true_negative = 0
    false_negative = 0
    false_possitive = 0
    ########### variables part
    ########### for loop part
    for i in range(40):
        ################ condition part
        ############ true possitive
        if (result[i] == 1) and (labels[i] == 1):
            true_possitive += 1
        # true positive
        # true negative
        elif (result[i] == -1) and (labels[i] == -1):
            true_negative += 1
        # true negative
        # false positive
        elif (result[i] == 1) and (labels[i] == -1):
            false_possitive += 1
        # false positive
        # false negative
        elif (result[i] == -1) and (labels[i] == 1):
            false_negative += 1
        # false negative
        ################ condition part
    accuracy = ((true_possitive + true_negative) / (40))*100
    ########### for loop part
    #################### print part
    print('######################\n',
          'SLP CONFUSION MATRIX\n',
          'true_possitive : ' + str(true_possitive),
          '\ntrue_negative : ' + str(true_negative),
          '\nfalse_negative : ' + str(false_negative),
          '\nfalse_possitive : ' + str(false_possitive),
          '\naccuracy : ' + str(accuracy),
          '\n###################### \n')
    #################### print part

########### confusion matrix

########################## figures
############## figure1
def figure(features, firstClassDF, secondClassDF,bias,weights):
    print(features)
    plt.figure(figsize=(12, 6))
    plt.title('Graph')
    plt.cla()
    # class 1
    plt.scatter(firstClassDF[features[0]], firstClassDF[features[1]], color='red')
    # class 2
    plt.scatter(secondClassDF[features[0]], secondClassDF[features[1]], color='blue')
    # features name
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    # relation between two classes
    data_con=pd.concat([firstClassDF,secondClassDF])
    x = np.linspace(data_con.loc[:, features[0]].min(), data_con.loc[:, features[0]].max(), 4000)
    y = (-((weights[0] * x) + bias)) / weights[1]
    plt.plot(x, y, '-r', linestyle='solid')
    plt.show()
############## figure1
################ figure2
def graph2():
    ################### feature inputs
    print(garphF1Value.get())
    print(garphF2Value.get())
    # input data
    input1=garphF1Value.get()
    input2=garphF2Value.get()
    # input data
    print("Selected features:" , garphF2Value , garphF1Value)
    # array of input data
    arr = []
    arr.append(input1)
    arr.append(input2)
    # array of input data
    plt2.figure(figsize=(12, 6))
    plt2.title('Graph 2')
    # class 1
    plt2.scatter(first[arr[0]], first[arr[1]], color='red')
    # class 2
    plt2.scatter(second[arr[0]], second[arr[1]], color='green')
    # class 3
    plt2.scatter(third[arr[0]], third[arr[1]], color='blue')
    # features name
    plt2.xlabel(input1)
    plt2.ylabel(input2)
    plt2.show()
    ################### feature inputs
################ figure2
########################## figures


def get_data():
    ###############vars
    selected_classes = []
    selected_features = []
    ###############vars
    ##################### learning rate
    print(learningRateTB_entry.get())
    learning_rate = float(learningRateTB_entry.get())
    print("Learning rate:", learning_rate)
    ##################### learning rate
    ##################### bias
    print(epotxt_entry.get())
    epochs = int(epotxt_entry.get())
    print("Epochs:", epochs)
    ##################### bias
    ###################class inputs
    print(class1Value.get())
    print(class2Value.get())
    selected_classes.append(class1Value.get())
    selected_classes.append(class2Value.get())
    print("Selected classes:", selected_classes)
    ###################class inputs
    ################### feature inputs
    print(feature1Value.get())
    print(feature2Value.get())
    selected_features.append(feature1Value.get())
    selected_features.append(feature2Value.get())
    print("Selected features:", selected_features)
    ################### feature inputs
    #########################bias
    print(biasValue.get())
    if biasValue.get() == 'yes':
        bias = 1
    elif biasValue.get() == 'no':
        bias = 0
    print("Bias:", bias)
    #########################bias

    #########################filtering data
    data_split(selected_classes, selected_features, learning_rate, epochs, bias)
    #########################filtering data





if __name__ == "__main__":
    #######GUI
    ########### master window
    master = Tk()
    master.geometry('500x500')  # Size of the window
    master.title("Task1 Form")
    ########### master window

    ########################## learning rate
    # ----------learning rate label----------
    learningRateLB = Label(master, text="learning rate :", font=('Times New Roman', '12'))
    learningRateLB.place(x=10, y=10)
    # ----------learning rate input----------
    learningRateTB = StringVar()
    learningRateTB_entry = Entry(master, textvariable=learningRateTB, width="20")
    learningRateTB_entry.place(x=140, y=15)
    ########################## learning rate

    ########################## epochs
    # ----------number of epochs label----------
    epo = Label(master, text="Epochs num :", font=('Times New Roman', '12'))
    epo.place(x=10, y=45)
    # ----------number of epochs input---------
    epotxt = StringVar()
    epotxt_entry = Entry(master, textvariable=epotxt, width="20")
    epotxt_entry.place(x=140, y=50)
    ########################## epochs

    ########################## class1
    # ----------class1 label---------
    class1 = Label(master, text="class1 :", font=('Times New Roman', '12'))
    class1.place(x=10, y=80)
    # ----------class1 input---------
    sp = set(species)
    sp = list(sp)
    class1Value = StringVar()
    class1Value.set(sp[0])
    class1DropMenu = OptionMenu(master, class1Value, *sp).place(x=100, y=80)
    ########################## class1

    ########################## class2
    # ----------class2 label---------
    class2 = Label(master, text="class2 :", font=('Times New Roman', '12'))
    class2.place(x=10, y=120)

    # ----------class2 input---------
    class2Value = StringVar()
    class2Value.set(sp[1])
    class2DropMenu = OptionMenu(master, class2Value, *sp).place(x=100, y=120)
    ########################## class2

    ########################## feature1
    # ----------feature1 label---------
    feature1 = Label(master, text="feature1 :", font=('Times New Roman', '12'))
    feature1.place(x=260, y=80)
    # ----------feature1 input---------
    feature.remove('species')
    feature1Value = StringVar()
    feature1Value.set(feature[0])
    feature1DropMenu = OptionMenu(master, feature1Value, *feature).place(x=360, y=80)
    ########################## feature1

    ########################## feature2
    # ----------feature2 label---------
    feature2 = Label(master, text="feature2 :", font=('Times New Roman', '12'))
    feature2.place(x=260, y=120)
    # ----------feature2 input---------
    feature2Value = StringVar()
    feature2Value.set(feature[1])
    feature2DropMenu = OptionMenu(master, feature2Value, *feature).place(x=360, y=120)
    ########################## feature2

    ########################## BIAS
    # -----------bias label-------------
    bias = Label(master, text="bias :", font=('Times New Roman', '12'))
    bias.place(x=10, y=165)
    # -----------bias input-------------
    bias_arr = ('yes', 'no')
    biasValue = StringVar()
    biasDropMenu = OptionMenu(master, biasValue, *bias_arr).place(x=100, y=160)
    ########################## BIAS
    # ----------Submit button----------
    Submit = Button(master, text="Submit", width="20", command=get_data)
    Submit.place(x=150, y=200)

    # ----------label---------
    feature1 = Label(master, text="---------------- choose 2 feature for the second graph -------------------", font=('Times New Roman', '12'))
    feature1.place(x=10, y=250)
    # ----------input---------
    ########################## feature1
    # ----------feature1 label---------
    feature1 = Label(master, text="feature1 :", font=('Times New Roman', '12'))
    feature1.place(x=10, y=300)
    # ----------feature1 input---------
    garphF1Value = StringVar()
    garphF1Value.set(feature[0])
    garphF1DropMenu = OptionMenu(master, garphF1Value, *feature).place(x=80, y=295)
    ########################## feature1

    ########################## feature2
    # ----------feature2 label---------
    feature2 = Label(master, text="feature2 :", font=('Times New Roman', '12'))
    feature2.place(x=260, y=300)
    # ----------feature2 input---------
    garphF2Value = StringVar()
    garphF2Value.set(feature[1])
    garphF2DropMenu = OptionMenu(master, garphF2Value, *feature).place(x=340, y=295)
    ########################## feature2
    # ----------show button----------
    show = Button(master, text="Show Graph", width="20", command=graph2)
    show.place(x=150, y=350)
    ######## main loop
    master.mainloop()

