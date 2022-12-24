import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
#import globals
from model5 import temp
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)

class Reg:
    def data_preprocessing(self):

        self.data = pd.read_csv('./austin_weather2.csv')
        print("\n",self.data.head())
        # drop or delete the unnecessary columns in the data.
        self.events = self.data[['Events']].replace(' ', 'None')
        self.events = self.events[['Events']].replace('None',0)
        self.events = self.events[['Events']].replace('Thunderstorm', 1)
        self.events = self.events[['Events']].replace(' Thunderstorm', 1)
        self.events = self.events[['Events']].replace('Snow', 2)
        self.events = self.events[['Events']].replace('Rain', 3)
        self.events = self.events[['Events']].replace('Rain ', 3)
        self.events = self.events[['Events']].replace('Fog', 4)
        self.events = self.events[['Events']].replace('Fog ', 4)

        self.data = self.data.drop(['Events','Date', 'SeaLevelPressureHighInches',
                  'SeaLevelPressureLowInches'], axis=1)

        # some values have 'T' which denotes trace rainfall
        # we need to replace all occurrences of T with 0
        # so that we can use the data in our model
        self.data = self.data.replace('T', 0.0)

        # the data also contains '-' which indicates no
        # or NIL. This means that data is not available
        # we need to replace these values as well.
        self.data = self.data.replace('-', 0.0)

        # save the data in a csv file
        self.data.to_csv('./austin_final.csv')

        # read the cleaned data
        self.data = pd.read_csv("./austin_final.csv")


    def Feature_Extraction(self):
        # the features or the 'x' values of the data
        # these columns are used to train the model
        # the last column, i.e, precipitation column
        # will serve as the label

        self.X = self.data

        columns_of_interest = ['TempAvgF','DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH', 'PrecipitationSumInches']
        self.X=self.data[columns_of_interest]

        # the output or the label.
        self.Y = self.events
        # reshaping it into a 2-D vector
        self.Y = self.Y.values.reshape(-1, 1)

        # consider a random day in the dataset
        # we shall plot a graph and observe this
        # day
        self.day_index = 798
        self.days = [i for i in range(self.Y.size)]


    def regression_model(self):
        # initialize a linear regression classifier
        clf = LinearRegression()
        # train the classifier with our
        # input data.
        #self.X = self.X.drop([174,175,176,177,596,597,598,638,639,741,742,953])
        print("lenght of Self.x",len(self.X))
        print("lenght of Self.y",len(self.Y))
        clf.fit(self.X, self.Y)

        # give a sample input to test our model
        # this is a 2-D vector that contains values
        # for each column in the dataset.
        inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45],
                [57], [29.68], [10], [7], [2], [0], [20], [4], [31]])
        inp = inp.reshape(1, -1)

        # print the output.
        new_yhat=np.delete(temp,0)
        new_yhat=new_yhat.reshape(1,-1)
        #yhat.pop(0)
        print('The precipitation in inches for the input is:', clf.predict(new_yhat))

        # plot a graph of the precipitation levels
        # versus the total number of days.
        # one day, which is in red, is
        # tracked here. It has a precipitation
        # of approx. 2 inches.
        print("the precipitation trend graph: ")
        plt.scatter(self.days, self.Y, color='g')
        plt.scatter(self.days[self.day_index], self.Y[self.day_index], color='r')
        plt.title("Precipitation level")
        plt.xlabel("Days")
        plt.ylabel("Precipitation in inches")

        plt.show()
        self.x_vis = self.X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                  'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
                  'WindAvgMPH'], axis=1)

        # plot a graph with a few features (x values)
        # against the precipitation or rainfall to observe
        # the trends

        print("Precipitation vs selected attributes graph: ")

        for i in range(self.x_vis.columns.size):
            plt.subplot(3, 2, i + 1)
            plt.scatter(self.days, self.x_vis[self.x_vis.columns.values[i][:100]],
                        color='g')

            plt.scatter(self.days[self.day_index],
                        self.x_vis[self.x_vis.columns.values[i]][self.day_index],
                        color='r')

            plt.title(self.x_vis.columns.values[i])

        plt.show()

#A day (in red) having precipitation of about 2 inches is tracked across multiple parameters
# (the same day is tracker across multiple features such as temperature, pressure, etc).
# The x-axis denotes the days and the y-axis denotes the magnitude of the feature such as temperature, pressure, etc.
# From the graph, it can be observed that rainfall can be expected to be high when the temperature is high and humidity is high.



Reg=Reg()
Reg.data_preprocessing()
Reg.Feature_Extraction()
Reg.regression_model()

#generating a time series graph
df = pd.read_csv('./austin_weather.csv',header=0, index_col=0, parse_dates=True)
df.plot()
plt.title("Time Stamp")
plt.ylabel("Temperature")
plt.show()