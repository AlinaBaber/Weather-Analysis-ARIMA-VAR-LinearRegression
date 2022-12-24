import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)

class ts:
    def data_preprocessing(self):
        self.df = pd.read_csv('./austin_weather.csv')
        self.df.set_index('Date').sort_index()

        # use average data only
        self.columns_of_interest = ['TempAvgF','DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH', 'PrecipitationSumInches']
        self.data = self.df[self.columns_of_interest]
        self.events = self.df[['Events']].replace(' ', 'None')

        self.events.Events.value_counts().plot(kind='bar', figsize=(10,5))
        plt.show()

        #Get unique events categories
        self.unique_events = set()
        for value in self.events.Events.value_counts().index:
            self.splitted = [x.strip() for x in value.split(',')]
            self.unique_events.update(self.splitted)
        print("unique_events:\n",self.unique_events)


        self.single_events = pd.DataFrame()
        for event_type in self.unique_events:
            self.event_occurred = self.events.Events.str.contains(event_type)
            self.single_events = pd.concat([self.single_events, pd.DataFrame(data={event_type: self.event_occurred.values})], join='outer', axis=1)

        #single_events.head()
        ax = self.single_events.sum().sort_values(ascending=False).plot.bar(figsize=(10,5))
        ax.set_title("Weather events in dataset", fontsize=18)
        ax.set_ylabel("Number of occurrences", fontsize=14)
        for i in ax.patches:
            ax.text(i.get_x()+.18, i.get_height()+5, i.get_height(), fontsize=12)
        plt.show()



        print("\nsingle_events:\n",self.single_events.head())

        #Check how many traces do we have in PrecipitationSumInches collum
        self.precipitation = self.data[pd.to_numeric(self.data.PrecipitationSumInches, errors='coerce').isnull()].PrecipitationSumInches.value_counts()
        print("\nprecipitation:\n",self.precipitation)

        # this function returns array with one item for each row
        # each item indicates if the row with columns of our interest had non-numeric data
    def isColumnNotNumeric(self,columns_of_interest, data):
        result = np.zeros(data.shape[0], dtype=bool)
        for column_name in columns_of_interest:
            result = result | pd.to_numeric(data[column_name], errors='coerce').isnull()
        return result

    def getDataFrameWithNonNumericRows(self,dataFrame):
        return self.data[ts.isColumnNotNumeric(self.columns_of_interest, self.data)]

    def numberOrZero(self,value):
        try:
            parsed = float(value)
            return parsed
        except:
            return 0

    def Feature_Extraction(self):

        non_numeric_rows_count = ts.getDataFrameWithNonNumericRows(self.data).shape[0]

        print("\nNon numeric rows: {0}\n".format(non_numeric_rows_count))


        # this line is unnecessary if we run script from top to bottom,
        # but it helps debugging this part of code to get fresh PrecipitationSumInches column
        self.data['PrecipitationSumInches'] = self.df['PrecipitationSumInches']

        #Find rows indices with "T" values
        self.has_precipitation_trace_series = ts.isColumnNotNumeric(['PrecipitationSumInches'], self.data).astype(int)
        #data['PrecipitationTrace'] = has_precipitation_trace_series
        #data.loc[:,'PrecipitationTrace'] = has_precipitation_trace_series
        self.data = self.data.assign(PrecipitationTrace=self.has_precipitation_trace_series.values)

        self.data['PrecipitationSumInches'] = self.data['PrecipitationSumInches'].apply(ts.numberOrZero)
        print("\n",self.data.iloc[0:10,:])


        #Check how many non numeric rows we still have

        print("\nCheck how many non numeric rows we still have:\n",ts.getDataFrameWithNonNumericRows(self.data))

        self.row_indices_for_missing_values = ts.getDataFrameWithNonNumericRows(self.data).index.values
        print("\nrow indices for missing values:\n",self.row_indices_for_missing_values)
        self.df=self.df.drop(self.row_indices_for_missing_values)
        self.data_prepared = self.data.drop(self.row_indices_for_missing_values)
        self.events_prepared = self.single_events.drop(self.row_indices_for_missing_values)
        print("Data rows: {0}, Events rows: {1}".format(self.data_prepared.shape[0], self.events_prepared.shape[0]))

        #Convert dataframe columns to be treated as numbers
        print("\ndata types:\n",self.data_prepared.dtypes)
        self.data_prepared = self.data_prepared.apply(pd.to_numeric)
        print("\nafter converting data types\n",self.data_prepared.dtypes)

        #Normalize input data
        self.data_values = self.data_prepared.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()

        self.data_prepared = pd.DataFrame(min_max_scaler.fit_transform(self.data_prepared), columns=self.data_prepared.columns, index=self.data_prepared.index)




        self.data_prepared["Date"]=self.df["Date"]
        self.data_prepared['Date'] = pd.to_datetime(self.data_prepared['Date'])
        self.data_prepared=self.data_prepared.set_index('Date')
        print(self.data_prepared.head())
        print("\n\nafter inserting the date:\n",self.data_prepared.dtypes)

        #Final look at the prepared data


        print("\nfinal look at prepared data:\n",self.data_prepared.head())
        print("\nfinal look at event prepared\n",self.events_prepared.head())


        # for i in columns_of_interest:
        #     z = data_prepared[i].resample('MS').mean()
        #     z.plot(figsize=(15, 6))
        #     plt.ylabel(i)
        #     plt.show()
        #     decomposition = sm.tsa.seasonal_decompose(z, model='additive')
        #     fig = decomposition.plot()
        #     plt.title(i)
        #     plt.show()

    def model_fitting(self):
        #creating the train and validation set
        self.train = self.data_prepared[:int(0.8*(len(self.data_prepared)))]
        self.valid = self.data_prepared[int(0.8*(len(self.data_prepared))):]

        #fit the model
        from statsmodels.tsa.vector_ar.var_model import VAR

        model = VAR(endog=self.train)
        model_fit = model.fit()

        # make prediction on validation
        self.prediction = model_fit.forecast(model_fit.y, steps=len(self.valid))

        print('prediction:\n',self.prediction)
        print("\n\n predection 1 chunk:\n",self.prediction[0])
        #converting predictions to dataframe
        self.pred = pd.DataFrame(index=range(0,len(self.prediction)),columns=[self.data_prepared.columns])
        for j in range(0,8):
            for i in range(0, len(self.prediction)):
               self.pred.iloc[i][j] = self.prediction[i][j]
        print("\npred:\n",self.pred)

        self.pred.columns=['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH', 'PrecipitationSumInches', 'PrecipitationTrace']


        print("tarin shape:",self.train.shape)
        print("valid shape:",self.valid.shape)
        print("pred shape:",self.pred.shape)
        for i in self.data_prepared.columns:
            print('rmse value for', i, 'is : ', np.sqrt(mean_squared_error(self.pred[i], self.valid[i])))

        #make final predictions
        model = VAR(endog=self.data_prepared)
        model_fit = model.fit()
        self.yhat = model_fit.forecast(model_fit.y, steps=1)
        print("\nyaht:\n",self.yhat)

        self.pred["Date"]=self.valid.index
        #pred['Date'] = pd.to_datetime(data_prepared['Date'])
        self.pred=self.pred.set_index('Date')
        print(self.pred)
        for col in self.pred.columns:
            plt.figure(figsize=(15, 8))
            # x = valid.index
            # y = pred[col]
            # z=valid[col]
            # plt.plot(x, y,'b')
            # plt.plot(x,z,'r')
            plt.plot(self.valid[col])
            plt.plot(self.pred[col])
            plt.xlabel("Date")
            plt.ylabel(col)
            plt.title("Date and "+ col + " relationship")
            plt.legend(['valid', 'prediction'], loc='upper left')
            plt.show()
            return self.yhat

ts=ts()
ts.data_preprocessing()
ts.Feature_Extraction()
temp=ts.model_fitting()