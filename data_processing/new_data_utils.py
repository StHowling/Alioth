import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb

RAW_FEATURE_NUM = 218 
LIBVIRT_FEATURE_NUM = 15
SAR_FEATURE_NUM = 24
PQOS_FEATURE_NUM = 4

SPECIAL_APPS = ['milc', 'rabbitmq']

def get_data(df, feature_selection, qos_metric):
    cols = list(df.columns)

    if feature_selection == 'all':
        selected_cols = cols[1:-7]
    elif feature_selection == 'online':
        num_feature_round = len(cols) // RAW_FEATURE_NUM
        selected_cols = []
        for i in range(num_feature_round):
            selected_cols = selected_cols + cols[1 + i*RAW_FEATURE_NUM : 1 + i*RAW_FEATURE_NUM + LIBVIRT_FEATURE_NUM ]
            selected_cols = selected_cols + cols[1 + (i+1)*RAW_FEATURE_NUM - PQOS_FEATURE_NUM - SAR_FEATURE_NUM : 1 + (i+1)*RAW_FEATURE_NUM - PQOS_FEATURE_NUM ]
    
    if isinstance(selected_cols[0], int):
        feature = df.iloc[:,selected_cols]
    elif isinstance(selected_cols[0],str):
        feature = df.loc[:,selected_cols]
    else:
        raise ValueError

    
    # df.loc[df['app']=='milc','qos2'] =  df.loc[df['app']=='milc','qos1']
    tmp = df.loc[df['app']=='rabbitmq',['qos1','qos2']].apply(np.mean, axis=1)

    if qos_metric == 'tps':
        label = df.loc[:,'qos1'].to_frame('qos')
        label.loc[df['app']=='rabbitmq','qos'] = tmp
    elif qos_metric == 'latency':
        label = 2 - df.loc[:,'qos2'].to_frame('qos')
        label.loc[df['app']=='rabbitmq','qos'] = tmp
        label.loc[df['app']=='milc','qos'] = df.loc[df['app']=='milc','qos1']
    else:
        raise ValueError

    label.loc[label['qos']<0,'qos'] = 0

    data = pd.concat([feature,label, df.iloc[:,-5:]],axis=1)

    return data

def get_l1o_split(df, target_app):
    numeric_cols = list(df.columns[:-5])
    train_part = df.loc[(df['app']!=target_app), numeric_cols]
    test_part = df.loc[(df['app']==target_app), numeric_cols]

    transformations = [("scaler", MinMaxScaler(clip=False), list(train_part.columns)[:-1])]
    preprocessor = ColumnTransformer(transformations,remainder="passthrough")

    preprocessor.fit(train_part)
    train_data = preprocessor.transform(train_part)
    test_data = preprocessor.transform(test_part)
    train_feature = train_data[:,:-1]
    train_label = train_data[:,-1]
    test_feature = test_data[:,:-1]
    test_label = test_data[:,-1]

    return train_feature,train_label,test_feature,test_label

def get_self_split(df, target_app, random_state=42):
    numeric_cols = list(df.columns[:-5])
    data = df.loc[(df['app']==target_app), numeric_cols]
    train_part, test_part = train_test_split(data, test_size=0.2, random_state=random_state)

    transformations = [("scaler", MinMaxScaler(clip=False), list(train_part.columns)[:-1])]
    preprocessor = ColumnTransformer(transformations,remainder="passthrough")

    preprocessor.fit(train_part)
    train_data = preprocessor.transform(train_part)
    test_data = preprocessor.transform(test_part)
    train_feature = train_data[:,:-1]
    train_label = train_data[:,-1]
    test_feature = test_data[:,:-1]
    test_label = test_data[:,-1]

    return train_feature,train_label,test_feature,test_label


def get_all_split(df, target_app, random_state=42):
    numeric_cols = list(df.columns[:-5])
    # data = df.loc[(df['app']!=''), numeric_cols]
    train_part, test_part = train_test_split(df, test_size=0.2, random_state=random_state)
    train_part = train_part.loc[(train_part['app']!=''),numeric_cols]
    test_part = test_part.loc[(test_part['app']==target_app),numeric_cols]

    transformations = [("scaler", MinMaxScaler(clip=False), list(train_part.columns)[:-1])]
    preprocessor = ColumnTransformer(transformations,remainder="passthrough")

    preprocessor.fit(train_part)
    train_data = preprocessor.transform(train_part)
    test_data = preprocessor.transform(test_part)
    train_feature = train_data[:,:-1]
    train_label = train_data[:,-1]
    test_feature = test_data[:,:-1]
    test_label = test_data[:,-1]

    return train_feature,train_label,test_feature,test_label


if __name__=='__main__':
    new_df = pd.read_csv('mul_all_intall_warming3_nosliding_mean-max-min.csv')
    new_df.drop(new_df[new_df['app']=='noapp'].index,inplace=True)

    app_list = list(pd.unique(new_df['app']))
    new_df_ltc = get_data(new_df,'all','latency')
    # train: target app, test: target app
    for app in app_list:
        print(app)
        # train all\target, test target, use l1o
        # train all-target, test target, use self
        train_x,train_y,test_x,test_y = get_self_split(new_df_ltc,app)
        model = xgb.XGBRegressor(n_estimators=100,verbosity=1).fit(train_x, train_y)
        predict = model.predict(test_x)
        print("MSE",mean_squared_error(predict, test_y))
        print("MAE",mean_absolute_error(predict, test_y))
        print("MAPE",mean_absolute_percentage_error(predict, test_y))
    
    # train: all app, test: all app
    numeric_cols = list(new_df_ltc.columns[:-5])
    data = new_df_ltc.loc[(new_df_ltc['app']!=''), numeric_cols]
    train_part, test_part = train_test_split(data, test_size=0.2, random_state=42)

    transformations = [("scaler", MinMaxScaler(clip=False), list(train_part.columns)[:-1])]
    preprocessor = ColumnTransformer(transformations,remainder="passthrough")

    preprocessor.fit(train_part)
    train_data = preprocessor.transform(train_part)
    test_data = preprocessor.transform(test_part)
    train_feature = train_data[:,:-1]
    train_label = train_data[:,-1]
    test_feature = test_data[:,:-1]
    test_label = test_data[:,-1]
    model = xgb.XGBRegressor(n_estimators=100,verbosity=1).fit(train_feature, train_label)
    predict = model.predict(test_feature)
    print("MSE",mean_squared_error(predict, test_label))
    print("MAE",mean_absolute_error(predict, test_label))
    print("MAPE",mean_absolute_percentage_error(predict, test_label))
