import pandas as pd
import numpy as np
import os

DATADIR = "./data/output"

# Consider all disk and net related metrics as log-keys and ignore them in filtering extremely large outliers (As inspected, there is not such faulty data in them)

LOG_KEYS = ['net_rd_byte',
            'net_wr_byte',
            'net_rd_packet',
            'net_wr_packet',
            'rd_bytes',
            'wr_bytes',
            'rd_total_times',
            'wr_total_times',
            'flush_total_times',
            'tps',
            'rtps',
            'wtps',
            'bread_s',
            'bwrtn_s',
            'rxpck_s',
            'txpck_s',
            'rxkB_s',
            'txkB_s']

def flatten_name(postfix, src_names):
    ret = []
    for c in src_names:  
        ret.append(c + '.' + postfix)
    return ret

def func_to_apply_for_feature(series, interval, warming_up, consecutive_sliding, func_type):
    assert warming_up < len(series)
    val = series.values[warming_up:]
    if interval == -1:
        indices = [0]
        interval = len(val)
    else:
        if consecutive_sliding:
            indices = np.arange(len(val) - interval + 1)
        else:
            indices = np.arange(0 , len(val) - interval + 1, interval)
     

    if isinstance(val[0],str):
        result = val[indices]
    else:
        result = []
        if func_type == 'mean':
            for idx in indices:
                result.append(np.mean(val[idx:idx+interval]))
        elif func_type == 'max':
            for idx in indices:
                result.append(np.max(val[idx:idx+interval]))
        elif func_type == 'min':
            for idx in indices:
                result.append(np.min(val[idx:idx+interval]))
        elif func_type == 'max_diff':
            for idx in indices:
                result.append(np.max(val[idx:idx+interval]) - np.min(val[idx:idx+interval]))
        elif func_type == 'std':
            for idx in indices:
                result.append(np.std(val[idx:idx+interval]))
        else:
            raise NotImplementedError
        result = np.array(result)

    return pd.Series(result, copy=True)



def func_to_apply_for_timestamp(series, interval, warming_up, consecutive_sliding):
    assert warming_up < series.shape[0]
    val = series.values[warming_up:]

    if interval == -1:
        return pd.Series([val[0]],copy=True)
    
    if consecutive_sliding:
        result = val[: (1-interval)]
    else:
        result = val[:(1-interval):interval]

    return pd.Series(result,copy=True)



def extract_temporal_feature(df, interval=3, warming_up=3, consecutive_sliding=True, func_types=['mean', 'max', 'min']):
    app_workload_intensity_map = {}
    app_list = pd.unique(df['app'])
    for app in app_list:
        app_workload_intensity_map[app] = pd.unique(df.loc[df['app']==app, 'workload'])
    stress_intensity_map = {}
    stress_types = pd.unique(df['stress_type'])
    for stress_type in stress_types:
        stress_intensity_map[stress_type] = pd.unique(df.loc[df['stress_type'] == stress_type, 'stress_intensity'])

    df_out = pd.DataFrame()    

    for app in app_list:
        for intensity in app_workload_intensity_map[app]:
            for stress in stress_types:
                for itst in stress_intensity_map[stress]:
                    args = (interval, warming_up, consecutive_sliding)
                    tmp = df.loc[(df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst), 'timestamp'].to_frame().apply(func_to_apply_for_timestamp, args=args)

                    for type in func_types:
                        args = (interval, warming_up, consecutive_sliding, type)
                        tmp_feature = df.loc[(df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst), list(df.columns)[1:-6]].apply(func_to_apply_for_feature, args=args)
                        tmp_feature.columns = flatten_name(type, tmp_feature.columns)
                        tmp = pd.concat([tmp,tmp_feature],axis=1)

                    args = (interval, warming_up, consecutive_sliding, 'mean')
                    tmp_label = df.loc[(df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst), list(df.columns)[-6:-4]].apply(func_to_apply_for_feature, args=args)
                    tmp = pd.concat([tmp,tmp_label],axis=1)

                    args = (interval, warming_up, consecutive_sliding, 'min')
                    tmp_auxlabel = df.loc[(df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst), list(df.columns)[-4:]].apply(func_to_apply_for_feature, args=args)
                    tmp = pd.concat([tmp,tmp_auxlabel],axis=1)

                    if interval == -1:
                        actual_interval = df.loc[(df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst), 'timestamp'].shape[0]-warming_up
                        tmp.insert(tmp.shape[1], 'interval', np.ones(tmp.shape[0])*actual_interval)
                    else:
                        tmp.insert(tmp.shape[1], 'interval', np.ones(tmp.shape[0])*interval)

                    df_out = pd.concat([df_out, tmp], axis=0)

    df_out.reset_index(drop=True,inplace=True)
    log_transform_df(df_out)
    return df_out

def log_transform_df(df):
    cols = df.columns
    for col in cols:
        if col.split('.')[0] in LOG_KEYS:
            tmp = df[col].copy()
            tmp = np.log(tmp+1)
            # df[i > 1] = np.log(df[df[i > 1]])
            df[col] = tmp
    return df

def adjust_csv_file(filename):
    # Making up for some faults during inspecting the effects of extract_temporal_feature(). Clearly max, min, max_diff and std of QoS is not our interested label to be predicted
    # This function exists only because that rerunning extract_temporal_feature() is too time-consuming
    # No longer useful as extract_temporal_feature() has been corrected.  Avoid calling it in main
    df = pd.read_csv(filename)
    keys = list(df.columns)
    crt_idx = keys.index('qos2.mean')
    label = df.iloc[:, crt_idx-1:crt_idx+1]
    df_out = pd.DataFrame()
    for i in range(len(keys) // (crt_idx+1)):
        partial_feature = df.iloc[:, i*(crt_idx+1) : (i+1)*(crt_idx+1)-2]
        df_out = pd.concat([df_out,partial_feature],axis=1)

    df_out = pd.concat([df_out,label],axis=1)
    auxlabel = df.iloc[:, -5:] # fuck, previously set to -4 and run. now it is meaningless and boil down to the very beginning
    df_out = pd.concat([df_out, auxlabel], axis=1)

    # print(list(df_out.columns))
    df_out.to_csv(filename, index=False)

def extract_no_stress_data_from_csv(filename, savename, allow_duplicate=False):
    df = pd.read_csv(filename)

    if allow_duplicate:
        # allow multiple no_stress target of single (app,stress) tuple for DAE usage
        df_out = df.loc[df['stress_type']=='no_stress', list(df.columns)]
    else:
        # warning: data-specific implementation
        # regret that why haven't just use -1 as interval label of 'all'
        df_out = df.loc[(df['stress_type']=='no_stress') & (df['interval'] > 10), list(df.columns)]

    df_out.to_csv(savename,index=False)
    


if __name__ == '__main__':
    os.chdir(DATADIR)
    df = pd.read_csv('./basic-mul-all-merged.csv')

    # original flavor of data preprocessing

    # df_out = extract_temporal_feature(df, 3, 0, False, ['mean'])

    # df_out.to_csv('./mul_all_int3_nowarming_nosliding_mean.csv', index=False)


    # all-included sliding

    intervals = [3, 5, 10, -1] # not enough data points for interval=20 (even for 15)

    df_out = pd.DataFrame()
    for i in intervals:
        print('Dealing with interval =', i)
        tmp = extract_temporal_feature(df, i, func_types=['mean', 'max', 'min', 'max_diff', 'std'])
        df_out = pd.concat([df_out,tmp])

    df_out.to_csv('./mul_all_intall_warming3_withsliding_all.csv', index=False)


    # all-included no sliding

    # df_out = pd.DataFrame()
    # for i in intervals:
    #     print('Dealing with interval =', i)
    #     tmp = extract_temporal_feature(df, i, consecutive_sliding=False, func_types=['mean', 'max', 'min', 'max_diff', 'std'])
    #     df_out = pd.concat([df_out,tmp])

    # df_out.to_csv('./mul_all_intall_warming3_nosliding_all.csv', index=False)


    # practical (with only mean-max-min) sliding

    df_out = pd.DataFrame()
    for i in intervals:
        print('Dealing with interval =', i)
        tmp = extract_temporal_feature(df, i)
        df_out = pd.concat([df_out,tmp])

    df_out.to_csv('./mul_all_intall_warming3_withsliding_mean-max-min.csv', index=False)

    # practical (with only mean-max-min) no sliding

    df_out = pd.DataFrame()
    for i in intervals:
        print('Dealing with interval =', i)
        tmp = extract_temporal_feature(df, i, consecutive_sliding=False)
        df_out = pd.concat([df_out,tmp])

    df_out.to_csv('./mul_all_intall_warming3_nosliding_mean-max-min.csv', index=False)

    # adjust_csv_file('./mul_all_intall_warming3_withsliding_all.csv')
    # adjust_csv_file('./mul_all_intall_warming3_nosliding_all.csv')
    # adjust_csv_file('./mul_all_intall_warming3_withsliding_mean-max-min.csv')
    # adjust_csv_file('./mul_all_intall_warming3_nosliding_mean-max-min.csv')
