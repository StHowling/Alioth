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



def extract_temporal_feature(df, interval=3, warming_up=3, consecutive_sliding=True, func_types=['mean', 'max', 'min'], allow_no_stress_duplicate=False):
    app_workload_intensity_map = {}
    app_list = pd.unique(df['app'])
    for app in app_list:
        app_workload_intensity_map[app] = pd.unique(df.loc[df['app']==app, 'workload'])
    stress_intensity_map = {}
    stress_types = set(pd.unique(df['stress_type']))
    stress_types.remove('NO_STRESS')
    for stress_type in stress_types:
        stress_intensity_map[stress_type] = pd.unique(df.loc[df['stress_type'] == stress_type, 'stress_intensity'])

    df_out = pd.DataFrame()    

    for app in app_list:
        for intensity in app_workload_intensity_map[app]:
            for stress in stress_types:
                for itst in stress_intensity_map[stress]:
                    args = (interval, warming_up, consecutive_sliding)
                    tmp = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst)), 'timestamp'].to_frame().apply(func_to_apply_for_timestamp, args=args)

                    for ft in func_types:
                        args = (interval, warming_up, consecutive_sliding, ft)
                        # "1：-6" means all metrics
                        tmp_feature = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst)), list(df.columns)[1:-6]].apply(func_to_apply_for_feature, args=args)
                        tmp_feature.columns = flatten_name(ft, tmp_feature.columns)
                        tmp = pd.concat([tmp,tmp_feature],axis=1)

                    args = (interval, warming_up, consecutive_sliding, 'mean')
                    # -6,-5 are "qos1" and "qos2"
                    tmp_label = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst)), list(df.columns)[-6:-4]].apply(func_to_apply_for_feature, args=args)
                    tmp = pd.concat([tmp,tmp_label],axis=1)

                    args = (interval, warming_up, consecutive_sliding, 'min')
                    # the last 4 columns are string labels of stress type, stress intensity, app, and workload (intensity)
                    tmp_auxlabel = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst)), list(df.columns)[-4:]].apply(func_to_apply_for_feature, args=args)
                    tmp = pd.concat([tmp,tmp_auxlabel],axis=1)

                    if interval == -1:
                        actual_interval = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == stress) & (df['stress_intensity'] == itst)), 'timestamp'].shape[0]-warming_up
                        tmp.insert(tmp.shape[1], 'interval', np.ones(tmp.shape[0])*actual_interval)
                    else:
                        tmp.insert(tmp.shape[1], 'interval', np.ones(tmp.shape[0])*interval)

                    df_out = pd.concat([df_out, tmp], axis=0)

            # deal with no_stress metrics for this app-intensity, allow duplicate means each under-stress metric vector will have multiple targets in the DAE training
            if allow_no_stress_duplicate:
                tmp = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), 'timestamp'].to_frame().apply(
                    func_to_apply_for_timestamp, args=(interval, warming_up, consecutive_sliding))
                
                for ft in func_types:
                    tmp_feature = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), list(df.columns)[1:-6]].apply(
                        func_to_apply_for_feature, args=(interval,warming_up,consecutive_sliding,ft))
                    tmp_feature.columns = flatten_name(ft, tmp_feature.columns)
                    tmp = pd.concat([tmp,tmp_feature],axis=1)

                tmp_label = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), list(df.columns)[-6:-4]].apply(
                    func_to_apply_for_feature, args=(interval, warming_up, consecutive_sliding, 'mean'))
                tmp = pd.concat([tmp,tmp_label],axis=1)

                tmp_auxlabel = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), list(df.columns)[-4:]].apply(
                    func_to_apply_for_feature, args=(interval,warming_up,consecutive_sliding,'min'))
                tmp = pd.concat([tmp,tmp_auxlabel],axis=1)

                tmp.insert(tmp.shape[1], 'interval', np.ones(tmp.shape[0])*interval)

            else:
                start_timestamp = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), 'timestamp'].iloc[warming_up]
                tmp = pd.DataFrame([start_timestamp], columns=['timestamp'])


                for ft in func_types:
                    tmp_feature = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), list(df.columns)[1:-6]].apply(
                        func_to_apply_for_feature, args=(-1,warming_up,False,ft))
                    
                    tmp_feature.columns = flatten_name(ft, tmp_feature.columns)
                    tmp = pd.concat([tmp,tmp_feature],axis=1)                   

                tmp.insert(tmp.shape[1], 'qos1', np.ones(1))
                tmp.insert(tmp.shape[1], 'qos2', np.ones(1))

                tmp_auxlabel = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), list(df.columns)[-4:]].apply(
                    func_to_apply_for_feature, args=(-1,warming_up,False,'min'))
                tmp = pd.concat([tmp,tmp_auxlabel],axis=1)

                actual_interval = df.loc[((df['app']==app) & (df['workload']==intensity) & (df['stress_type'] == 'NO_STRESS')), 'timestamp'].shape[0]-warming_up
                tmp.insert(tmp.shape[1], 'interval', np.ones(1)*actual_interval)

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


if __name__ == '__main__':
    os.chdir(DATADIR)
    df = pd.read_csv('./basic-mul-all-merged.csv')

    # original flavor of data preprocessing

    # df_out = extract_temporal_feature(df, 3, 0, False, ['mean'])

    # df_out.to_csv('./mul_all_int3_nowarming_nosliding_mean.csv', index=False)


    # all-included sliding

    intervals = [3, 5, 10, -1] # not enough data points for interval=20 (even for 15)

    # df_out = pd.DataFrame()
    # for i in intervals:
    #     print('Dealing with interval =', i)
    #     tmp = extract_temporal_feature(df, i, func_types=['mean', 'max', 'min', 'max_diff', 'std'])
    #     df_out = pd.concat([df_out,tmp])

    # df_out.to_csv('./mul_all_intall_warming3_withsliding_all.csv', index=False)


    # all-included no sliding

    df_out = pd.DataFrame()
    for i in intervals:
        print('Dealing with interval =', i)
        tmp = extract_temporal_feature(df, i, consecutive_sliding=False, func_types=['mean', 'max', 'min', 'max_diff', 'std'])
        df_out = pd.concat([df_out,tmp])

    df_out.to_csv('./mul_all_intall_warming3_nosliding_all.csv', index=False)


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


