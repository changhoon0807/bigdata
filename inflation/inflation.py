import pickle

import pandas as pd
from pandas.tseries.offsets import QuarterEnd, QuarterBegin, MonthEnd, MonthBegin, BDay
from pandas.tseries.offsets import MonthEnd, MonthBegin
from pandas.tseries.offsets import DateOffset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker

import requests

import statsmodels.api as sm
import statsmodels.formula.api as smf

import math

import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet as ENet
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import ExtraTreesRegressor as EXT

from datetime import datetime, timedelta
from multiprocessing import Process

import glob
import os

from time import time
from tqdm import tqdm
import itertools

from tqdm import notebook

import s3fs
    
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


idx = pd.IndexSlice

plt.rcParams["font.family"] = "NanumGothicCoding"
plt.rcParams['axes.unicode_minus'] = False


def Nth_friday(year, month, order):
    """Return 1st, 2nd, ... , last friday of given year and month,
    order : 0, 1, 2, ..., -1
    """
    fridays = pd.date_range('2000-01-01', '2050-12-31', freq='W-FRI')
    fridays_of_the_month = [date for date in fridays if (date.year == year) & (date.month == month)]

    try:
        return fridays_of_the_month[order]
    except:
        print(f'The month has no {order + 1}th friday')
        
def gen_lagged(X, lags):

    temp = X.copy()

    for l in range(1, 1 + lags):
        lX = temp.shift(l)
        lX.columns = ['l' + str(l) + '_' + col for col in temp.columns]
        X = pd.concat([X, lX], axis=1)

    return X

def get_train_data_v5(df0,
                      l = 0,
                      data_group = 1,
                      excl_alt = True,
                      m1 = None,
                      fillna='ffill',
                      rolling = 0,
                      sm = '2006-01-01',
                      predictors=None
                     ):
    s3_repo_path = 's3://newtech/public/inf_nowcasting'
    
    df = df0.copy()

    if data_group in [1, 2, 3]:
        vspec = pd.read_excel(f'{s3_repo_path}/input/data_list_all_v5.xlsx', index_col = None)
        vspec.index.names = [None]
        alt_var_list = vspec.loc[vspec.Adcode.eq(1), 'My ID'].values
        
        # include variables with Gcode between 1 and data_group
        # if data_group == 1, we use variables with Gcode 1
        # if data_group == 3, we use variables with Gcode 1, 2, 3
        df = df[vspec.loc[vspec.Gcode.between(1, data_group), 'My ID'].values]

        if excl_alt:
            df = df[[col for col in df.columns if col not in alt_var_list]]

    else:
        df = df.loc[:, data_group]

    # 최종 예측시점 월까지 시계열 연장을 위해 인덱스 추가
    if m1:
        df = df.reindex(pd.date_range(df.index[0], m1, freq='M'))

    # 월중 해당월 기준 통계가 공표되는 변수들(lag0_var_list)은 그대로 예측변수에 포함하고
    # 이외 변수들은 shift(1)하여 예측변수에 포함 (직전월 기준 통계가 공표되는 변수들 lag1_var_list, ...)
    # 예를 들어, 9월중 전망시계 9월 인플레이션(h=0인 타겟변수)를 전망(나우캐스팅)할때,
    # lag0_var_list 변수들은 9월값(없는 경우 보간된 값), lag1_var_list 변수들은 8월값(없는 경우 보간된 값), .. 등을 이용
    #X = df[lag0_var_list].copy()
    #X = pd.concat([X, df[[col for col in df.columns if col not in lag0_var_list]].shift(1)], axis=1)
    X = df.copy()

    # 결측치 보간 ffill, 2004년 이후 데이터 이용
    if fillna == 'ffill':
        X = X.fillna(method='ffill').loc[sm:] #NaN을 이전값으로 채우고, 2004-01-01이후의 데이터만 사용

    # 예측변수 정규화
    Xm = X.mean()
    Xs = X.std()
    Xn = (X - Xm)/Xs

    # 결측치 보간 at head
    #Xn = Xn.fillna(0)
    Xn = Xn.fillna(method = 'bfill')
    
    # 결측치 보간 at head
    #Xn = Xn.fillna(0)
    Xn = Xn.fillna(method = 'bfill')
    
    # 선형회귀를 위해 예측변수가 주어진 경우
    if not(predictors is None):
        Xn = Xn.loc[:, predictors]

    # lag variable 생성
    LX = gen_lagged(Xn, l)

    # rolling or recursive
    if (rolling > 0) and (len(LX) >= rolling * 12):
        dump_months = len(LX) - rolling * 12
        LX = LX.iloc[dump_months:]

    # 타겟변수
    y = df.loc[:, 'P_cpi_1']

    # 설명변수에서 타겟변수 제거
    LX = LX.drop('P_cpi_1', axis=1)

    return LX, y

def get_error_by_vintage(pred, act):
    err = pred.copy()

    hors = err.columns.get_level_values(0).unique()
    cols = err.columns.get_level_values(1).unique()

    for hor, col in itertools.product(hors, cols):
        err.loc[:, idx[hor, col]] -= act.loc[col]

    return err

def align_error_by_week(error):

    err = pd.DataFrame()
    mae = pd.DataFrame()
    rmse = pd.DataFrame()

    hors = error.columns.get_level_values(0).unique()

    for hor in hors:
        df = error[hor].dropna(how='all', axis=1)
        tmp = pd.DataFrame(index=np.arange(-52, 0))

        for col in df.columns:
            dfi = df.loc[:, col].dropna()
            dfi = dfi.reset_index(drop=True)
            dfi.index = dfi.index - len(dfi)
            tmp = pd.concat([tmp, dfi], axis=1)

        tmp = tmp.dropna(axis=0, how='all')
        mae0 = tmp.apply(lambda x: np.mean(np.abs(x)), axis=1).to_frame(hor)
        rmse0 = tmp.apply(lambda x: np.sqrt(np.mean(x**2)), axis=1).to_frame(hor)

        tmp = pd.concat([tmp], axis=1, keys=[hor])
        err = pd.concat([err, tmp], axis=1)
        mae = pd.concat([mae, mae0], axis=1)
        rmse = pd.concat([rmse, rmse0], axis=1)

    err = err.sort_index()
    mae = mae.sort_index()
    rmse = rmse.sort_index()

    return err, mae, rmse

def get_pred_last(pred):

    hors = pred.columns.get_level_values(0).unique()
    targets = pred[hors[0]].columns

    pred_last = pd.DataFrame(index=targets, columns=hors)
    for tm in targets:
        for hor in hors:
            try:
                pred_last.loc[tm, hor] = pred[hor].loc[:, tm].dropna().iloc[-1]
            except IndexError:
                pass

    return pred_last

def get_mda(pred_last, act, hor = 0, base='act', scale=1, print_result=True, model = ''):

    targets = pred_last.index

    act_diff = act[targets] - act[targets].shift(hor + 1)
    jump_size = scale*act_diff.std()
    big_jumps = np.abs(act_diff) > jump_size

    act_sign = np.sign(act_diff.dropna())

    if base == 'act':
        pred_sign = np.sign((pred_last[targets] - act[targets].shift(hor + 1)).dropna())
    else:
        pred_sign = np.sign((pred_last[targets] - pred_last[targets].shift(hor + 1)).dropna())

    matched_signs = act_sign == pred_sign
    mda = matched_signs.sum()/len(matched_signs)

    mda_big_den = matched_signs[big_jumps].sum()
    mda_big_num = len(matched_signs[big_jumps])
    mda_big = mda_big_den/mda_big_num

    p1 = f'mda: {mda:.2f} ({matched_signs.sum()}/{len(matched_signs)})'
    p2 = f'mda_big: {mda_big:.2f} ({mda_big_den}/{mda_big_num}, {jump_size:.2f})'

    if print_result:
        print(p1, end=', ')
        print(p2, end=' ')
        print(model)

    return matched_signs, mda, mda_big, p1, p2

def get_pred(pred_files, model_names):

    PRED = pd.DataFrame()

    for file, name in zip(pred_files, model_names):

        pred = pd.read_pickle(file)
        pred1 = pd.concat([pred], axis=1, keys=[name])
        PRED = pd.concat([PRED, pred1], axis=1)

    return PRED


def get_eval_last_pred(PRED, model_names, act):

    MAE = pd.DataFrame()
    RMSE = pd.DataFrame()
    PRED_last = pd.DataFrame()

    for name in tqdm(model_names): #notebook.tqdm(model_names):

        pred = PRED[name]

        err, mae, rmse = align_error_by_week(get_error_by_vintage(pred, act))
        mae = pd.concat([mae], axis=1, keys=[name])
        rmse = pd.concat([rmse], axis=1, keys=[name])
        MAE = pd.concat([MAE, mae], axis=1)
        RMSE = pd.concat([RMSE, rmse], axis=1)

        pred_last = get_pred_last(pred)
        pred_last = pd.concat([pred_last], axis=1, keys=[name])
        PRED_last = pd.concat([PRED_last, pred_last], axis=1)

    MAE = MAE.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    RMSE = RMSE.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    PRED_last = PRED_last.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    return MAE, RMSE, PRED_last

def plot_mae_rmse(mae, rmse, rw_mae, rw_rmse, h=0, best=5, good=20, title='ARIMA',
                  ncol=3, loc=3, figsize=(20, 10), bbox_to_anchor=(1, -0.1), fontsize=12, rw=True, ylim=None):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    mae_best = mae[h].loc[-1].sort_values().iloc[:best].index.tolist()
    rmse_best = rmse[h].loc[-1].sort_values().iloc[:best].index.tolist()

    mae_good = mae[h].loc[-1].sort_values().iloc[:good].index.tolist()
    rmse_good = rmse[h].loc[-1].sort_values().iloc[:good].index.tolist()

    for m in list(set(mae_good + rmse_good)):
        lw = 4 if m in mae_best else 1.5
        mae[h][m].plot(ax=axs[0], lw=lw, alpha=0.7)
        lw = 4 if m in rmse_best else 1.5
        rmse[h][m].plot(ax=axs[1], lw=lw, alpha=0.7)

    if rw:
        rw_mae[h].plot(ax=axs[0], lw=6, color='k', alpha=0.7, label='rw')
        rw_rmse[h].plot(ax=axs[1], lw=6, color='k', alpha=0.7, label='rw')

    for ax in axs.ravel():
        ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
        ax.grid()
        if ylim:
            ax.set_ylim(ylim)

    fig.suptitle(f"{title} with h={h}", fontsize=25, y=1.05)
    fig.tight_layout()

    print('(MAE)', end=' ')
    for i in mae_best:
        print(f"{i}: {mae[h][i].iloc[-1]:.3f}", end=' ')
    print('\n(RMSE)', end=' ')
    for i in rmse_best:
        print(f"{i}: {rmse[h][i].iloc[-1]:.3f}", end=' ')
    print(f"\n{'rw'}: {rw_mae[h].iloc[-1]:.3f}", end=' ')
    

#####
def has_s3key():
    s3_key = os.environ.get('FSSPEC_S3_KEY')
    
    if s3_key:
        return True
    else:
        return False


def upload_to_s3(local_path, s3_path):    
    if has_s3key():
        bidas_fs = s3fs.S3FileSystem(anon=False)
        _ = bidas_fs.put(local_path, s3_path)
        

def download_from_s3(s3_path, local_path):
    if has_s3key():
        bidas_fs = s3fs.S3FileSystem(anon=False)
    else:
        bidas_fs = s3fs.S3FileSystem(anon=True)

    _ = bidas_fs.get(s3_path, local_path)
        

def set_local_path():
    current_dir = os.getcwd()
    if os.access(current_dir, os.W_OK):
        local_path = current_dir
    else:
        local_path = os.getenv('HOME')
        
    return local_path

