import pandas as pd
from pandas.tseries.offsets import MonthEnd
from pandas.tseries.offsets import DateOffset

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime
from multiprocessing import Process

from tqdm import tqdm
import itertools

from tqdm import notebook

idx = pd.IndexSlice

# 한글폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

def get_best_ens_models(ens_models):
    ens_best = {0:0, 3:0, 12:0}    
    
    # h=0, 3, 12 일때 best 조합)
    for i, ens in enumerate(ens_models):
        if set(ens) == set(['lm_roll0_lag4_modelF_fb_1:P_adm:P_cpi_1:P_eir:GB_cp_3', 'lm_roll0_lag6_modelRE_atpi_1:P_adm:P_cpi_1:P_eir:GB_cp_3', 'ext_roll0_lag0_d10_g1']):
            # print("h0 == ", i, ens)
            ens_best[0] = i
        if set(ens) == set(['lm_roll0_lag6_modelP_ipi_6:RE_atpi_1:P_cpi_1:P_eir:GB_cp_3','lm_roll0_lag3_modelRE_atpi_1:P_cpi_1:P_eir:GB_cp_3', 'ext_roll0_lag0_d10_g1']):
            # print("h3 == ", i, ens)
            ens_best[3] = i
        if set(ens) == set(['lm_roll0_lag4_modelP_ipi_6:RE_atpi_1:P_cpi_1:GB_cp_3', 'ext_roll0_lag0_d6_g1']):
            # print("h12 == ", i, ens)
            ens_best[12] = i
            
    return ens_best

def nth_friday(year, month, order):
    """Return 1st, 2nd, ... , last friday of given year and month,
    order : 0, 1, 2, ..., -1
    """
    fridays = pd.date_range('2000-01-01', '2050-12-31', freq='W-FRI') # 2000년 1월 1일부터 2050년 12월 31일까지 매주 금요일 날짜 생성
    fridays_of_the_month = [date for date in fridays if (date.year == year) & (date.month == month)] # year, month 인자에 해당하는 월의 금요일 날짜 생성

    try:
        return fridays_of_the_month[order] # 월의 order번째 금요일 날짜 반환
    except:
        print(f'The month has no {order + 1}th friday')
        
def get_forecasting_dates(hor, tm, vintages):
    """예측시계별로 타겟월에 대한 첫 예측 빈티지 시점과 마지막 빈티지 시점을 반환
    
    Arguments:
        hor: 예측시계 (forecasting horizon)
        tm: 타겟  (target month)
    
    Returns:
        m0, m1: 첫 예측 빈티지 시점과 마지막 빈티지 시점
    """
    if hor == 0:
        m0 = tm - DateOffset(months = 1) # tm의 1개월전
        m1 = tm - DateOffset(months = hor) + MonthEnd(0) # MonthEnd(0)는 해당월의 마지막날짜
    else:
        m0 = tm - DateOffset(months = hor + 3) + MonthEnd(0) # hor=3이면 tm의 6개월전, h=12면 tm의 15개월전
        m1 = tm - DateOffset(months = hor) + MonthEnd(0) # hor=3이면 tm의 3개월전, hor=12면 tm의 12개월전
        
    tmp = vintages[nth_friday(m0.year, m0.month, 1) <= vintages] # m0.year, m0.month의 1주차 금요일 이후의 빈티지 시점들
    forecasting_dates = tmp[tmp <= nth_friday(m1.year, m1.month, -1)] # m1.year, m1.month의 마지막 주차 금요일 이전의 빈티지 시점들
    
    return forecasting_dates, m0, m1

def gen_lagged(X, lags):

    temp = X.copy()

    for l in range(1, 1 + lags):
        lX = temp.shift(l)
        lX.columns = ['l' + str(l) + '_' + col for col in temp.columns]
        X = pd.concat([X, lX], axis=1)

    return X

def get_train_data_v5(df0,
                      lag = 0,
                      data_group = 1,
                      excl_alt = True,
                      m1 = None,
                      fillna='ffill',
                      rolling = 0,
                      sm = '2006-01-01',
                      predictors=None
                     ):
    #s3_repo_path = 's3://newtech/public/inf_nowcasting'
    
    df = df0.copy()

    if data_group in [1, 2, 3]:

        vspec = pd.read_excel('input/data_list_all_v9.xlsx', index_col = None)
        vspec = vspec[vspec['My ID'] != 'P_ppi_2'] # P_ppi_2 제외
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
    if m1 is not None:
        df = df.reindex(pd.date_range(df.index[0], m1, freq='M')) # m1까지 인덱스 추가

    # 월중 해당월 기준 통계가 공표되는 변수들(lag0_var_list)은 그대로 예측변수에 포함하고
    # 이외 변수들은 shift(1)하여 예측변수에 포함 (직전월 기준 통계가 공표되는 변수들 lag1_var_list, ...)
    # 예를 들어, 9월중 전망시계 9월 인플레이션(h=0인 타겟변수)를 전망(나우캐스팅)할때, 
    # lag0_var_list 변수들은 9월값(없는 경우 보간된 값), lag1_var_list 변수들은 8월값(없는 경우 보간된 값), .. 등을 이용
    #X = df[lag0_var_list].copy()
    #X = pd.concat([X, df[[col for col in df.columns if col not in lag0_var_list]].shift(1)], axis=1)
    X = df.copy()

    # 결측치 보간 ffill, 2006년 이후 데이터 이용
    if fillna == 'ffill':
        X = X.fillna(method='ffill').loc[sm:] #NaN을 이전값으로 채우고, 2006-01-01이후의 데이터만 사용

    # 예측변수 정규화
    Xm = X.mean()
    Xs = X.std()
    Xn = (X - Xm)/Xs

    # 결측치 보간 at head
    #Xn = Xn.fillna(0)
    Xn = Xn.fillna(method = 'bfill')
    
    # 선형회귀를 위해 예측변수가 주어진 경우
    if predictors is not None:
        Xn = Xn.loc[:, predictors]

    # lag variable 생성
    LX = gen_lagged(Xn, lag)

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
    """빈티지별 예측 오차를 계산
    
    예측값(pred)에서 실제값(act)을 빼서 각 빈티지, 예측시계, 타겟월별 
    예측 오차를 계산
    
    Arguments:
        pred: 예측값 DataFrame
              - index: 빈티지 날짜들
              - columns: MultiIndex(예측시계, 타겟월)
        act: 실제값 Series 또는 DataFrame
             - index: 타겟월들
    
    Returns:
        DataFrame: 예측 오차 (pred - act)
                  - 구조는 pred와 동일
                  - 각 셀은 (빈티지, 예측시계, 타겟월)별 예측 오차
    
    Example:
        >>> pred.columns = MultiIndex([(0, '2025-01'), (3, '2025-01'), ...])
        >>> act.index = ['2025-01', '2025-02', ...]
        >>> err = get_error_by_vintage(pred, act)
        >>> err.loc['2020-01-06', (0, '2025-01')]  # 특정 빈티지의 예측 오차
    """
    err = pred.copy()

    # pred.index = vintages, pred.columns = (hor, targets)
    hors = err.columns.get_level_values(0).unique() # 예측시계(hor) 추출
    cols = err.columns.get_level_values(1).unique() # targets 추출

    for hor, col in itertools.product(hors, cols): # ex) [0, 3, 12], [2025-01-31, 2025-02-28, 2025-03-31, 2025-04-30, 2025-05-31]
        err.loc[:, idx[hor, col]] -= act.loc[col] # 같은 자리에 예측오차를 채우기

    return err

def align_error_by_week(error):

    err = pd.DataFrame()
    mae = pd.DataFrame()
    rmse = pd.DataFrame()

    hors = error.columns.get_level_values(0).unique() # 예측시계(hor) 추출 [0, 3, 12]

    for hor in hors:
        df = error[hor].dropna(how='all', axis=1) # axis=1: 열 방향으로 결측치 제거
        tmp = pd.DataFrame(index=np.arange(-52, 0)) # 주차별 예측 오차

        for col in df.columns: # df.columns = targets
            dfi = df.loc[:, col].dropna()
            dfi = dfi.reset_index(drop=True) # 인덱스 초기화, drop=True: 기존 vintages인덱스 제거
            dfi.index = dfi.index - len(dfi) # 몇주전 예측오차인지를 인덱스로
            tmp = pd.concat([tmp, dfi], axis=1) 

        tmp = tmp.dropna(axis=0, how='all') # 행 방향으로 na 제거
        mae0 = tmp.apply(lambda x: np.mean(np.abs(x)), axis=1).to_frame(hor) # .to_frame(hor): hor을 열 이름으로 추가
        rmse0 = tmp.apply(lambda x: np.sqrt(np.mean(x**2)), axis=1).to_frame(hor) 

        tmp = pd.concat([tmp], axis=1, keys=[hor]) # keys=[hor]: hor을 첫번째 레벨에 멀티인덱스로 추가
        err = pd.concat([err, tmp], axis=1)
        mae = pd.concat([mae, mae0], axis=1)
        rmse = pd.concat([rmse, rmse0], axis=1)

    err = err.sort_index()
    mae = mae.sort_index()
    rmse = rmse.sort_index()

    return err, mae, rmse

def get_pred_last(pred):

    hors = pred.columns.get_level_values(0).unique()

    targets = pred[0].columns # 두번째 레벨의 인덱스가 targets

    pred_last = pd.DataFrame(index=targets, columns=hors)
    for tm in targets:
        for hor in hors:
            pred_last.loc[tm, hor] = pred[hor].loc[:, tm].dropna().iloc[-1] # 빈티지 예측값들 중 마지막값 추출

    return pred_last

def get_mda(pred_last, act, hor = 0, base='act', scale=1, print_result=True, model = ''):

    targets = pred_last.index

    #act_diff = act[targets] - act[targets].shift(hor + 1)
    act_diff = act - act.shift(hor + 1)
    act_diff = act_diff[targets]
    jump_size = scale*act_diff.std()
    big_jumps = np.abs(act_diff) > jump_size

    act_sign = np.sign(act_diff.dropna())
    act_sign = act_sign.iloc[1:]

    if base == 'act':
        #pred_sign = np.sign((pred_last[targets] - act[targets].shift(hor + 1)).dropna())
        pred_sign = np.sign((pred_last[targets] - act.shift(hor + 1)[targets]).iloc[1:])
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

    for file, name in zip(pred_files, model_names): # zip: 두 리스트의 같은 인덱스끼리 쌍으로 묶음

        pred = pd.read_pickle(file)
        pred1 = pd.concat([pred], axis=1, keys=[name])
        PRED = pd.concat([PRED, pred1], axis=1)

    return PRED

def get_eval_last_pred(PRED, model_names, act):

    MAE = pd.DataFrame()
    RMSE = pd.DataFrame()
    PRED_last = pd.DataFrame()

    for name in notebook.tqdm(model_names):

        pred = PRED[name]

        err, mae, rmse = align_error_by_week(get_error_by_vintage(pred, act))
        mae = pd.concat([mae], axis=1, keys=[name])
        rmse = pd.concat([rmse], axis=1, keys=[name])
        MAE = pd.concat([MAE, mae], axis=1)
        RMSE = pd.concat([RMSE, rmse], axis=1)

        pred_last = get_pred_last(pred)
        pred_last = pd.concat([pred_last], axis=1, keys=[name])
        PRED_last = pd.concat([PRED_last, pred_last], axis=1)

    MAE = MAE.reorder_levels([1, 0], axis=1).sort_index(axis=1) # reorder_levels: 레벨 순서 변경, sort_index: 인덱스 정렬
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

    for ax in axs.ravel(): # .ravel(): 1차원 배열로 변환
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
    print(f"\n{'rw mae'}: {rw_mae[h].iloc[-1]:.3f}", end=' ')
    print(f"{'rw rmse'}: {rw_rmse[h].iloc[-1]:.3f}")
    
def plot_rmse_mae(arima_mae, arima_rmse, ens_mae, ens_rmse, lm_mae, lm_rmse, ext_mae, ext_rmse, rw_mae, rw_rmse, 
                  h=0, best=5, good=20, title='ARIMA', ncol=3, loc=3, figsize=(20, 10), 
                  bbox_to_anchor=(1, -0.1), fontsize=12, rw=True, ylim=None):
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    arima_mae_best = arima_mae[h].loc[-1].sort_values().iloc[:best].index.tolist()
    arima_rmse_best = arima_rmse[h].loc[-1].sort_values().iloc[:best].index.tolist()
    
    ens_mae_best = ens_mae[h].loc[-1].sort_values().iloc[:best].index.tolist()
    ens_rmse_best = ens_rmse[h].loc[-1].sort_values().iloc[:best].index.tolist()
    
    lm_mae_best = lm_mae[h].loc[-1].sort_values().iloc[:best].index.tolist()
    lm_rmse_best = lm_rmse[h].loc[-1].sort_values().iloc[:best].index.tolist()
    
    ext_mae_best = ext_mae[h].loc[-1].sort_values().iloc[:best].index.tolist()
    ext_rmse_best = ext_rmse[h].loc[-1].sort_values().iloc[:best].index.tolist()

    arima_mae_good = arima_mae[h].loc[-1].sort_values().iloc[:good].index.tolist()
    arima_rmse_good = arima_rmse[h].loc[-1].sort_values().iloc[:good].index.tolist()

    ens_mae_good = ens_mae[h].loc[-1].sort_values().iloc[:good].index.tolist()
    ens_rmse_good = ens_rmse[h].loc[-1].sort_values().iloc[:good].index.tolist()
    
    lm_mae_good = lm_mae[h].loc[-1].sort_values().iloc[:good].index.tolist()
    lm_rmse_good = lm_rmse[h].loc[-1].sort_values().iloc[:good].index.tolist()
    
    ext_mae_good = ext_mae[h].loc[-1].sort_values().iloc[:good].index.tolist()
    ext_rmse_good = ext_rmse[h].loc[-1].sort_values().iloc[:good].index.tolist()
    
    for m in list(set(arima_rmse_good + arima_mae_good)):
        lw = 4 if m in arima_mae_best else 1.5
        arima_rmse[h][m].plot(ax=axs[0], lw=lw, alpha=0.7, label=r'$ARIMA$', marker='o', markersize=10)
        lw = 4 if m in arima_rmse_best else 1.5
        arima_mae[h][m].plot(ax=axs[1], lw=lw, alpha=0.7, label=r'$ARIMA$', marker='o', markersize=10)
        
    for m in list(set(ens_rmse_good + ens_mae_good)):
        lw = 4 if m in ens_mae_best else 1.5
        ens_rmse[h][m].plot(ax=axs[0], lw=lw, alpha=0.7, label=r'$ENS$', marker='o', markersize=10)
        lw = 4 if m in ens_rmse_best else 1.5
        ens_mae[h][m].plot(ax=axs[1], lw=lw, alpha=0.7, label=r'$ENS$', marker='o', markersize=10)
        
    for m in list(set(lm_rmse_good + lm_mae_good)):
        lw = 4 if m in lm_mae_best else 1.5
        lm_rmse[h][m].plot(ax=axs[0], lw=lw, alpha=0.7, label=r'$Reg$', marker='o', markersize=10)
        lw = 4 if m in lm_rmse_best else 1.5
        lm_mae[h][m].plot(ax=axs[1], lw=lw, alpha=0.7, label=r'$Reg$', marker='o', markersize=10) 

    for m in list(set(ext_rmse_good + ext_mae_good)):
        lw = 4 if m in ext_mae_best else 1.5
        ext_rmse[h][m].plot(ax=axs[0], lw=lw, alpha=0.7, label=r'$EXT$', marker='o', markersize=10)
        lw = 4 if m in ext_rmse_best else 1.5
        ext_mae[h][m].plot(ax=axs[1], lw=lw, alpha=0.7, label=r'$EXT$', marker='o', markersize=10)         

    if rw:
        rw_mae[h].plot(ax=axs[0], lw=6, color='k', alpha=0.7, label=r'$RW$', marker='o', markersize=10)
        rw_rmse[h].plot(ax=axs[1], lw=6, color='k', alpha=0.7, label=r'$RW$', marker='o', markersize=10)

    for ax in axs.ravel():
        ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
        ax.grid()
        if ylim:
            ax.set_ylim(ylim)

    fig.suptitle(f"{title} with h={h}", fontsize=25, y=1.05)
    fig.tight_layout()

    #print('(MAE)', end=' ')
    #for i in mae_best:
    #    print(f"{i}: {mae[h][i].iloc[-1]:.3f}", end=' ')
    #print('\n(RMSE)', end=' ')
    #for i in rmse_best:
    #    print(f"{i}: {rmse[h][i].iloc[-1]:.3f}", end=' ')
    #print(f"\n{'rw'}: {rw_mae[h].iloc[-1]:.3f}", end=' ')
    
def plot_rmse(arima_rmse, ens_rmse, lm_rmse, ext_rmse, rw_rmse, 
              best=5, good=20, title='ARIMA', ncol=3, loc=3, figsize=(20, 10), 
              bbox_to_anchor=(1, -0.1), fontsize=12, rw=True, ylim=None):
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
        
    arima_rmse_best0 = arima_rmse[0].loc[-1].sort_values().iloc[:best].index.tolist()
    arima_rmse_best3 = arima_rmse[3].loc[-1].sort_values().iloc[:best].index.tolist()
    arima_rmse_best12 = arima_rmse[12].loc[-1].sort_values().iloc[:best].index.tolist()
    
    ens_rmse_best0 = ens_rmse[0].loc[-1].sort_values().iloc[:best].index.tolist()
    ens_rmse_best3 = ens_rmse[3].loc[-1].sort_values().iloc[:best].index.tolist()
    ens_rmse_best12 = ens_rmse[12].loc[-1].sort_values().iloc[:best].index.tolist()
    
    lm_rmse_best0 = lm_rmse[0].loc[-1].sort_values().iloc[:best].index.tolist()
    lm_rmse_best3 = lm_rmse[3].loc[-1].sort_values().iloc[:best].index.tolist()
    lm_rmse_best12 = lm_rmse[12].loc[-1].sort_values().iloc[:best].index.tolist()
    
    ext_rmse_best0 = ext_rmse[0].loc[-1].sort_values().iloc[:best].index.tolist()
    ext_rmse_best3 = ext_rmse[3].loc[-1].sort_values().iloc[:best].index.tolist()
    ext_rmse_best12 = ext_rmse[12].loc[-1].sort_values().iloc[:best].index.tolist()

    arima_rmse_good0 = arima_rmse[0].loc[-1].sort_values().iloc[:good].index.tolist()
    arima_rmse_good3 = arima_rmse[3].loc[-1].sort_values().iloc[:good].index.tolist()
    arima_rmse_good12 = arima_rmse[12].loc[-1].sort_values().iloc[:good].index.tolist()

    ens_rmse_good0 = ens_rmse[0].loc[-1].sort_values().iloc[:good].index.tolist()
    ens_rmse_good3 = ens_rmse[3].loc[-1].sort_values().iloc[:good].index.tolist()
    ens_rmse_good12 = ens_rmse[12].loc[-1].sort_values().iloc[:good].index.tolist()
    
    lm_rmse_good0 = lm_rmse[0].loc[-1].sort_values().iloc[:good].index.tolist()
    lm_rmse_good3 = lm_rmse[3].loc[-1].sort_values().iloc[:good].index.tolist()
    lm_rmse_good12 = lm_rmse[12].loc[-1].sort_values().iloc[:good].index.tolist()
    
    ext_rmse_good0 = ext_rmse[0].loc[-1].sort_values().iloc[:good].index.tolist()
    ext_rmse_good3 = ext_rmse[3].loc[-1].sort_values().iloc[:good].index.tolist()
    ext_rmse_good12 = ext_rmse[12].loc[-1].sort_values().iloc[:good].index.tolist()
    
    for m in list(set(arima_rmse_good0)):
        lw = 4 if m in arima_rmse_best0 else 1.5
        arima_rmse[0][m].plot(ax=axs[0], lw=lw, alpha=0.5, label=r'$ARIMA$', marker='o', markersize=10, fontsize = 15)
        axs[0].set_title(r'$h = 0$', fontsize = 20)
        lw = 4 if m in arima_rmse_best3 else 1.5
        arima_rmse[3][m].plot(ax=axs[1], lw=lw, alpha=0.5, label=r'$ARIMA$', marker='o', markersize=10, fontsize = 15)
        axs[1].set_title(r'$h = 3$', fontsize = 20)
        lw = 4 if m in arima_rmse_best12 else 1.5
        arima_rmse[12][m].plot(ax=axs[2], lw=lw, alpha=0.5, label=r'$ARIMA$', marker='o', markersize=10, fontsize = 15)
        axs[2].set_title(r'$h = 12$', fontsize = 20)
        
    for m in list(set(ens_rmse_good0)):
        lw = 4 if m in ens_rmse_best0 else 1.5
        ens_rmse[0][m].plot(ax=axs[0], lw=lw, color='k', alpha=0.9, label=r'$ENS$', marker='o', markersize=10, fontsize = 15)
    for m in list(set(ens_rmse_good3)):
        lw = 4 if m in ens_rmse_best3 else 1.5
        ens_rmse[3][m].plot(ax=axs[1], lw=lw, color='k', alpha=0.9, label=r'$ENS$', marker='o', markersize=10, fontsize = 15)
    for m in list(set(ens_rmse_good12)):
        lw = 4 if m in ens_rmse_best12 else 1.5
        ens_rmse[12][m].plot(ax=axs[2], lw=lw, color='k', alpha=0.9, label=r'$ENS$', marker='o', markersize=10, fontsize = 15)
        
    for m in list(set(lm_rmse_good0)):
        lw = 4 if m in lm_rmse_best0 else 1.5
        lm_rmse[0][m].plot(ax=axs[0], lw=lw, alpha=0.5, label=r'$Reg$', marker='o', markersize=10, fontsize = 15)
    for m in list(set(lm_rmse_good3)):
        lw = 4 if m in lm_rmse_best3 else 1.5
        lm_rmse[3][m].plot(ax=axs[1], lw=lw, alpha=0.5, label=r'$Reg$', marker='o', markersize=10, fontsize = 15)
    for m in list(set(lm_rmse_good12)):
        lw = 4 if m in lm_rmse_best12 else 1.5
        lm_rmse[12][m].plot(ax=axs[2], lw=lw, alpha=0.5, label=r'$Reg$', marker='o', markersize=10, fontsize = 15)

    for m in list(set(ext_rmse_good0)):
        lw = 4 if m in ext_rmse_best0 else 1.5
        ext_rmse[0][m].plot(ax=axs[0], lw=lw, alpha=0.5, label=r'$EXT$', marker='o', markersize=10, fontsize = 15)
    for m in list(set(ext_rmse_good3)):
        lw = 4 if m in ext_rmse_best3 else 1.5
        ext_rmse[3][m].plot(ax=axs[1], lw=lw, alpha=0.5, label=r'$EXT$', marker='o', markersize=10, fontsize = 15)
    for m in list(set(ext_rmse_good12)):
        lw = 4 if m in ext_rmse_best12 else 1.5
        ext_rmse[12][m].plot(ax=axs[2], lw=lw, alpha=0.5, label=r'$EXT$', marker='o', markersize=10, fontsize = 15)

    if rw:
        rw_rmse[0].plot(ax=axs[0], lw=4, color='0.4', alpha=0.9, label=r'$RW$', marker='o', markersize=10)
        rw_rmse[3].plot(ax=axs[1], lw=4, color='0.4', alpha=0.9, label=r'$RW$', marker='o', markersize=10)
        rw_rmse[12].plot(ax=axs[2], lw=4, color='0.4', alpha=0.9, label=r'$RW$', marker='o', markersize=10)

    for ax in axs.ravel():
        ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
        ax.grid()
        if ylim:
            ax.set_ylim(ylim)
    
    fig.suptitle(f"{title}", fontsize=25, y=1.05)
    fig.tight_layout()
    
    plt.savefig('rmse_h_m.png', dpi='figure')

    #print('(MAE)', end=' ')
    #for i in mae_best:
    #    print(f"{i}: {mae[h][i].iloc[-1]:.3f}", end=' ')
    #print('\n(RMSE)', end=' ')
    #for i in rmse_best:
    #    print(f"{i}: {rmse[h][i].iloc[-1]:.3f}", end=' ')
    #print(f"\n{'rw'}: {rw_mae[h].iloc[-1]:.3f}", end=' ')
    
def plot_mae(arima_mae, ens_mae, lm_mae, ext_mae, rw_mae, 
              best=5, good=20, title='ARIMA', ncol=3, loc=3, figsize=(20, 10), 
              bbox_to_anchor=(1, -0.1), fontsize=12, rw=True, ylim=None):
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    arima_mae_best0 = arima_mae[0].loc[-1].sort_values().iloc[:best].index.tolist()
    arima_mae_best3 = arima_mae[3].loc[-1].sort_values().iloc[:best].index.tolist()
    arima_mae_best12 = arima_mae[12].loc[-1].sort_values().iloc[:best].index.tolist()
    
    ens_mae_best0 = ens_mae[0].loc[-1].sort_values().iloc[:best].index.tolist()
    ens_mae_best3 = ens_mae[3].loc[-1].sort_values().iloc[:best].index.tolist()
    ens_mae_best12 = ens_mae[12].loc[-1].sort_values().iloc[:best].index.tolist()
    
    lm_mae_best0 = lm_mae[0].loc[-1].sort_values().iloc[:best].index.tolist()
    lm_mae_best3 = lm_mae[3].loc[-1].sort_values().iloc[:best].index.tolist()
    lm_mae_best12 = lm_mae[12].loc[-1].sort_values().iloc[:best].index.tolist()
    
    ext_mae_best0 = ext_mae[0].loc[-1].sort_values().iloc[:best].index.tolist()
    ext_mae_best3 = ext_mae[3].loc[-1].sort_values().iloc[:best].index.tolist()
    ext_mae_best12 = ext_mae[12].loc[-1].sort_values().iloc[:best].index.tolist()

    arima_mae_good0 = arima_mae[0].loc[-1].sort_values().iloc[:good].index.tolist()
    arima_mae_good3 = arima_mae[3].loc[-1].sort_values().iloc[:good].index.tolist()
    arima_mae_good12 = arima_mae[12].loc[-1].sort_values().iloc[:good].index.tolist()

    ens_mae_good0 = ens_mae[0].loc[-1].sort_values().iloc[:good].index.tolist()
    ens_mae_good3 = ens_mae[3].loc[-1].sort_values().iloc[:good].index.tolist()
    ens_mae_good12 = ens_mae[12].loc[-1].sort_values().iloc[:good].index.tolist()
    
    lm_mae_good0 = lm_mae[0].loc[-1].sort_values().iloc[:good].index.tolist()
    lm_mae_good3 = lm_mae[3].loc[-1].sort_values().iloc[:good].index.tolist()
    lm_mae_good12 = lm_mae[12].loc[-1].sort_values().iloc[:good].index.tolist()
    
    ext_mae_good0 = ext_mae[0].loc[-1].sort_values().iloc[:good].index.tolist()
    ext_mae_good3 = ext_mae[3].loc[-1].sort_values().iloc[:good].index.tolist()
    ext_mae_good12 = ext_mae[12].loc[-1].sort_values().iloc[:good].index.tolist()
    
    for m in list(set(arima_mae_good0)):
        lw = 4 if m in arima_mae_best0 else 1.5
        arima_mae[0][m].plot(ax=axs[0], lw=lw, alpha=0.5, label=r'$ARIMA$', marker='o', markersize=10, fontsize = 15)
        axs[0].set_title(r'$h = 0$', fontsize = 20)
        lw = 4 if m in arima_mae_best3 else 1.5
        arima_mae[3][m].plot(ax=axs[1], lw=lw, alpha=0.5, label=r'$ARIMA$', marker='o', markersize=10, fontsize = 15)
        axs[1].set_title(r'$h = 3$', fontsize = 20)
        lw = 4 if m in arima_mae_best12 else 1.5
        arima_mae[12][m].plot(ax=axs[2], lw=lw, alpha=0.5, label=r'$ARIMA$', marker='o', markersize=10, fontsize = 15)
        axs[2].set_title(r'$h = 12$', fontsize = 20)
        
    for m in list(set(ens_mae_good0)):
        lw = 4 if m in ens_mae_best0 else 1.5
        ens_mae[0][m].plot(ax=axs[0], lw=lw, color='k', alpha=0.9, label=r'$ENS$', marker='o', markersize=10)
    for m in list(set(ens_mae_good3)):
        lw = 4 if m in ens_mae_best3 else 1.5
        ens_mae[3][m].plot(ax=axs[1], lw=lw, color='k', alpha=0.9, label=r'$ENS$', marker='o', markersize=10)
    for m in list(set(ens_mae_good12)):
        lw = 4 if m in ens_mae_best12 else 1.5
        ens_mae[12][m].plot(ax=axs[2], lw=lw, color='k', alpha=0.9, label=r'$ENS$', marker='o', markersize=10)
        
    for m in list(set(lm_mae_good0)):
        lw = 4 if m in lm_mae_best0 else 1.5
        lm_mae[0][m].plot(ax=axs[0], lw=lw, alpha=0.5, label=r'$Reg$', marker='o', markersize=10)
    for m in list(set(lm_mae_good3)):
        lw = 4 if m in lm_mae_best3 else 1.5
        lm_mae[3][m].plot(ax=axs[1], lw=lw, alpha=0.5, label=r'$Reg$', marker='o', markersize=10)
    for m in list(set(lm_mae_good12)):
        lw = 4 if m in lm_mae_best12 else 1.5
        lm_mae[12][m].plot(ax=axs[2], lw=lw, alpha=0.5, label=r'$Reg$', marker='o', markersize=10)

    for m in list(set(ext_mae_good0)):
        lw = 4 if m in ext_mae_best0 else 1.5
        ext_mae[0][m].plot(ax=axs[0], lw=lw, alpha=0.5, label=r'$EXT$', marker='o', markersize=10)
    for m in list(set(ext_mae_good3)):
        lw = 4 if m in ext_mae_best3 else 1.5
        ext_mae[3][m].plot(ax=axs[1], lw=lw, alpha=0.5, label=r'$EXT$', marker='o', markersize=10)
    for m in list(set(ext_mae_good12)):
        lw = 4 if m in ext_mae_best12 else 1.5
        ext_mae[12][m].plot(ax=axs[2], lw=lw, alpha=0.5, label=r'$EXT$', marker='o', markersize=10)

    if rw:
        rw_mae[0].plot(ax=axs[0], lw=4, color='0.4', alpha=0.9, label=r'$RW$', marker='o', markersize=10)
        rw_mae[3].plot(ax=axs[1], lw=4, color='0.4', alpha=0.9, label=r'$RW$', marker='o', markersize=10)
        rw_mae[12].plot(ax=axs[2], lw=4, color='0.4', alpha=0.9, label=r'$RW$', marker='o', markersize=10)

    for ax in axs.ravel():
        ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor)
        ax.grid()
        if ylim:
            ax.set_ylim(ylim)

    fig.suptitle(f"{title}", fontsize=25, y=1.05)
    fig.tight_layout()
    
    plt.savefig('mae_h_m.png', dpi='figure')

    #print('(MAE)', end=' ')
    #for i in mae_best:
    #    print(f"{i}: {mae[h][i].iloc[-1]:.3f}", end=' ')
    #print('\n(RMSE)', end=' ')
    #for i in rmse_best:
    #    print(f"{i}: {rmse[h][i].iloc[-1]:.3f}", end=' ')
    #print(f"\n{'rw'}: {rw_mae[h].iloc[-1]:.3f}", end=' ')