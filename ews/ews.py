"""<데이터 기반 금융˙외환 조기경보모형*을 위한 라이브러리>
* BOK 이슈노트 No.2024-11(김태완, 박정희, 이현창, 2024)

조기경보모형 관련 데이터 입수/변환, 하이퍼파라미터 튜닝 및 평가, 학습 및 예측을 위해 필요한 클래스와 함수로 구성

주요 클래스:
- Bidas: Bidas 데이터를 API나 엑셀 파일로부터 입수하기 위한 클래스
- SignalExtraction: (Scikit-learn 호환) 신호추출법 모형의 구현체
- EarlyWarningModel: 조기경보모형의 학습, 실행, 저장, 로딩을 위한 클래스

주요 함수:
- preprocess: CFPI 구성변수 및 조기경보모형 입력변수를 산정
- get_crises: CFPI가 임계치를 넘는 기간을 기준으로, 위기기간 및 학습데이터의 그룹을 산정
- run_cv: 각 모델에 대해 하이퍼파라미터 튜닝한 후 예측결과를 산출
"""
import pickle
import warnings

from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm.auto import tqdm


# 기본 설정
warnings.filterwarnings(action='ignore')
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 기본 팔레트값
DEFAULT_PALETTE = {
    '자금조달': 'darkred',
    '레버리지': 'red',
    '자산가격': 'orange',
    '금리': 'navy',
    '변동성': 'royalblue',
    '경기': 'darkgreen'
}


# I. 데이터 입수 및 변환

class Bidas:
    """Bidas 시계열 데이터를 API나 엑셀 파일로부터 입수하기 위한 클래스"""

    def __init__(self, source_type='API', api_headers={}, file_name='', file_sheet='data'):
        assert source_type in ['API', 'Excel', 'GoogleDocs']
        self.source_type = source_type
        self.api_headers = api_headers # 필요시 사용자 인증 정보 포함
        self.file_name = file_name # 로컬 엑셀파일인 경우 파일명, 구글닥스인 경우 파일id
        self.file_sheet = file_sheet # 엑셀 시트명
        self.series = {} # Bidas id별 시계열 데이터
        self.freqs = {} # Bidas id별 시계열 데이터 빈도

    def load_series(self, bidas_ids):
        """데이터 소스유형에 따라 Bidas 시계열 데이터를 읽어온다."""
        if self.source_type == 'API':
            self._load_series_from_api(bidas_ids)
        else:
            self._load_series_from_file(bidas_ids)

    def _load_series_from_api(self, bidas_ids):
        """Bidas API를 이용하여 Bidas 시계열 데이터를 읽어온다."""
        api_url = 'http://datahub.boknet.intra/api/v1/obs/lists'
        res = requests.post(api_url, headers=self.api_headers, data={'ids': bidas_ids})
        for raw_data in res.json()['data'][0]:
            try:
                bidas_id = raw_data['series_id']
                obs = pd.DataFrame(raw_data['observations'])
                series = obs['value'].apply(lambda x: np.nan if x == '' else x).set_axis(pd.to_datetime(obs['period'])).dropna()
                series.name = bidas_id
                if series.dtype == 'O':
                    series = series.str.strip().str.replace(',', '').astype(float)
                self.series[bidas_id] = series
                self.freqs[bidas_id] = self._get_freq(series)
            except:
                print(f'{bidas_id} is not loaded.')

    def _load_series_from_file(self, bidas_ids):
        """Bidas 엑셀 플러그인을 통해 미리 데이터를 다운받은 엑셀 파일로부터 Bidas 시계열 데이터를 읽어온다."""
        if self.source_type == 'Excel': # 로컬 엑셀 파일
            # 엑셀에서는 Bidas id(1행)과 메타데이터(2~13행)를 각각의 행으로 인식 → 메타데이터 행 skip
            raw_data = pd.read_excel(self.file_name, header=0, skiprows=range(1, 13), sheet_name=self.file_sheet)
        else: # 구글닥스로 변환한 엑셀 파일
            gd_url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'
            file_url = gd_url.format(self.file_name, self.file_sheet)
            raw_data = pd.read_csv(file_url, header=0, low_memory=False)
            # 구글닥스에서는 Bidas id와 메타데이터를 합쳐서 하나의 행으로 인식 → Bidas id를 분리하여 홀수번째열에 배정
            raw_data.columns = [f'Unnamed: {i}' if i % 2 else column.split(' ')[0] for i, column in enumerate(raw_data.columns)]
        for bidas_id in bidas_ids:
            try:
                # i열에는 Bidas id 및 period가, i+1열에는 각 period별 데이터값이 존재
                index = raw_data.columns.get_loc(bidas_id)
                series = raw_data.iloc[0:, index+1].set_axis(pd.to_datetime(raw_data.iloc[0:, index])).rename_axis('period').dropna()
                series.name = bidas_id
                if series.dtype == 'O':
                    series = series.str.strip().str.replace(',', '').astype(float)
                self.series[bidas_id] = series
                self.freqs[bidas_id] = self._get_freq(series)
            except:
                print(f'{bidas_id} is not loaded.')

    def _get_freq(self, series):
        """입력받은 시계열 데이터의 빈도를 데이터포인트간의 시점차이로 추정한다.

        Args:
            series: 빈도를 추정할 시계열

        Returns:
            freq: 추정된 빈도(D:일별, M:월별, Q:분기별, A:연별)
        """
        freq = None
        interval = pd.to_timedelta(np.diff(series.index).min()).days
        if interval == 1:
            freq = 'D'
        elif 28 <= interval <= 31:
            freq = 'M'
        elif 90 <= interval <= 92:
            freq = 'Q'
        elif 365 <= interval <= 366:
            freq = 'A'
        return freq

    def get_table(self, bidas_ids, freq, range_from=None, range_to=None, downsample='mean', upsample='ffill'):
        """Bidas 시계열 데이터를 지정한 빈도에 맞게 변환하여 제공한다.

        Args:
            bidas_ids: Bidas 시계열 아이디 목록
            freq: 출력 시계열의 빈도 (D:일별, M:월별, Q:분기별, A:연별)
            range_from: 데이터 시작일자(YYYY-MM-DD)
            range_to: 데이터 종료일자(YYYY-MM-DD)
            downsample: {freq}보다 원본 데이터의 빈도가 높을 경우 다운샘플링 방법(e.g. mean, sum, max)
            upsample: {freq}보다 원본 데이터의 빈도가 낮을 경우 업샘플링 방법(e.g. ffill, bfill)

        Returns:
            table: 빈도가 {freq}로 일치된 시계열 데이터 테이블
        """
        new_bidas_ids = [bidas_id for bidas_id in bidas_ids if bidas_id not in self.series]
        if len(new_bidas_ids) > 0:
            self.load_series(new_bidas_ids)
        table = pd.DataFrame()
        intervals = {'D': 1, 'M': 30, 'Q': 90, 'A': 365}
        table_interval = intervals[freq]
        for bidas_id in bidas_ids:
            series_interval = intervals[self.freqs[bidas_id]]
            if series_interval < table_interval:
                ts = self.series[bidas_id][range_from:range_to].resample(freq).agg(downsample).to_period(freq)
            else:
                ts = self.series[bidas_id][range_from:range_to].resample(freq).agg(upsample).to_period(freq)
            table = table.join(ts, how='outer')
        return table


class Transform:
    """시계열 데이터 변환을 위한 utility 클래스"""

    @staticmethod
    def scale(x):
        """샘플의 평균과 표준편차로 표준화"""
        result = (x - x.mean())/x.std()
        return result

    @staticmethod
    def link(x):
        """단절된 시계열을 앞선 시계열의 가중치를 조정하여 연결"""
        prev_weight = 1
        prev_result = x.iloc[:, 0]
        for i in range(len(x.columns)-1):
            link = x.iloc[:, i].dropna().index.min()
            weight = prev_weight = prev_weight * x.iloc[:, i].loc[link] / x.iloc[:, i+1].loc[link]
            result = prev_result = pd.concat([x.iloc[:, i+1].loc[:link].iloc[:-1] * weight, prev_result.loc[link:]])
        return result.to_frame()

    @staticmethod
    def cmax(ts):
        """CMAX 계산"""
        result = (-1*ts/ts.rolling(24).max())
        return result

    @staticmethod
    def beta(x):
        """beta 계산"""
        ret = pd.DataFrame({'ts': x.iloc[:, 0], 'base': x.iloc[:, 1]}).pct_change(12) * 100
        cov = ret.rolling(12).cov().unstack()['base']['ts']
        var = ret.rolling(12).var()['base']
        ret['beta'] = cov.div(var, axis=0)
        result = ret.apply(lambda x: x.beta if x.beta >= 1 and x.ts < x.base else 0, axis=1).to_frame()
        return result

    @staticmethod
    def mvol(ts, horizon=6):
        """변동성(이동평균표준편차) 계산"""
        result = ts.pct_change().rolling(horizon).std()
        return result

    @staticmethod
    def gvol(ts, horizon=12, min_sample=30, recursive=False):
        """GARCH 변동성 계산"""
        min_sample = 30 if recursive else len(ts)
        scale = 100 if recursive else 1
        ts_diff = ts.pct_change().bfill() * scale
        for i in range(len(ts)-min_sample+1):
            garch_model = arch_model(ts_diff.iloc[:i+min_sample], mean='AR',
                                     lags=horizon, vol='Garch', p=1, q=1, rescale=False)
            result = garch_model.fit(update_freq=10, disp='off').conditional_volatility * 100
            results = result if i == 0 else pd.concat([results, result.iloc[-1:]])
        return results


def preprocess(data):
    """CFPI 구성변수 및 조기경보모형 입력변수를 산정한다.

    Args:
        data: 사전정의된 메타데이터에 따라 수집한 Bidas의 시계열 데이터 테이블

    Returns:
        data: Bidas 데이터를 바탕으로 새로이 산정한 변수를 포함한 데이터
    """
    # 0. 공통 변수
    # CD 스프레드(CD수익률 - 통안증권 수익률)
    data['cd_sp'] = data['cd91'] - data['ms91']
    # (마이너스) 기간 프리미엄(국채3년물 - 통안증권1년물)
    data['tp_sp_neg'] = -(data['kb3y'] - data['ms1y'])
    # 은행업 지수(KOSPI 은행업지수 + KRX 은행업지수)
    data['post_kosbank'] = data['krxbank']['2020-06':] # KOSPI 만료(2022.06) 2년전을 시작점으로 설정
    data['stockbank'] = data[['post_kosbank', 'kosbank']].transform(Transform.link)

    # 1. CFPI 구성변수
    # 은행업 지수 변동성 (GARCH)
    data['bank_gv'] = data['stockbank'].transform(Transform.gvol)
    # 회사채 스프레드(회사채AA - 국채3년)
    data['cr_sp'] = data['cbaa3y'] - data['kb3y']
    # (마이너스) 주식 수익률 (KOSPI 종가)
    data['stock_ret'] = -data['kospi'].pct_change(12)
    # 주식 변동성 (GARCH)
    data['stock_gv'] = data['kospi'].transform(Transform.gvol)
    # 미달러 환율 변동성 (GARCH)
    data['er_gv'] = data['er'].transform(Transform.gvol)

    # 2. 조기경보모형 입력변수
    # (마이너스) 단기외채 / 외환보유액 비율
    data['res_sdebt'] = data['reserve']/data['short_ex_debt']
    data['res_sdebt_neg'] = -data['res_sdebt']
    # 은행 레버리지 비율 12개월 차분
    data['bank_lev'] = data['bank_asset']/(data['bank_capital'])
    data['bank_lev_diff'] = data['bank_lev'].diff(12)
    # 은행 예대율 12개월 차분
    data['bank_ldr'] = (data['bank_loan'] / data['bank_dep'])*100
    data['bank_ldr_diff'] = data['bank_ldr'].diff(12)
    # 가계신용 (hc1975 + hc2002 + hc2008)
    data['hc1975'] = data[['hc1975_lbond', 'hc1975_sbond', 'hc1975_loan', 'hc1975_gov']].dropna().sum(1)
    data['hc2002'] = data[['hc2002_bond', 'hc2002_loan', 'hc2002_gov']].dropna().sum(1)
    data['hc2008'] = data[['hc2008_bond', 'hc2008_loan', 'hc2008_gov']].dropna().sum(1)
    data['hc2'] = data[['hc2008', 'hc2002', 'hc1975']].transform(Transform.link)
    # GDP대비 가계신용 12개월 차분
    data['hc2_gdp'] = (data['hc2']/data['gdp'].rolling(12).sum()*3).rolling(3).mean()
    data['hc_gdp_diff'] = data['hc2_gdp'].diff(12)
    # 기업신용(cc1975 + cc2002 + cc2008)
    data['cc1975'] = data[['cc1975_lbond', 'cc1975_sbond', 'cc1975_loan', 'cc1975_gov']].dropna().sum(1)
    data['cc2002'] = data[['cc2002_bond', 'cc2002_loan', 'cc2002_gov']].dropna().sum(1)
    data['cc2008'] = data[['cc2008_bond', 'cc2008_loan', 'cc2008_gov']].dropna().sum(1)
    data['cc'] = data[['cc2008', 'cc2002', 'cc1975']].transform(Transform.link)
    # GDP대비 기업신용 12개월 차분
    data['cc_gdp'] = (data['cc']/data['gdp'].rolling(12).sum()*3).rolling(3).mean()
    data['cc_gdp_diff'] = data['cc_gdp'].diff(12)
    # KB 주택매매가격지수 3개월 이동평균 백분율 변화량
    data['kb_hp_pchg'] = data['kb_hp'].rolling(3).mean().pct_change()
    # CD 스프레드 12개월 차분
    data['cd_sp_diff'] = data['cd_sp'].diff(12)
    # CP 스프레드 12개월 차분
    data['cp_sp'] = data['cp91'] - data['call']
    data['cp_sp_diff'] = data['cp_sp'].diff(12)
    # 국가신용 스프레드 12개월 차분
    data['sr_sp'] = data['kb10y'] - data['ub10y']
    data['sr_sp_diff'] = data['sr_sp'].diff(12)
    # (마이너스) 기간 프리미엄 스프레드 12개월 차분
    data['tp_sp_neg_diff'] = data['tp_sp_neg'].diff(12)
    # KOSPI 200 지수 CMAX
    data['stock_cmax'] = data['kospi2'].transform(Transform.cmax)
    # 은행업 변동성
    data['bank_mv'] = data['stockbank'].transform(Transform.mvol)
    # 미달러 환율 변동성
    data['er_mv'] = data['er'].transform(Transform.mvol)
    # (마이너스) GDP 성장률
    data['gdp_growth_neg'] = -data['gdp_growth']

    return data


def get_crises(cfpi, k=1, horizon=6, group_bgn_ext=3, group_end_ext=3):
    """CFPI가 임계치를 넘는 기간을 기준으로, 위기기간 및 학습데이터의 그룹을 산정한다.

    Args:
        k: 위기 임계치(CFPI 표준편차의 배수)
        horizon: 위기기간 앞쪽에서 위기에 확장하여 포함할 기간(개월)
        group_bgn_ext: 확장된 위기기간 앞쪽에서 위기와 동일 그룹에 포함할 기간(개월)
        group_end_ext: 확장된 위기기간 뒷쪽에서 위기와 동일 그룹에 포함할 기간(개월)

    Returns:
        crises: 기본/확장/디레버리징(term/ext_term/post_term)위기 기간 및 학습그룹(group)정보
    """
    crises = cfpi.rename('cfpi').to_frame()
    # {k} 표준편차 이상의 CFPI에 대해 위기기간으로 지정
    crises['term'] = (cfpi > cfpi.std()*k).astype(int)
    # 긱 위기기간의 직전 {horizon}개월을 확장 위기기간으로 지정 (예측 시계 확장)
    crises['ext_term'] = crises['term'][::-1].rolling(horizon+1, min_periods=1).max()[::-1].astype(int)
    # 각 위기기간은 학습/예측시에 나눌 수 없는 하나의 그룹으로, 나머지는 각 데이터 포인트를 별개의 그룹으로 지정
    crises['groups'] = crises['ext_term'].rolling(2, min_periods=1).apply(lambda x: 0 if sum(x) == 2 else 1).cumsum()
    # 각 위기기간에서, CFPI가 정점을 찍은 후 내려오는 디레버리징 기간을 구분 (학습시 선택적으로 제외)
    is_post_term = lambda x: ((x > cfpi.std()*k) & (x.cummax() == x.max()) & (x.cummax() == x.cummax().shift(1))) * 1
    crises['post_term'] = crises.groupby('groups')['cfpi'].apply(is_post_term).set_axis(crises.index)
    # 각 확장 위기기간에 대해 이전 {group_bgn_ext}개월과 이후 {group_end_ext}개월을 동일 그룹으로 지정 (버퍼)
    if group_bgn_ext + group_end_ext > 0:
        get_group_scope = [lambda x: min(x.index)-group_bgn_ext, lambda x: max(x.index)+group_end_ext]
        group_scope = crises[crises['ext_term'] == 1].groupby('groups')['ext_term'].agg(get_group_scope)
        prev_scope = crises.index.min()
        for group, scope in group_scope.iterrows():
            assert prev_scope < scope[0]
            crises.loc[scope[0]:scope[1], 'groups'] = group
            prev_scope = scope[1]
    return crises


def plot_cfpi(cfpi, gdp_growth, k, horizon=6, xlim=['1999-01', '2024-01']):
    """CFPI를 위기기간, GDP성장률과 함께 차트에 그린다.

    Args:
        cfpi: CFPI 데이터
        gdp_growth: GDP성장률 데이터
        k: 위기기간 산정을 위한 CFPI 임계치
        horizon: 위기기간 예측 시계(월)
        xlim: 표시 기간
    """
    crises = get_crises(cfpi, k, horizon)
    _, ax = plt.subplots()
    # CFPI 및 위기기간 차트 표시
    render_crises(crises, ax, True)
    # GDP성장률 및 국소최저점 차트 표시
    render_gdp_drop(gdp_growth, freq='M', ax=ax)
    # x축 설정
    ax.tick_params(axis='x', labelsize=15, which='both')
    ax.set_xlabel('')
    # y축(이중축) 설정
    ax.tick_params(axis='y', labelsize=15)
    ax_right = ax.twinx()
    ax_right.tick_params(axis='y', labelsize=15)
    ax_right.set_ylim(ax.get_ylim())
    if xlim is not None:
        ax.set_xlim(xlim)


def render_crises(crises, ax=None, cfpi_plot=False, crises_plot=True):
    """CFPI 및 위기기간을 차트에 표시한다.

    Args:
        crises: 위기기간 데이터
        ax: 차트축(Axes) 개체
        cfpi_plot: CFPI 표시 여부
        crises_plot: 위기기간 표시 여부
    """
    cfpi = crises.cfpi
    if cfpi_plot:
        # CFPI 라인차트
        if ax is None:
            ax = cfpi.plot(color='black')
        else:
            cfpi.plot(ax=ax, color='black')
        # y={0, CFPI임계치} 기준선 표시
        ax.axhline(y=0, linestyle='-', color='black')
        ax.axhline(y=crises[crises.term == 1].cfpi.min(), linestyle=':', color='black')
    if ax is None:
        _, ax = plt.subplots()
    if crises_plot:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax])
        # 예측시계(red), 위기기간중 상승(red+red)/하강(red+red+yellow) 구간을 색의 중첩으로 구분
        ax.fill_between(cfpi.index, ymin, ymax, color='red', where=crises.ext_term, alpha=0.1)
        ax.fill_between(cfpi.index, ymin, ymax, color='red', where=crises.term, alpha=0.2)
        ax.fill_between(cfpi.index, ymin, ymax, color='yellow', where=crises.post_term, alpha=0.1)


def render_gdp_drop(gdp_growth, freq='Q', ax=None, hgrid=False):
    """GDP 성장률 및 국소최저점(local minima)을 차트에 표시한다.

    Args:
        gdp_growth: GDP성장률 데이터
        freq: 표시할 빈도의 단위(M, Q)
        ax: 차트축(Axes) 개체
        hgrid: y=0축 표시 여부
    """
    # 저점이 과거 1년 평균보다 1표준편차 이상 떨어진 경우 local minima로 설정
    loc_min = lambda x, t=0, q=4: x[(x.shift(1) > x) & (x.shift(-1) > x) &
                                    (t > x) & (x.rolling(q).mean() - x.std() > x)]
    if ax is None:
        ax = gdp_growth.resample(freq).ffill().plot()
    if hgrid:
        ax.axhline(y=0, linestyle=':')
    # GDP성장률이 국소최저점을 지나는 시점마다 차트에 세로축 표시
    gdp_drops = gdp_growth.resample('Q').mean().agg(loc_min).index
    for date in gdp_drops.asfreq(freq):
        ax.axvline(x=date, color='k', linestyle=':', linewidth=3)


# II. 조기경보모형 하이퍼파라미터 튜닝 및 평가

class SignalExtraction(BaseEstimator, ClassifierMixin):
    """(Scikit-learn 호환) 신호추출법 모형의 구현체"""

    def __init__(self, significance=0.75):
        # 허용 가능한 NSR의 최대값 설정
        self.significance = significance

    def fit(self, X, y, **kwargs):
        """주어진 데이터셋을 학습한다."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.cutoffs = []
        self.weights = []
        # 입력변수별로 적용할 임계치와 가중치를 산정
        for var in range(X.shape[1]):
            # NSR을 최소화하는 변수값의 임계치(cutoff) 도출
            var_values = np.unique(X[:, var])
            nsrs = [get_perf(y, X[:, var], var_value)['nsr'] for var_value in var_values]
            cutoff = np.nanargmin(nsrs)
            self.cutoffs.append(var_values[cutoff])
            # 최소 NSR이 {significance} 이하이면 NSR의 역수를 가중치로 사용, 이상이면 무시
            self.weights.append(1/nsrs[cutoff] if (nsrs[cutoff] < self.significance) else 0)
        norm = sum(self.weights)
        self.weights = [weight / norm for weight in self.weights]
        return self

    def predict(self, X):
        """고정값(0.5)을 기준으로 예측값을 0, 1로 나눈다."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5) * 1

    def predict_proba(self, X):
        """변수별로 각 임계치를 넘는 값에 대해 각 가중치를 곱하고 합하여 예측값을 산정한다."""
        check_is_fitted(self)
        X = check_array(X)
        proba = ((X > self.cutoffs) * self.weights).sum(axis=1)
        return np.stack([1 - proba, proba], 1)


def get_perf(Y, Y_pred, threshold=0.5, calc_auc=False):
    """이진 분류 모델의 예측값과 실제값을 비교하여 예측성능을 평가한다.

    Args:
        Y: 분류 실제값
        Y_pred: 분류 예측값
        threshold: 분류 임계치
        calc_auc: AUC 점수 계산 여부

    Returns:
        perf: 성능평가기준(acc, nsr, f1, auc 등)별 수치
    """
    # 입력받은 분류 임계치를 기준으로 confusion matrix 구성
    Actl = np.array(Y, dtype='bool')
    Pred = Y_pred >= threshold
    tp = np.logical_and( Actl,  Pred).sum()
    tn = np.logical_and(~Actl, ~Pred).sum()
    fp = np.logical_and(~Actl,  Pred).sum()
    fn = np.logical_and( Actl, ~Pred).sum()
    perf = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    # 정확도(accuracy)
    perf['acc'] = float(tp + tn) / float(tp + tn + fp + fn) if ((tp + tn + fp + fn > 0))  else np.NaN
    # 재현율/민감도(recall, sensitivity) = true positive rate = (1 - false negative rate)
    perf['tpr'] = float(tp) / float(tp + fn)        if ((tp + fn > 0))  else np.NaN
    # 1종오류율 = false positive rate = (1 - true negative rate)
    perf['fpr'] = 1 - float(tn) / float(tn + fp)    if (tn + fp > 0)    else np.NaN
    # NSR(Noise-to-Signal) 비율 = 1종오류율 / 재현율 = 1종오류율 / (1 - 2종오류율)
    perf['nsr'] = perf['fpr'] / perf['tpr']         if ((perf['fpr'] > 0) & (perf['tpr'] > 0)) else np.NaN
    # F1 점수 = 정밀도(TP/(TP+FP))와 재현율의 조화평균
    perf['f1']  = 2 * tp / (2 * tp + fp + fn)       if (tp + fp + fn > 0)    else np.NaN
    # ROC-AUC(Receiver Operating Characteristic - Area Under the Curve) 점수
    if calc_auc:
        perf['auc'] = roc_auc_score(Actl, Y_pred)   if ((tp + fn > 0) & (tn + fp > 0)) else np.NaN
    return perf


def run_cv(models, model_param_grids, X, y, exclude_post_term=True):
    """각 모델에 대해 하이퍼파라미터 튜닝한 후 예측결과를 산출한다.

    Args:
        models: (Scikit-learn 호환) 모델 목록
        model_param_grids: 하이퍼파라미터의 grid 탐색을 위한 사전 설정값 목록
        X: 입력변수
        y: 라벨 데이터
        exclude_post_term: 위기기간중 하강구간의 학습데이터 제외 여부
    """
    train_exclusion = y.post_term if exclude_post_term else None
    # 교차검증용 학습/테스트 데이터셋 생성
    folds = load_folds(X, y.ext_term, y.groups, train_exclusion)
    # 그리드 탐색을 통해 모델별 최적 하이퍼파라미터 산출
    best_params = grid_search_folds(folds, models, model_param_grids)
    print('best_params = ', best_params)
    # 최적 하이퍼파라미터를 바탕으로 모델별 학습 및 예측
    results = train_and_test_folds(folds, models, best_params)
    # 모델별 예측성능을 표시
    summarize_results(results, models)
    return best_params, results


def load_folds(X, y, groups, train_exclusion=None, n_splits=5, bootstrap=True):
    """stratified group K-fold 교차검증을 위한 학습/테스트 데이터셋을 생성한다.

    Args:
        X: 입력변수
        y: 라벨 데이터
        groups: 교차검증 그룹 id 목록
        train_exclusion: 데이터포인트별 학습 데이터 제외여부(0:포함, 1:제외) 목록
        n_splits: 교차검증 폴드 개수
        bootstrap: 부트스트래핑 여부

    Returns:
        folds: 생성된 교차검증 데이터셋
    """
    folds = []
    splitter = StratifiedGroupKFold(n_splits=n_splits)
    for (train_idx, test_idx) in splitter.split(X, y, groups):
        # 위기기간중 디레버리징 구간 등 학습에서 제외할 구간이 있으면 제외
        if train_exclusion is not None:
            train_idx = np.array([idx for idx in train_idx if train_exclusion[idx] == 0])
        # 학습데이터의 class imbalance 완화를 위해 minor class 데이터를 부트스트래핑/업샘플링
        if bootstrap:
            pos_idx = train_idx[y[train_idx] == 1]
            neg_idx = train_idx[y[train_idx] == 0]
            major_cls = neg_idx if len(neg_idx) > len(pos_idx) else pos_idx
            minor_cls = pos_idx if len(neg_idx) > len(pos_idx) else neg_idx
            train_idx = np.concatenate((np.random.choice(major_cls, size=len(major_cls), replace=False),
                                        np.random.choice(minor_cls, size=len(major_cls), replace=True)))
        fold = {}
        # 학습/테스트 데이터셋별로 표준화
        scaler = StandardScaler()
        fold['train_X'] = scaler.fit_transform(X.iloc[train_idx])
        fold['train_y'] = y[train_idx]
        fold['train_groups'] = groups[train_idx] # 중첩 교차검증시 필요
        fold['test_X'] = scaler.transform(X.iloc[test_idx])
        fold['test_y'] = y[test_idx]
        fold['test_idx'] = test_idx # 데이터포인트별 테스트 결과 저장시 필요
        folds.append(fold)
    return folds


def grid_search_folds(folds, models, model_param_grids, scoring='roc_auc'):
    """stratified group K-fold 교차검증으로 모델별 성능평가와 최적 하이퍼파라미터 탐색을 수행한다.

    Args:
        folds: 교차검증 데이터셋
        models: (Scikit-learn 호환) 모델 목록
        model_param_grids: 하이퍼파라미터의 grid 탐색을 위한 사전 설정값 목록
        scoring: 하이퍼파라미터 튜닝시 성능평가 기준(e.g. roc_auc, f1, accuracy)

    Returns:
        best_params: 각 모델별 최적 하이퍼파라미터값
    """
    best_params = {}
    # 각 모델의 성능을 평가
    for model in tqdm(models):
        best_param = {}
        best_score = 0
        classifier = models[model]
        for f, fold in enumerate(folds):
            # 각 폴드내 중첩(nested) 교차검증을 통한 하이퍼파라미터 튜닝
            splitter = StratifiedGroupKFold(n_splits=4)
            cv_inner = splitter.split(fold['train_X'], fold['train_y'], fold['train_groups'])
            grid = GridSearchCV(classifier, cv=cv_inner, param_grid=model_param_grids[model],
                                scoring=scoring, verbose=False, error_score='raise')
            grid.fit(fold['train_X'], fold['train_y'])
            print('%s %d - Best Score: %f, Best Params: %s' % (model, f, grid.best_score_, grid.best_params_))
            if grid.best_score_ > best_score:
                best_param = grid.best_params_
                best_score = grid.best_score_
        best_params[model] = best_param
    return best_params


def train_and_test_folds(folds, models, model_params=None):
    """하이퍼파라미터를 고정한 모델별로 교차검증 데이터셋에 대해 학습 및 테스트를 수행한다."""
    results = pd.DataFrame(index=np.arange(np.sum([len(fold['test_y']) for fold in folds])),
                           columns=(['fold', 'actl'] + [model for model in models]))
    for f, fold in enumerate(folds):
        results['fold'].iloc[fold['test_idx']] = f
        results['actl'].iloc[fold['test_idx']] = fold['test_y']
    for model in tqdm(models):
        classifier = models[model]
        if model_params is not None:
            model_param = model_params[model]
            classifier.set_params(**model_param)
        for fold in folds:
            # 학습
            sample_weight = compute_sample_weight('balanced', fold['train_y'])
            classifier.fit(fold['train_X'], fold['train_y'], sample_weight=sample_weight)
            # 테스트
            prob = classifier.predict_proba(fold['test_X'])
            results[model].iloc[fold['test_idx']] = prob[:, 1] if len(prob.shape) > 1 else [prob[1]]
    return results


def summarize_results(results, models):
    """분류예측결과를 입력받아 모델별 예측성능을 계산하여 표시한다."""
    perfs = []
    for model in models:
        missing = results[model].isna()
        perf = get_perf(results[~missing]['actl'], results[~missing][model], calc_auc=True)
        perfs.append(perf)
    print(pd.DataFrame(perfs, index=models))


def plot_roc_curve(preds):
    """각 모델별 분류예측 결과를 바탕으로 ROC 곡선을 그린다.

    Args:
        results: 각 컬럼은 fold(폴드 번호), actl(라벨값)을 제외하고 모델명(예측값)으로 구성
    """
    model_names = [model_name for model_name in preds.columns if model_name not in ['fold', 'actl']]
    # 범례에 표시할 모델별 ROC-AUC점수를 fold별로 집계/평균하여 산정
    agg_auc = lambda x: pd.DataFrame([roc_auc_score(list(x.actl), x[model]) for model in model_names],
                                     index=model_names).T
    agg_perf = pd.DataFrame(preds.groupby('fold').apply(agg_auc).mean()).T
    # ROC 곡선 표시
    actl = preds['actl'].to_list()
    plt.figure(figsize=(6, 6))
    for model in model_names:
        fpr, tpr, _ = roc_curve(actl, preds[model])
        plt.plot(fpr, tpr, lw=3, alpha=0.5, label='%s (avg. auc=%0.2f)' % (model, agg_perf[model]))
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc='lower right', fontsize='small', prop={'size': 14}, handlelength=2, handletextpad=0.5, labelspacing=0.5)
    plt.show()


# III. 조기경보모형의 활용

class EarlyWarningModel:
    """조기경보모형의 학습, 실행, 저장, 로딩을 위한 클래스"""

    def __init__(self, model=None):
        self.model = model
        self.scaler = StandardScaler()

    def train(self, X, y):
        """학습데이터를 기준으로 scaler와 model의 파라미터를 업데이트한다."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def scale(self, X):
        """학습한 데이터와 동일한 기준으로 신규 데이터를 표준화한다."""
        return self.scaler.transform(X)

    def predict(self, X, decompose=False):
        """학습된 모델로 신규 데이터를 예측하고, 변수별 기여도를 분해한다.

        Args:
            X: 입력 변수
            decompose: 기여도 분해 여부

        Returns:
            results: 예측결과
            impacts: 변수별 기여도 분해 결과
        """
        results = pd.DataFrame(columns=['period', 'pred']).set_index('period')
        impacts = pd.DataFrame(columns=['period', 'variable', 'impact']).set_index(['period', 'variable'])
        X_scaled = self.scale(X)
        for time_idx, time in enumerate(X.index):
            X_original = X_scaled[time_idx, :].reshape(1, -1)
            original_value = self.model.predict_proba(X_original)[:, 1][0]
            period = pd.to_datetime(time.strftime('%Y-%m'))
            results.loc[period] = original_value
            # 변수별 기여도 분해
            if decompose and time_idx > 0:
                for var_idx, var in enumerate(X.columns):
                    X_modified = np.copy(X_original)
                    X_modified[0, var_idx] = X_scaled[time_idx-1, var_idx]
                    modified_value = self.model.predict_proba(X_modified)[:, 1][0]
                    impact = original_value - modified_value
                    impacts.loc[(period, var), :] = impact
        return results, impacts

    def save(self, id=None):
        """학습된 model과 scaler를 저장한다."""
        with open(f'{id}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{id}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, id=None):
        """학습된 model과 scaler를 불러온다."""
        with open(f'{id}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{id}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)


def plot_predicted(results, crises=None, perc70=None, perc90=None,
                   line_styles=['-', '--', '-.', '.']):
    """조기경보모형의 예측결과를 차트에 표시한다.

    Args:
        results: 예측결과
        crises: 위기기간 데이터(지정한 경우만 표시)
        perc70: 기준 예측치의 70분위 수치(지정한 경우만 표시)
        perc90: 기준 예측치의 90분위 수치(지정한 경우만 표시)
        line_styles: 모델이 여러개인 경우 차트를 구분하기 위한 라인 스타일
    """
    _, ax = plt.subplots()
    # 모델별 라인차트
    for i, (model_name, result) in enumerate(results.items()):
        ax.plot(result.index, result, label=model_name, color='black',
                linestyle=line_styles[i], linewidth=2)
    # x축 설정
    ax.set_xlim(result.index.min(), result.index.max())
    ax.set_xticklabels([]) # 원래 코드 아래는 수정부분
    

    
    # y축 설정
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='y', labelsize=15)
    # 범례 설정
    ax.legend(loc='upper left', frameon=False, fontsize=18)
    # 위기기간 표시
    if crises is not None:
        render_crises(crises, ax=ax)
    # 70분위, 90분위 기준선 표시
    if perc70 is not None:
        ax.axhline(perc70, color='dimgrey', linestyle='--', lw=1.5)
    if perc90 is not None:
        ax.axhline(perc90, color='dimgrey', linestyle=':', lw=1.5)


def plot_decomposed(impacts, feature_ids, feature_groups,
                    palette=DEFAULT_PALETTE, bar_width=0.4, legend_row=2, legend_col=3):
    """조기경보모형의 예측결과를 바탕으로 변수별 기여도를 차트에 표시한다.

    Args:
        impacts: 변수별 기여도 데이터
        feature_ids: 입력 변수명 목록
        feature_groups: 입력 변수 그룹명 목록
        palette: 변수 그룹별 색상
        bar_width: 단일 바 넓이
        legend_row: 범례 행 개수
        legend_col: 범례 열 개수
    """
    _, ax = plt.subplots(figsize=(9, 4))
    periods = impacts.index.get_level_values(0).unique()
    # 바차트 - 각 일자/변수 그룹별로 양/음의 영향도를 더하여 표시
    for period_idx, period in enumerate(periods):
        bottom_pos = 0
        bottom_neg = 0
        for feature_id, feature_group in zip(feature_ids, feature_groups):
            impact = impacts.loc[(period, feature_id)][0]
            if impact > 0:
                ax.bar(period_idx+0.5, impact, bottom=bottom_pos, color=palette[feature_group], width=bar_width)
                bottom_pos += impact
            else:
                ax.bar(period_idx+0.5, impact, bottom=bottom_neg, color=palette[feature_group], width=bar_width)
                bottom_neg += impact
    # x축 설정
    ax.set_xlim([0 - bar_width/2, len(periods) - bar_width/2])
    ax.set_xticks([i+0.5 for i in range(len(periods))])
    xticklabels = [period.strftime('%b\n%y') if period.month in [3, 6, 9, 12] else '' for period in periods]
    ax.set_xticklabels(xticklabels, fontsize=18)
    # y축 설정
    y_top = impacts[impacts >= 0].groupby('period').sum().max()[0]
    y_bottom = impacts[impacts < 0].groupby('period').sum().min()[0]
    ax.set_ylim([max(y_bottom*1.2, -1), min(y_top*1.2, 1)])
    ax.tick_params(axis='y', labelsize=18)
    # y=0 기준선 표시
    ax.axhline(0, color='black', linewidth=1)
    # 범례 설정
    labels = np.array([label for label in palette]).reshape(legend_row, legend_col).T.flatten()
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[label]) for label in labels]
    ax.legend(handles, labels, loc='lower center', frameon=False, ncol=legend_col,
              bbox_to_anchor=(0.5, -0.55), fontsize=18, columnspacing=0.5)


def plot_pdp_oneway(model, x, feature_names, top=15, n_cols=3, width=15, height=25):
    """partial dependency plot을 변수별로(one-way) 표시한다.

    Args:
        model: 조기경보모형
        x: 입력 변수
        feature_names: 입력 변수명 목록
        top: 중요도순으로 표시할 변수 개수
        n_cols: 차트 열 개수
        width: 차트 넓이
        height: 차트 높이
    """
    top_features = np.argsort(model.feature_importances_)[-top:]
    # 전체 pdp를 한번에 표시
    pdp = PartialDependenceDisplay.from_estimator(
        model, x, n_cols=n_cols, features=top_features, feature_names=feature_names,
        kind='both', ice_lines_kw={'color': 'gray', 'alpha': 0.3}, pd_line_kw={'color': 'red'})
    pdp.figure_.set_figwidth(width)
    pdp.figure_.set_figheight(height)
    # 개별 차트의 제목 설정 및 축라벨/범례 제거
    nrows = np.ceil(len(top_features) / n_cols).astype(int)
    for row in range(nrows):
        for col in range(n_cols):
            pdp.axes_[row][col].set_title(pdp.axes_[row][col].get_xlabel())
            pdp.axes_[row][col].set_ylim([0.0, 1.0])
            pdp.axes_[row][col].set_xlabel('')
            pdp.axes_[row][col].set_ylabel('')
            legend = pdp.axes_[row][col].get_legend()
            legend.set_visible(False)


def plot_pdp_twoway(model, x, feature_names, feature_types,
                    top=6, n_cols=2, width=16, height=30):
    """partial dependency plot을 변수의 쌍으로(two-way) 표시한다.

    Args:
        model: 조기경보모형
        x: 입력 변수
        feature_names: 입력 변수명 목록
        feature_types: 입력 변수 유형(취약성, 트리거) 목록
        top: 중요도순으로 표시할 변수 개수
        n_cols: 차트 열 개수
        width: 차트 넓이
        height: 차트 높이
    """
    # 각 그룹(취약성, 트리거)별 변수의 조합을 생성
    top_features = np.argsort(model.feature_importances_)[-top:]
    top_vul_features = [idx for idx in top_features if feature_types[idx]=='취약성']
    top_trg_features = [idx for idx in top_features if feature_types[idx]=='트리거']
    top_feature_combinations = [(vul, trg) for vul in top_vul_features for trg in top_trg_features]
    # 각 행/열별 subplot으로 pdp를 표시
    nrows = np.ceil(len(top_feature_combinations) / n_cols).astype(int)
    fig, axes = plt.subplots(nrows, n_cols, figsize=(width, height))
    for i, (vul_idx, trg_idx) in enumerate(top_feature_combinations):
        ax = axes[i//n_cols, i%n_cols] if nrows > 1 else axes[i%n_cols]
        PartialDependenceDisplay.from_estimator(
            model, x, grid_resolution=20, ax=ax,
            features=[(vul_idx, trg_idx)], feature_names=feature_names, kind='average')
        ax.set_title(f'(취약성) {feature_names[vul_idx]} & (트리거) {feature_names[trg_idx]}', fontsize=18)
