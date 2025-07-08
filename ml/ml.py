import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FredMdTransformer:
    
    def __init__(self, meta, id_col = 'fred', tcode_col = 'tcode'):
        self.meta = meta
        self.id_col = id_col
        self.tcode_col = tcode_col
        self.tcode_map = meta.set_index(id_col)[tcode_col].astype(int).to_dict()
        
    def transform(self, df):
        ret = pd.DataFrame(index = df.index, columns = df.columns)
        
        for col in df.columns:
            
            x = df[col]
            tcode = self.tcode_map[col] # col번째 변수의 tcode 값
            
            if col == 'CPIAUCSL':
                y = x.pct_change(12) * 100
            else:
                if tcode == 1:
                    y = x
                elif tcode == 2:
                    y = x.diff() # diff() default = 1
                elif tcode == 3:
                    y = x.diff().diff()
                elif tcode == 4:
                    y = np.log(x)
                elif tcode == 5:
                    y = np.log(x).diff()
                elif tcode == 6:
                    y = np.log(x).diff().diff()
                elif tcode == 7:
                    y = (x / x.shift(1) - 1.0).diff()
                    
            ret[col] = y
        
        return ret
    

class RandomFeatureSelector(TransformerMixin, BaseEstimator): #
    #랜덤으로 k개 피처를 뽑기
    def __init__(self, k=3, random_state=None):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        # 0..n_features-1 중에서 k개를 랜덤 선정
        self.selected_idx_ = rng.choice(n_features, size=self.k, replace=False)
        return self

    def transform(self, X):
        # DataFrame일 경우 .iloc, array일 경우 numpy 인덱싱
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_idx_]
        else:
            return X[:, self.selected_idx_]