import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from itertools import groupby

class ChangeTrendPercentajeIdentifier(BaseEstimator, ClassifierMixin):

    def __init__(self, last_base_value, last_eval_value, threshold):
        self.last_base_value = last_base_value
        self.last_eval_value = last_eval_value
        self.threshold = threshold

    def compute_trend_percentage(self, X):
        last_values = X.last(f'{self.last_eval_value}M')['consumo'].tolist()
        X = X.drop(X.index[-1:])
        before_last_values = X.last(f'{self.last_base_value}M')['consumo'].tolist()
        base = np.mean(before_last_values)
        actual = np.mean(last_values)
        return 100 * actual / base

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_copy = X.copy()
        X_copy.sort_values('date', inplace=True)
        X_copy.set_index('date', inplace=True)
        X_copy = X_copy.groupby(['index']).apply(self.compute_trend_percentage).rename('trend_perc').reset_index()
#         X_copy = X_copy.groupby(['index']).swifter.apply(self.compute_trend_percentage).rename('trend_perc').reset_index()
        X_copy['is_fraud'] = X_copy.trend_perc.progress_apply(lambda x: 1 if (100 - x) > self.threshold else 0)
        return X_copy

class LowMeanLastMonthClassifier(BaseEstimator, ClassifierMixin):

        def __init__(self, fit_valid_years, last_month, err_cant):
            self.fit_valid_years = fit_valid_years
            self.last_month = last_month
            self.err_cant = err_cant

        def fit(self, X, y=None):
            df = X[X.year.isin(self.fit_valid_years)]
            df = df.groupby(['categoria', 'mes']).agg({'consumo': [np.std, np.size, np.mean]}).reset_index()
            df.columns = ["".join(x) for x in df.columns]
            df['err'] = df['consumostd'] / np.sqrt(df.consumosize.values)
            self.model = df
            return self

        def get_cons_last_months(self, df):
            last_months = self.last_month  # obtener los ultimos n meses
            df['periodo'] = df.groupby(['index'])['date'].rank(ascending=False)
            df = df[df['periodo'] <= last_months]
            df = df.drop(columns=['periodo'])
            return df

        def predict(self, X):
            df = self.get_cons_last_months(X)
            self.df_last = df
            df = df.merge(self.model, on=['categoria', 'mes'], how='left')  # se elimina fase y modifica mes
            df = df.groupby(['index']).agg(
                {'consumo': np.mean, 'consumomean': np.mean, 'err': np.mean}).reset_index()
            df['prediccion'] = (df.consumo < (df.consumomean - (self.err_cant * df.err))).astype(int)

            return df

        

class ConstantConsumptionClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, months):
        self.last_months = months
        
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        """ esto funciona para formato long"""
        df_variance=self.get_variance(X)
        df_cant_zeros=self.get_cant_zeros(X)
        df_variance = df_variance.merge(df_cant_zeros, on=['index'])
        df_variance['has_almost_0'] = (df_variance.cant_ceros > 0).astype(int)  # por lo menos 1 cero
        df_variance['has_all_0'] = (df_variance.cant_ceros == self.last_months).astype(int)  # todos son cero
        df_variance['has_all_non_0'] = (df_variance.cant_ceros == 0).astype(int)  # ninguno es cero
        df_variance = self.get_rules(df_variance)
        return df_variance

    def get_variance(self, df):
        # obtener la varianza. Ya que esta nos indica cuanto varian los datos y esto nos dice que tan diferentes son
        df_variance = df.groupby(['index']).agg({'consumo': [np.var, np.mean]}).reset_index()
        df_variance.columns = ['index', 'var_consumo', 'mean_consumo']
        return df_variance

    def get_cant_zeros(self, df):
        # obtener cantidad de cero por numcta o index
        df_g_cant_ceros = df.groupby(['index']).apply(lambda x: np.sum(x.consumo.values == 0)).reset_index()
        df_g_cant_ceros.columns = ['index', 'cant_ceros']
        return df_g_cant_ceros

    def get_rules(self, df):
        # mascara: exactamente el mismo valor osea es constante hasta 5mts3
        msk_rule1 = (df.has_all_non_0 == 1) & (df.var_consumo == 0) & (df.mean_consumo <= 5)
        df['regla_const_under_5mtr3'] = msk_rule1.astype(int)
        # mascara: exactamente el mismo valor osea es constante mayor a 5mts3
        msk_rule2 = (df.has_all_non_0 == 1) & (df.var_consumo == 0) & (df.mean_consumo > 5)
        df['regla_consump_constant'] = msk_rule2.astype(int)
        # mascara: consumos similares pero hasta 6mts3
        msk_rule3 = (df.has_all_non_0 == 1) & (df.var_consumo > 0) & (df.var_consumo < 0.5) & (df.mean_consumo <= 6)
        df['regla_similar_lt_6mts3'] = msk_rule3.astype(int)
        return df

class ChangeTrendPercentajeIdentifierWide(BaseEstimator, ClassifierMixin):

    def __init__(self, last_base_value, last_eval_value, threshold, is_wide = True):
        self.last_base_value = last_base_value
        self.last_eval_value = last_eval_value
        self.threshold = threshold
        self.is_wide = is_wide
        
    def convert_wide(self, df):
        df_wide=pd.pivot(df, index=['index'], columns=['date'], values=['consumo']).reset_index()
        # organizar las columnas con nombres apropiados
        df_wide.columns = ['index']+[str(i)+'_anterior' for i in range(self.last_eval_value + self.last_base_value)][::-1]
        return df_wide
    
    def get_cant_cols(self):
        #obtener columnas base y columnas usadas para evaluar
        cols_base = [str(i)+'_anterior' for i in range(self.last_eval_value+1,self.last_base_value+self.last_eval_value+1)][::-1]#last_base_value
        cols_eval = [str(i)+'_anterior' for i in range(1,self.last_eval_value+1)][::-1]#last_eval_value
#         print('[INFO]...cols base:', cols_base)
#         print('[INFO]...cols eval:', cols_eval)
        return cols_base, cols_eval
        
    def compute_trend_percentage_wide(self, X):
        if self.is_wide==False:
            X = self.convert_wide(X)
        
        cols_base, cols_eval = self.get_cant_cols()
        X['trend_perc'] = 100 * X[cols_eval].mean(axis=1)/(X[cols_base].mean(axis=1)+0.000001)
        return X

    def fit(self, X, y=None):
        
        return self

    def predict(self, X):
        X_copy = X.copy()
        X_copy = self.compute_trend_percentage_wide(X_copy)
        X_copy['is_fraud_trend_perc'] = (100-X_copy['trend_perc']>self.threshold).astype(int)
        return X_copy[['trend_perc','is_fraud_trend_perc']]

    
class ConstantConsumptionClassifierWide(BaseEstimator, ClassifierMixin):
    
    def __init__(self, min_count_constante):
        self.min_count_constante = min_count_constante
        
    def fit(self, X, y=None):
        return self
    
    def len_max_consumo_constante_seg(self,consumo):
#         print(consumo)
        g = [[k, len(list(v))] for k, v in groupby(consumo)]
        g = [x for x in g if (x[1] >= self.min_count_constante)]
        if any(g):
            return 1
#             return sorted(g, reverse=True, key=lambda x: x[-1])[0][1]
        else:
            return 0

    def predict(self, X):
        pred = X.apply(lambda x : self.len_max_consumo_constante_seg(x.values),axis=1)
        return pred