
class Config(object):
    cfg = dict()
    
    @classmethod
    def init_config(cls):
        # bucket_path = 's3://iadbprod-ine-tsp-analyticaldata/' 
        # key = 'EEGSA-GUATEMALA/data_clean/'
        
        Config = {}
        workspace = {'data_path' : '../../data/clean_data/', 'model_path' : '../../model_files/'}
        model = {'periodo':12,
                 'files_cols_derivadas':'features_by_const_boruta2_12.csv',
                 'cols_uc':['departamento','codigo_postal','actividad_economica','tipo_tarifa','tarfia','nivel_de_tension','cod_mat','cant_ttarifa','medidor_interior'],
                 'tsfel_names_path' : 'tsfel_boruta2_12_features.json',
                 'hip_lgbm' : {'lgbmclassifier__colsample_bytree': 0.4886705267429202, 'lgbmclassifier__learning_rate': 0.016333514000872486, 'lgbmclassifier__max_bin': 158,         'lgbmclassifier__max_depth': 10, 'lgbmclassifier__min_child_samples': 118, 'lgbmclassifier__min_child_weight': 0.1, 'lgbmclassifier__num_leaves': 22, 'lgbmclassifier__reg_alpha': 0, 'lgbmclassifier__reg_lambda': 5, 'lgbmclassifier__scale_pos_weight': 1, 'lgbmclassifier__subsample': 0.8945613420997809, 'lgbmclassifier__subsample_freq': 18},
                 'hip_cat' : {'catboostclassifier__bagging_temperature': 100, 'catboostclassifier__border_count': 128, 'catboostclassifier__depth': 4, 'catboostclassifier__l2_leaf_reg': 52.073120365069194, 'catboostclassifier__learning_rate': 0.03139743331517567} ,
                 'hip_trend' : {'last_base_value':3,'last_eval_value':3,'threshold':60},
                 'hip_const' : {'min_count_constante':5},
                 'hip_NN':{'batch_size' : 2048},
                 'cols_comb_model' : ['prob_cb','prob_lgbm','prob_ffn','prob_lstmffn','fraud_constante','fraud_trend'],
                 'lgbm_model_name' : 'model_fraud_lgbm.pkl',
                 'cat_model_name' : 'model_fraud_cat.pkl',
                 'nn_model_name' : 'model_fraud_nn.h5',
                 'lstmnn_model_name' : 'model_fraud_lstmnn.h5',
                 'comb_model_name' : 'model_fraud_comb.pkl',
                 'pipe_feat_name' : 'pipe_features.pkl',
                 'pipe_spent_name' : 'pipe_spent.pkl',
                  }
        data = {'input_data' : f'eegsa_etiquetado_wide_completo_periodo_12.parquet' }
        evaluacion = {'evaluation_data' : 'eegsa_data_evaluacion_2022-07-01_periodos_12.parquet',
                      'mes_ultimo_consumo' : '2022-07-01',
                      'th_critico' : 0.86,
                      'th_warning'  : 0.7
                       }
        Config['workspace'] = workspace
        Config['model'] = model
        Config['data'] = data
        Config['evaluacion'] = evaluacion
        cls.cfg = Config
        
    @classmethod
    def get_config(cls):
        return cls.cfg
    
    @classmethod
    def get(cls,key):
        return cls.cfg.get(key) 