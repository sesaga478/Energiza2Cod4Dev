cols_archivo_input = ['uc', 'situacao', 'classe', 'clase_cons', 'circuito', 'proprietario', 'alimentador', 'fase',
                      'data_ligacao', 'medidor', 'municipio', 'bairro', 'dt_situacao_uc',
                      'tensao', 'tipo_tarifa', 'flg_geradora', 'subestacao', 'coordenadas',
                      'abr-20', 'may-20', 'jun-20', 'jul-20', 'ago-20', 'sep-20', 'oct-20', 'nov-20', 'dic-20',
                      'ene-21', 'feb-21', 'mar-21']

cols_supervisado = ['uc', 'situacao', 'classe', 'clase_cons', 'circuito', 'proprietario', 'alimentador', 'fase',
                    'data_ligacao', 'medidor', 'municipio', 'bairro', 'dt_situacao_uc',
                    'tensao', 'tipo_tarifa', 'flg_geradora', 'subestacao', 'coordenadas',
                    '12_anterior', '11_anterior', '10_anterior', '9_anterior',
                    '8_anterior', '7_anterior', '6_anterior', '5_anterior', '4_anterior', '3_anterior',
                    '2_anterior', '1_anterior']
# tipos de variables
categoricas_vars_eval = ['uc', 'situacao', 'classe', 'clase_cons', 'circuito', 'proprietario', 'alimentador', 'fase',
                    'medidor', 'municipio', 'bairro', 'tensao', 'tipo_tarifa', 'flg_geradora', 'subestacao',
                    'coordenadas']
consumo_super = ['12_anterior', '11_anterior', '10_anterior', '9_anterior',
                 '8_anterior', '7_anterior', '6_anterior', '5_anterior', '4_anterior', '3_anterior',
                 '2_anterior', '1_anterior']

# para TRAIN
cols_train_inicial = ['uc', 'cliente', 'situacao', 'municipio', 'bairro', 'dt_situacao_uc', 'data_ligacao',
                      'classe', 'tensao', 'fase', 'clase_cons', 'tipo_tarifa', 'flg_geradora', 'circuito',
                      'subestacao', 'alimentador', 'proprietario', 'medidor', '12_anterior', '11_anterior',
                      '10_anterior', '9_anterior', '8_anterior', '7_anterior', '6_anterior', '5_anterior',
                      '4_anterior', '3_anterior', '2_anterior', '1_anterior', 'fecha_regul', '1_posterior',
                      '2_posterior', '3_posterior', '4_posterior', '5_posterior', '6_posterior', '7_posterior',
                      '5_posterior', '9_posterior', '10_posterior', '11_posterior', '12_posterior', 'target']

categoricas_vars_train = ['uc', 'cliente', 'situacao', 'classe', 'clase_cons', 'circuito', 'proprietario',
                          'alimentador',
                          'fase', 'municipio', 'bairro', 'tensao', 'tipo_tarifa', 'flg_geradora', 'subestacao',
                          'medidor']

numerics_vars_train = ['12_anterior', '11_anterior', '10_anterior', '9_anterior',
                       '8_anterior', '7_anterior', '6_anterior', '5_anterior', '4_anterior',
                       '3_anterior', '2_anterior', '1_anterior']

sel_cols_3 = ["cant_ceros_3",
              "max_cant_ceros_seg_3",
              "skew_cons3",
              "slope_3",
              "std_cons3",
              "var_cons3"]

sel_cols_6 = ["kurt_cons6",
            "cant_ceros_6", 
            "max_cant_ceros_seg_6",
            "skew_cons6",
            "slope_6",
            "std_cons6",
            "var_cons6"]

sel_cols_12 = ["kurt_cons12",
                "max_cant_ceros_seg_12",
                "skew_cons12",
                "slope_12"
                ]

cols_stat_spect_tmp = ["0_Centroid",
                        "0_ECDF Percentile_0",
                        "0_Entropy",
                        "0_FFT mean coefficient_0",
                        "0_FFT mean coefficient_2",
                        "0_FFT mean coefficient_3",
                        "0_FFT mean coefficient_4",
                        "0_FFT mean coefficient_5",
                        "0_FFT mean coefficient_6",
                        "0_Histogram_5",
                        "0_Interquartile range",
                        "0_Kurtosis",
                        "0_LPCC_0",
                        "0_LPCC_1",
                        "0_LPCC_10",
                        "0_LPCC_11",
                        "0_LPCC_12",
                        "0_LPCC_2",
                        "0_LPCC_3",
                        "0_LPCC_4",
                        "0_LPCC_5",
                        "0_LPCC_6",
                        "0_LPCC_7",
                        "0_LPCC_8",
                        "0_LPCC_9",
                        "0_MFCC_0",
                        "0_MFCC_1",
                        "0_MFCC_10",
                        "0_MFCC_11",
                        "0_MFCC_2",
                        "0_MFCC_3",
                        "0_MFCC_4",
                        "0_MFCC_6",
                        "0_MFCC_7",
                        "0_MFCC_8",
                        "0_MFCC_9",
                        "0_Max",
                        "0_Max power spectrum",
                        "0_Maximum frequency",
                        "0_Mean diff",
                        "0_Median absolute deviation",
                        "0_Median absolute diff",
                        "0_Median diff",
                        "0_Negative turning points",
                        "0_Peak to peak distance",
                        "0_Positive turning points",
                        "0_Skewness",
                        "0_Slope",
                        "0_Spectral centroid",
                        "0_Spectral entropy",
                        "0_Spectral kurtosis",
                        "0_Spectral roll-off",
                        "0_Spectral slope",
                        "0_Spectral spread",
                        "0_Spectral variation",
                        "0_Standard deviation",
                        "0_Sum absolute diff",
                        "0_Variance",
                        "0_Wavelet absolute mean_0",
                        "0_Wavelet energy_0",
                        "0_Wavelet entropy",
                        "0_Wavelet standard deviation_1",
                        "0_Wavelet standard deviation_2",
                        "0_Wavelet standard deviation_3",
                        "0_Wavelet variance_0",
                        "0_Wavelet variance_1",
                        "0_Wavelet variance_2",
                        "0_Wavelet variance_3",
                        "0_Wavelet variance_4",
                        "0_Wavelet variance_5",
                        "0_Zero crossing rate",
                        "cant_ceros_3",
                        "cant_ceros_6",
                        "kurt_cons12",
                        "kurt_cons6",
                        "max_cant_ceros_seg_12",
                        "max_cant_ceros_seg_3",
                        "max_cant_ceros_seg_6",
                        "skew_cons12",
                        "skew_cons3",
                        "skew_cons6",
                        "slope_12",
                        "slope_3",
                        "slope_6",
                        "std_cons3",
                        "std_cons6",
                        "var_cons3",
                        "var_cons6"]

# sel_cols_dummy = ['classe_comercial', 'classe_industrial', 'classe_otros', 'classe_residencial', 'classe_rural',
#                   'fase_bi', 'fase_mo', 'fase_tr', 'proprietario_c', 'proprietario_p', 'proprietario_sin_datos']

sel_cols_index = ['index', 'uc', 'clase_cons', 'municipio', 'classe',
                  'fase', 'tipo_tarifa', 'flg_geradora', 'proprietario']

vars_not_train = ['tipo_tarifa_convencional', 'tipo_tarifa_horaria_azul', 'tipo_tarifa_horaria_branca',
                  'tipo_tarifa_horaria_verde',
                  'flg_geradora_G', 'flg_geradora_ng', 'media_consumo_ciclo']

# cols que no se van a usar en el train
cols_to_drop_train = ['data_sitacao', 'data_ligacao', 'is_situ_menor_regul', 'medidor',
                      # 'data_geradora'-->no existe en archivo,
                      'circuito', 'subestacao', 'alimentador', 'bairro', 'cliente']

# para EVALUACION
cols_to_dummy = ['classe', 'fase', 'proprietario']
cols_to_dummy_long = cols_to_dummy + ['estacion']

# sel_vars_atoencoders = ['consumo', 'classe_residencial', 'classe_rural', 'classe_otros', 'classe_industrial',
#             'estacion_Verano', 'estacion_Invierno', 'estacion_Otono', 'estacion_Primavera',
#                         'fase_bi',
#                         'fase_mo',
#                         'fase_tr'
#                         ]

sel_vars_atoencoders = ['consumo', 'dummy_classe_residencial', 'dummy_classe_rural', 'dummy_classe_otros',
                        'dummy_classe_comercial', 'dummy_classe_industrial', 'dummy_fase_bi', 'dummy_fase_mo',
                        'dummy_fase_tr', 'dummy_estacion_Invierno', 'dummy_estacion_Otono', 'dummy_estacion_Primavera',
                        'dummy_estacion_Verano'] # 'dummy_proprietario_c', 'dummy_proprietario_p', 'dummy_proprietario_sin_datos',

# para dashboard
list_clases = ['dummy_classe_industrial', 'dummy_classe_comercial', 'dummy_classe_otros', 'dummy_classe_residencial',
               'dummy_classe_rural']
cols_dashboard = ['uc', '12_anterior', '11_anterior', '10_anterior', '9_anterior', '8_anterior',
                  '7_anterior', '6_anterior', '5_anterior', '4_anterior', '3_anterior', '2_anterior',
                  '1_anterior', 'mean_3', 'mean_6', 'mean_12', 'cant_ceros_3', 'cant_ceros_6',
                  'cant_ceros_12', 'max_cant_ceros_seg_3', 'max_cant_ceros_seg_6', 'max_cant_ceros_seg_12',
                  'min_cons12', 'max_cons12', 'min_cons6', 'max_cons6', 'min_cons3', 'max_cons3', 'latitude',
                  'longitude', 'circuito', 'classe', 'clase_cons',
                  'fase']  # , 'nro_ciclos'] # se saca nro_ciclos pq no se esta calculando en el data wrangling
####
###################################### TRAIN SUP ALL ################################
vars_finales = ['dummy_classe_comercial',
 'dummy_classe_industrial',
 'dummy_classe_otros',
 'dummy_classe_residencial',
 'dummy_classe_rural',
 'dummy_fase_bi',
 'dummy_fase_mo',
 'dummy_fase_tr',
 'dummy_proprietario_0',
 'dummy_proprietario_c',
 'dummy_proprietario_p',
 '0_MFCC_3',
 '0_MFCC_2',
 '0_MFCC_11',
 '0_MFCC_10',
 '0_MFCC_0',
 '0_LPCC_9',
 '0_LPCC_8',
 '0_LPCC_7',
 '0_LPCC_6',
 '0_LPCC_5',
 '0_LPCC_4',
 '0_LPCC_3',
 '0_LPCC_2',
 '0_LPCC_12',
 '0_MFCC_5',
 '0_LPCC_11',
 '0_LPCC_1',
 '0_LPCC_0',
 '0_FFT mean coefficient_1',
 '0_Zero crossing rate',
 '0_Slope',
 '0_Negative turning points',
 '0_Median absolute diff',
 '0_Mean diff',
 '0_LPCC_10',
 '0_MFCC_7',
 '0_MFCC_9',
 '0_Max power spectrum',
 '0_Wavelet entropy',
 '0_Spectral centroid',
 '0_Spectral entropy',
 '0_Spectral slope',
 '0_Spectral variation',
 '0_Spectral spread',
 '0_Centroid',
 'std_cons3',
 'var_cons3',
 'skew_cons3',
 'slope_6',
 'kurt_cons12',
 'skew_cons12',
 'max_cons6',
 'min_cons12',
 'slope_12',
 'kurt_cons6',
 'skew_cons6',
 'slope_3',
 'std_cons6',
 'cant_ceros_3',
 '0_Median absolute deviation',
 '0_Min',
 '0_Interquartile range',
 '0_MFCC_4',
 '0_FFT mean coefficient_3',
 '0_FFT mean coefficient_0',
 '0_Wavelet variance_4',
 '0_MFCC_6',
 '0_ECDF Percentile Count_1',
 '0_FFT mean coefficient_6',
 '0_MFCC_1',
 '0_MFCC_8',
 'clase_cons_prob',
 'municipio_prob']