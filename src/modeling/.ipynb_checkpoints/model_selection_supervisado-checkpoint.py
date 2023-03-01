from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import logging
# from scripts import preprocessing_datawrangling as preprocessing_tabela
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()


def def_clasificadores():
    print('[INFO]...Definiendo Clasificadores')
    clfs = {
        # 'lr': LogisticRegression(random_state=0),
        # 'rf': RandomForestClassifier(random_state=0, n_jobs=-1),
        'xgb': XGBClassifier(seed=0, verbosity=0, n_jobs=-1),
        # 'gbc': GradientBoostingClassifier(random_state=0),
        # 'lgbm': lightgbm.Classifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000,
        #                             learning_rate=0.01, metric='auc')
    }
    return clfs


def pipeline_clasificadores(clfs):
    print('[INFO]...Definiendo Pipeline Clasificadores')
    pipe_clfs = {}
    for name, clf in clfs.items():
        pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),
                                    ('clf', clf)])
    #     print(pipe_clfs['lr'])
    return pipe_clfs


def crear_params_clasif():
    print('[INFO]...Generando Parametros Para Clasificadores')
    # -------------- crear los parametros del grid para LR ------------------
    print('[INFO]...Parametros Logistic Regression')
    param_grids = {}
    C_range = [10 ** i for i in range(-4, 5)]
    param_grid = [{  # 'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'clf__C': C_range,
        'clf__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__class_weight': ['balanced', None]
    }]

    param_grids['lr'] = param_grid

    # ---------------- Crear parametros para RF ------------------------------
    print('[INFO]...Parametros Random Forest')
    param_grid = [{'clf__n_estimators': [600],  # [400,600],
                   #                    'clf__min_samples_split': [2, 10, 30],
                   #                    'clf__min_samples_leaf': [1, 10, 30],
                   'clf__max_depth': [50],  # [ 10,30,50],
                   'clf__max_features': ['sqrt', 'log2'],
                   'clf__criterion': ['gini', 'entropy'],
                   'clf__class_weight': ['balanced']
                   }]

    param_grids['rf'] = param_grid

    # ------------------------ crear parametros para xgboost -----------------
    print('[INFO]...Parametros XGBOOST')
    param_grid = [{
                # 'clf__eta': [10 ** i for i in range(-4, 1)],
        'clf__n_estimators': [10],#[1000],
                # 'clf__gamma': [0.5, 1, 1.5, 2, 5],
              # 'clf__lambda': [10 ** i for i in range(-4, 5)],
                'clf__min_child_weight': [1, 5, 10],
                'clf__colsample_bytree': [0.6, 0.8, 1.0],
                'clf__max_depth': [3,4, 6, 8]
    }]

    param_grids['xgb'] = param_grid

    # --------------- crear parametros para GRADIENTBOOSTING --------------------------
    print('[INFO]...Parametros Gradient Boosting')
    # crear parametros para gradientboosting {'clf__max_depth': 3, 'clf__max_features': 'sqrt', 'clf__n_estimators': 1000}
    param_grid = [{  # 'clf__learning_rate': [0.01, 0.02, 0.03],
        'clf__max_features': ['log2'],  # ['sqrt',],#,0.5],
        #                    'clf__subsample'    : [0.9, 0.5, 0.2],
        'clf__n_estimators': [1000],  # [200, 800,1000],#[200, 300, 400],
        'clf__max_depth': [3]  # ,6,10]#[2,3,4,6,8]
    }]

    param_grids['gbc'] = param_grid

    # ------------------------ crear parametros para lgbm ------------------------
    print('[INFO]...Parametros LGBM')
    param_grid = [{
        'num_leaves': [31, 127],
        'reg_alpha': [0.1, 0.5],
        'min_data_in_leaf': [30, 50, 100, 300, 400],
        'lambda_l1': [0, 1, 1.5],
        'lambda_l2': [0, 1]
    }]
    param_grids['lgbm'] = param_grid

    return param_grids


def get_grid_search(pipe_clfs, param_grids, X_train, y_train):
    print('[INFO]...GridSearch')
    # Hyperparameters
    print(X_train.shape)
    # lista de [best_score_, best_params_, best_estimator_]
    best_score_param_estimators = []

    # por cada clasificador
    for name in pipe_clfs.keys():
        print("[INFO]...NAME: ", name)
        gs = GridSearchCV(estimator=pipe_clfs[name],
                          param_grid=param_grids[name],
                          scoring='roc_auc',
                          n_jobs=-1,
                          #                           iid=False,
                          verbose=0,
                          cv=StratifiedKFold(n_splits=3,
                                             shuffle=True,
                                             random_state=0
                                             )
                          )

        gs = gs.fit(X_train, y_train)

        print(gs.best_score_, gs.best_params_)
        # actualizar: best_score_param_estimators
        best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    return best_score_param_estimators


def select_mejor_modelo(best_score_param_estimators):
    print('[INFO]...Seleccionar el Mejor Modelo')
    # ordernar best_score_param_estimators en order descendente para el parametro: best_score_
    best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x: x[0], reverse=True)

    # imprimir best_score_param_estimators
    for rank in range(len(best_score_param_estimators)):
        best_score, best_params, best_estimator = best_score_param_estimators[rank]
        print('Top', str(rank + 1))
        print('%-15s' % 'best_score:', best_score)
        print('%-15s' % 'best_estimator:'.format(20), type(best_estimator.named_steps['clf']))
        print('%-15s' % 'best_params:'.format(20), best_params, end='\n\n')
    # devolver el mejor modelo
    return best_score_param_estimators[0][2]


def get_mejor_clasificador(X_train, y_train):
    print('[INFO]...getting mejor classificador')
    clfs = def_clasificadores()
    pipe_clfs = pipeline_clasificadores(clfs)
    param_grids = crear_params_clasif()
    best_score_param_estimators = get_grid_search(pipe_clfs, param_grids, X_train, y_train)
    model_final = select_mejor_modelo(best_score_param_estimators)
    return model_final


def train_particion(X_train, y_train, variables2):
    """Se entrena el X_train con el mejor modelo"""
    print('[INFO]...Train_particion: X_train, y_train')
    best_model = get_mejor_clasificador(X_train[variables2], y_train)
    return best_model


# def entrenar_completo(df, dao, variables2, model, is_training,is_completo,is_precalc,is_read_precalc):
#     """is_training=True; para que genere el target encoding
#     is_completo=True; para descargar el target encoding
#     is_precalc=False; si previamente fue generado y no se quiere volver a calcular el tsfel. De otro modo poner True
#     is_read_precalc=True; Leer el TSFEL previamente calculado. Si se debe calcular poner False acá"""
#     logger.info('Entrenamiento - entrenar_completo: todo el dataset')
#     df, te = preprocessing_tabela.ingenieria_variables(df, dao, is_training, is_completo, is_precalc, is_read_precalc)
#     df = df.merge(te, on='clase_cons', how='left')
#     X, y = df[variables2], df.target
#     model.fit(X,y)# entrenar el df completo con el mejor modelo
#     logger.info("Descargado modelo final...")
#     # dao.descargar_modelo_supervisado(model)
#     logger.info("Done!")


# PASO 4: Entrenar particion para obtener el mejor modelo
# model = train_particion(X_train, y_train, variables2)

# PASO 5: Evaluar Train con el mejor modelo
# predecir_particion(model,X_train[variables2], y_train, 0.8, 0.8, 'Train' )

# PASO 6: Evaluar Test
# evaluar_test(model, X_test, y_test, te, variables2, th_pred=0.4, th_tpr=0.4)

# PASO 7: Entrenar dataset completo para generar y descargar MODELO FINAL que sera usado en la evaluacion en adelante
# Train_Modelar.entrenar_completo(df, dao, variables2, model, True, True, False, True)
