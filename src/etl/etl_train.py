import pandas as pd
import unidecode
from tqdm import tqdm
import glob
import numpy as np

def clean_historico_consumo_cols(df):
    df.columns = df.columns.str.lower()
    # Mes operacion
    mes_operacion_invalidos = [10, 16, 10.11, 'MES_OPERACION']
    df = df[~df.mes_operacion.isin(mes_operacion_invalidos)]
    df = df[~df.mes_operacion.isnull()]
    df.mes_operacion = df.mes_operacion.astype(int)
    df.mes_operacion = df.mes_operacion.astype(str)
    df['year'] = df['mes_operacion'].apply(lambda x: str(x)[:4])
    df['mes'] = df['mes_operacion'].apply(lambda x: str(x)[4:6])
    df['date'] = df['mes'] + '-' + df['year']
    df.date = pd.to_datetime(df.date)

    # ID usuario
    df.dropna(subset=['id_usuario'], inplace=True)
    df.id_usuario = df.id_usuario.astype(int)
    df.id_usuario = df.id_usuario.astype(str)

    # departamento municipio zona
    str_vars = ['departamento', 'municipio', 'zona']
    df.loc[:, str_vars] = df[str_vars].fillna('sin_dato').astype(str)
    df.loc[:, str_vars] = df.loc[:, str_vars].applymap(lambda x: x.lower())
    df.loc[:, str_vars] = df.loc[:, str_vars].applymap(lambda x: x.replace(' ', '_'))
    df.loc[:, str_vars] = df.loc[:, str_vars].applymap(lambda x: unidecode.unidecode(x))

    df.loc[df.departamento == 'gua$emala', 'departamento'] = 'guatemala'
    df.loc[df.departamento == 'guatema<a', 'departamento'] = 'guatemala'

    df.municipio = df.municipio.replace({'nueva_concepci?': 'nueva_concepcion', 'nueva_concepci??n': 'nueva_concepcion',
                                         'san_miguel_due+-as': 'san_miguel_duenas',
                                         'san_miguel_due??as': 'san_miguel_duenas',
                                         'san_miguel_due?s': 'san_miguel_duenas',
                                         'san_miguel_duea+-as': 'san_miguel_duenas'
                                         })

    df.zona = df.zona.replace({'0': 'zona_0', 'zona0': 'zona_0', 'zonza_0': 'zona_0',
                               'zona_02': 'zona_2',
                               'zona_03': 'zona_3', 'zona_0_3': 'zona_3', 'zona3': 'zona_3',
                               'zona_0_4': 'zona_4',
                               'zona8': 'zona_8',
                               'zona10': 'zona_10',
                               })
    # Consumo
    df.consumo_energia_total = df.consumo_energia_total.astype(float)
    df.rename(columns={'consumo_energia_total': 'consumo'}, inplace=True)

    return df


def clean_historico_ordenes_cols(df):
    df.columns = df.columns.str.lower().str.replace('.', '_').str.replace(' ', '_')

    if 'unnamed:_0' in df.columns:
        df.drop(columns=['unnamed:_0'], inplace=True)

    df.cod_mat = df.cod_mat.astype(str)
    df.contrato = df.contrato.astype(str)
    df['f_ejec'] = df['f_ejec'].astype(str)

    df.codigo_postal = df.codigo_postal.fillna('sin_dato').apply(lambda x: str(x).split('.')[0])
    df.tecnico = df.tecnico.fillna('sin_dato').str.lower().str.replace(' ', '').str.replace('verif16', 'verif-16')
    df.subclase = df.subclase.fillna('sin_dato').str.lower().str.replace(' ', '').str.replace('-', '').replace(
        {'<': 'sin_dato'})
    df.clase = df.clase.fillna('sin_dato').str.lower().str.replace(' ', '')
    df.cod_mat = df.cod_mat.replace({'nan': 'sin_dato', '-': 'sin_dato', '0': 'sin_dato', '31.08.2018': 'sin_dato'})

    # c_e
    df.c_e = df.c_e.fillna('sin_dato').str.strip().str.lower().replace({'cancel': 'can',
                                                                        'n0e': 'noe',
                                                                        'noer': 'noe',
                                                                        "n21`": 'n21'})

    fizc_cod_ej_anomalias = ['n08', 'n23', 'v24', 'v25', 'v26']
    fizc_cod_ej_no_ejec = ['n01', 'n02', 'n03', 'n14', 'n30', 'n37', 'noe', 'v45', 'v89', 'can']
    df['has_anomalia'] = df.c_e.isin(fizc_cod_ej_anomalias).astype(int)
    df['no_ejecuto'] = df.c_e.isin(fizc_cod_ej_no_ejec).astype(int)

    # Elimino las que no ejecuto
    df = df[df.no_ejecuto == 0]

    # Correccion fecha fizcalizacion
    df = df[df.f_ejec != 'nan']
    df = df[df.f_ejec != '0']
    df = df[df.f_ejec != '1435281']
    df['year'] = df.f_ejec.str.split('-').str[0]

    df['f_ejec2'] = df.f_ejec. \
        str.replace('31/008/2018', '31/08/2018'). \
        str.replace('2107-06-21 00:00:00', '2017-06-21 00:00:00'). \
        str.replace('25.01.2718', '25.01.2018'). \
        str.replace('06.06.2818', '06.06.2018'). \
        str.replace('2049-08-30 00:00:00', '2019-08-30 00:00:00'). \
        str.replace('2919-12-05 00:00:00', '2019-12-05 00:00:00'). \
        str.replace('16/11//2021', '16/11/2021'). \
        str.replace('2821-02-11 00:00:00', '2021-02-11 00:00:00').str.split(' ').str[0].str.replace('/',
                                                                                                    '-').str.replace(
        '.', '-').str.replace('.', '-')
    df['cant_carac_mes'] = df.f_ejec2.str.split('-').str[1].str.len()
    df['is_year_to_change'] = df.f_ejec2.apply(lambda x: 1 if len(str(x).split('-')[2]) == 4 else 0)
    df.loc[df.is_year_to_change == 1, 'f_ejec2'] = df.loc[df.is_year_to_change == 1].f_ejec2.apply(
        lambda x: str(x).split('-')[2] + '-' + str(x).split('-')[1] + '-' + str(x).split('-')[0])
    df['year2'] = df.f_ejec2.str.split('-').str[0]
    df['mes2'] = df.f_ejec2.str.split('-').str[1]
    df['dia2'] = df.f_ejec2.str.split('-').str[2]
    df['len_fizca2'] = df.f_ejec2.apply(len)

    df.loc[(df.file_year == '2013'), 'year2'] = '2013'
    df.loc[(df.file_year == '2014'), 'year2'] = '2014'
    df.loc[(df.file_year == '2020'), 'year2'] = '2020'
    df = df[df.year2 != '1900']

    df['f_ejec3'] = df.year2 + '-' + df.mes2 + '-' + df.dia2
    df.f_ejec3 = pd.to_datetime(df.f_ejec3)

    df.f_ejec = df.f_ejec3
    df.drop(columns=['f_ejec2', 'f_ejec3', 'cant_carac_mes', 'is_year_to_change', 'year', 'year2', 'len_fizca2', 'mes2',
                     'dia2'], inplace=True)

    # Correccion de repetidos
    df = df.drop_duplicates()
    df_order_cant = df.groupby('orden').size().rename('cant_filas').reset_index()
    df = df.merge(df_order_cant, on='orden')
    df_fizca_rep = df[df.cant_filas > 1].copy()
    df_fizca_no_rep = df[df.cant_filas == 1].copy()

    df_fizca_rep = df_fizca_rep.drop_duplicates(subset=['orden', 'contrato', 'f_ejec', 'c_e'])
    df_fizca_rep = df_fizca_rep.loc[df_fizca_rep.groupby(['orden', 'contrato', 'c_e']).f_ejec.idxmin()]
    df_fizca_rep = df_fizca_rep.loc[df_fizca_rep.groupby(['orden', 'contrato', 'f_ejec']).has_anomalia.idxmax()]
    df_fizca_rep = df_fizca_rep.loc[df_fizca_rep.groupby(['orden', 'contrato', 'has_anomalia']).f_ejec.idxmin()]
    df_fizca_rep['cant_filas'] = df_fizca_rep.groupby(['orden']).cant_filas.transform(lambda x: len(x))
    df_fizca_rep = df_fizca_rep[df_fizca_rep.cant_filas == 1]

    df = df_fizca_no_rep.append(df_fizca_rep)

    # Como vamos a analizar a nivel mensual pueden existir
    # contratos que puede tener varias ordenes por mes por lo tanto
    # me quedo con la orden que tiene fraude
    df['is_fraud'] = df.has_anomalia
    df['mes'] = df.f_ejec.dt.month
    df['year'] = df.f_ejec.dt.year
    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['mes'].astype(str))
    df.contrato = df.contrato.str.strip()
    df = df[df.contrato != '']
    df = df.loc[df.groupby(['contrato', 'date']).is_fraud.idxmax()]

    return df

def clean_zgm023_cols(df):
    # Normalizacion nombre de columnas
    df.columns = df.columns.str.lower(). \
        str.replace('\\(x\\)', ''). \
        str.replace('\\(#\\)', 'num'). \
        str.replace('#', 'num'). \
        str.replace('*', ''). \
        str.replace('[()]', ''). \
        str.replace('[=]', ''). \
        str.replace('[;]', ''). \
        str.replace('[.]', ' '). \
        str.replace('[-]', 'menos'). \
        str.replace('[+]', 'mas'). \
        str.replace('menos ', 'menos'). \
        str.strip(). \
        str.replace(' +', '_').map(unidecode.unidecode)

    # Tipos de datos str
    str_cols = ['contrato', 'instalacion', 'codigo_postal', 'contrato', 'no_orden_3', 'no_orden_2', 'no_orden_1',
                'orden_abierta']
    for x in str_cols:
        df[x] = df[x].astype(str)

    # Desocupado
    df.desocupado.fillna('No', inplace=True)

    # Rellenamos null --> sin_dato
    cols_1 = df.select_dtypes(include=['O']).columns.tolist()
    for x in cols_1:
        df[x].fillna('sin_dato', inplace=True)

    # no_de_poste
    df.no_de_poste = df.no_de_poste.astype(str)
    df.no_de_poste = df.no_de_poste.str.strip().str.split('.').str[0].str.replace('nan', 'sin_dato')
    # Hay valores de un digito --> preguntar si esto esta bien, hay valores de distintos tamaños e digitos 200 o 812382

    # no_orden_3 no_orden_2 no_orden_1
    df.no_orden_3 = df.no_orden_3.str.strip().str.split('.').str[0].str.replace('nan', 'sin_dato')
    df.no_orden_2 = df.no_orden_2.str.strip().str.split('.').str[0].str.replace('nan', 'sin_dato')
    df.no_orden_1 = df.no_orden_1.str.strip().str.split('.').str[0].str.replace('nan', 'sin_dato')

    # zona
    df.zona = df.zona.str.lower().str.replace('[ |-]', '').str.split('(\d+)').apply(
        lambda x: '_'.join([i for i in x if i != ''])). \
        str.replace('zona_02', 'zona_2'). \
        str.replace('zona_03', 'zona_3'). \
        str.replace('zona_08', 'zona_8'). \
        str.replace('zonza_0', 'zona_0')
    # Sigue quedando valores raros --> 0
    # orden_abierta
    df.orden_abierta = df.orden_abierta.str.strip().str.split('.').str[0].str.replace('nan', 'sin_dato')

    # Tipos de datos dates
    date_cols = ['fecha_de_alta', 'fecha_de_baja']
    df.loc[df.fecha_de_baja == '31/12/9999', 'fecha_de_baja'] = '31/12/2200'
    for x in date_cols:
        print(x)
        df[x] = pd.to_datetime(df[x], format='%d/%m/%Y')

    # --> no sabemos como tratar los valores 0000/00/00
    # consulta_de_fecha_de_lectura_num --> que significa

    # Tipos de datos numericos
    cols_3 = df.select_dtypes(exclude=['O', 'datetime']).columns.tolist()
    df[cols_3].head()
    # num_de_aviso_nl_notas_del_lector, contrato_menos5  --> parece un codigo?
    # nivel_de_tension --> los valores nan , pueden ser cero??
    # longitud_x , latitud_y --> quedan con nulos
    df.rename(columns={'material': 'cod_mat'}, inplace=True)
    #     cols = ['contrato','unidad_de_lectura','codigo_postal','fecha_de_alta','cod_mat','no_de_poste','tarfia','multiplicador','actividad_economica','kw_cont',
    #     'medidor_interior','folio','nivel_de_tension','fecha_de_baja','longitud_x','latitud_y','indice_de_solvencia']
    return df


def run_etl_historico_consumo(DATA_PATH_RAW):
    files = [file for file in glob.glob(DATA_PATH_RAW + "historico_consumo/**/*.txt", recursive=True)]
    df = pd.DataFrame()
    for f in files:
        print(f)
        df_csv = pd.read_csv(f, sep='\t')
        tipo_tarif = 'TNS' if 'TNS' in f.split('/')[-1] else 'TS'
        df_csv['tipo_tarifa'] = tipo_tarif
        df_csv = clean_historico_consumo_cols(df_csv)
        df = df.append(pd.concat([df_csv], axis=0, ignore_index=True), ignore_index=True)

    return df


def run_etl_historico_ordenes(DATA_PATH_RAW):
    files = [file for file in glob.glob(DATA_PATH_RAW + "historico_ordenes/*.xlsx")]
    df = pd.DataFrame()
    for f in files:
        print(f)
        df_csv = pd.read_excel(f, engine='openpyxl')
        df_csv['file_year'] = f.split(' ')[-1].split('.')[0]
        df = df.append(pd.concat([df_csv], axis=0, ignore_index=True), ignore_index=True)
    df = clean_historico_ordenes_cols(df)
    return df


def run_etl_zgm023(DATA_PATH_RAW):
    df_atributos_cols = pd.read_csv(DATA_PATH_RAW+'/cols_zgm023.csv')  # obtiene las columnas a ser usadas
    list_cols = df_atributos_cols.columns.tolist()
    list_cols = list_cols + ['date_filename']
    files = [file for file in glob.glob(DATA_PATH_RAW + "data_zgm023/*.txt")]
    df = pd.DataFrame()
    for f in files:
        print(f)
        df_csv = pd.read_csv(f, sep=',', header=None)
        date_filename = f.split('_')[2]  # sacarle la fecha al archivo
        df_csv[len(df_csv.columns)] = date_filename  # adiciona columna al final del df
        df = df.append(pd.concat([df_csv], axis=0, ignore_index=True), ignore_index=True)

    df.columns = list_cols
    df = clean_zgm023_cols(df)

    return df

def create_train_dataset(df_consumo,df_ordenes,df_exdata):
    # Creacion del train data set
    cant_periodos = 12
    min_date_data_ordenes = str(df_consumo.date.min() + pd.DateOffset(months=cant_periodos))
    fecha_fraud_list = df_ordenes[(df_ordenes.date >= min_date_data_ordenes)]['date'].sort_values().astype(
        str).unique().tolist()
    list_df = []
    for fecha_fraud in tqdm(fecha_fraud_list, total=len(fecha_fraud_list)):
        df_etiquetado_fraud = df_consumo[df_consumo.date < fecha_fraud].copy()
        ctas_fraud = df_ordenes[(df_ordenes.date == fecha_fraud)].contrato.unique().tolist()
        df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud.id_usuario.isin(ctas_fraud)]
        date_inicial = str(pd.to_datetime(fecha_fraud) - pd.DateOffset(months=cant_periodos))
        df_etiquetado_fraud = df_etiquetado_fraud[df_etiquetado_fraud['date'] >= date_inicial]

        # Otras variables
        df_zona = df_etiquetado_fraud.loc[df_etiquetado_fraud.groupby('id_usuario').date.idxmax()]
        df_ttarifa = df_etiquetado_fraud.groupby(['id_usuario']).tipo_tarifa.nunique().reset_index(name='cant_ttarifa')
        df_ttarifa = df_ttarifa.merge(df_zona[['id_usuario', 'departamento', 'municipio', 'zona', 'tipo_tarifa']],
                                      on='id_usuario')

        # Consumo
        cols_ant = [str(x) + '_anterior' for x in range(cant_periodos, 0, -1)]
        df_etiquetado_fraud = df_etiquetado_fraud.pivot_table(index=['id_usuario'], columns=['date'], values='consumo')
        df_etiquetado_fraud.columns = cols_ant
        df_etiquetado_fraud['date_fizcalizacion'] = fecha_fraud
        df_etiquetado_fraud.reset_index(inplace=True)
        df_etiquetado_fraud = df_etiquetado_fraud.merge(df_ttarifa, on=['id_usuario'])
        list_df.append(df_etiquetado_fraud)

    df_etiquetado_wide = pd.concat(list_df)
    df_etiquetado_wide.date_fizcalizacion = pd.to_datetime(df_etiquetado_wide.date_fizcalizacion)
    df_etiquetado_wide = df_etiquetado_wide.merge(df_ordenes[['contrato', 'date', 'cod_mat', 'is_fraud']],
                                                  left_on=['id_usuario', 'date_fizcalizacion'],
                                                  right_on=['contrato', 'date'])

    # Eliminamos aquellos con muchos nulls
    cols_ant = [str(x) + '_anterior' for x in range(cant_periodos, 0, -1)]
    df_etiquetado_wide['cant_null'] = df_etiquetado_wide[cols_ant].isnull().sum(axis=1)
    df_etiquetado_wide['eliminar'] = 0
    df_etiquetado_wide.loc[(df_etiquetado_wide.is_fraud == 0) & (df_etiquetado_wide.cant_null > 0), 'eliminar'] = 1
    df_etiquetado_wide.loc[(df_etiquetado_wide.is_fraud == 1) & (df_etiquetado_wide.cant_null > 6), 'eliminar'] = 1
    df_etiquetado_wide = df_etiquetado_wide[df_etiquetado_wide.eliminar == 0]

    # Add columnas extras
    extra_cols = ['contrato', 'unidad_de_lectura', 'codigo_postal', 'fecha_de_alta', 'no_de_poste', 'tarfia',
                  'multiplicador', 'actividad_economica', 'kw_cont',
                  'medidor_interior', 'folio', 'nivel_de_tension']
    df_exdata = df_exdata[extra_cols].drop_duplicates(subset=['contrato'])
    df_etiquetado_wide = df_etiquetado_wide.merge(df_exdata, left_on=['contrato'], right_on=['contrato'], how='left')
    df_etiquetado_wide['id'] = list(range(0, len(df_etiquetado_wide)))
    return df_etiquetado_wide


def create_prediction_dataset(df_consumo,df_exdata,mes_ultimo_consumo):
    df_exdata = df_exdata.drop_duplicates(subset=['contrato'])
    contratos_a_evaluar = df_exdata.contrato.unique()
    df_consumo = df_consumo[df_consumo.id_usuario.isin(contratos_a_evaluar)]

    cant_periodos = 12
    # tomo las variables del historico hasta donde tengo fecha
    max_date_consumo = str(df_consumo.date.dt.date.max())
    df_evaluation = df_consumo[(df_consumo.date <= max_date_consumo)].copy()
    date_inicial = str(pd.to_datetime(max_date_consumo) - pd.DateOffset(months=cant_periodos - 1))
    df_evaluation = df_evaluation[df_evaluation['date'] >= date_inicial]
    # Otras variables
    df_zona = df_evaluation.loc[df_evaluation.groupby('id_usuario').date.idxmax()]
    df_ttarifa = df_evaluation.groupby(['id_usuario']).tipo_tarifa.nunique().reset_index(name='cant_ttarifa')
    df_ttarifa = df_ttarifa.merge(df_zona[['id_usuario', 'departamento', 'municipio', 'zona', 'tipo_tarifa']],
                                  on='id_usuario')

    # Nuevos consumos
    # fecha_evaluation = '2022-07-01'
    cols_attr_consumo = [x for x in df_exdata.columns if 'consumo_menos' in x]
    df_new_consumo = df_exdata[['contrato'] + cols_attr_consumo].copy()
    df_new_consumo = df_new_consumo.rename(columns={'contrato': 'id_usuario'})
    df_new_consumo.columns = ['id_usuario'] + [str(x) + '_anterior' for x in range(1, cant_periodos + 1)]
    # # # ordenamos las columnas
    cols_ant = [str(x) + '_anterior' for x in range(cant_periodos, 0, -1)]
    df_new_consumo = df_new_consumo[['id_usuario'] + cols_ant]
    df_new_consumo = df_new_consumo.merge(df_ttarifa, on='id_usuario', how='left')
    # # otra variables
    df_new_consumo['cant_null'] = df_new_consumo[cols_ant].isnull().sum(axis=1)
    # df_new_consumo['date_fizcalizacion'] = fecha_evaluation

    # Todas las variables
    cols = ['contrato', 'unidad_de_lectura', 'codigo_postal', 'fecha_de_alta', 'cod_mat', 'no_de_poste', 'tarfia',
            'multiplicador', 'actividad_economica', 'kw_cont',
            'medidor_interior', 'folio', 'nivel_de_tension', 'fecha_de_baja', 'longitud_x', 'latitud_y',
            'indice_de_solvencia']
    df_evaluation = df_new_consumo.merge(df_exdata[cols], left_on=['id_usuario'], right_on=['contrato'], how='left')

    df_evaluation.rename(columns={'id_usuario': 'uc'}, inplace=True)
    # df.rename(columns={'id':'index'}, inplace=True)
    # Solo nos quedamos con aquellos usuarios que tienen al menos 10 meses de tiempo de conexion
    # No esten dados de baja y tengan al menos 6 consumos validos
    ultimo_mes_consumo = pd.to_datetime(mes_ultimo_consumo)
    mes_evaluacion = ultimo_mes_consumo + pd.DateOffset(months=1)
    df_evaluation['cant_meses_conectado'] = ((mes_evaluacion - df_evaluation.fecha_de_alta) / np.timedelta64(1, 'M'))
    df_evaluation['is_baja'] = df_evaluation.fecha_de_baja < mes_evaluacion
    df_evaluation = df_evaluation[
        (df_evaluation.is_baja == False) & (df_evaluation.cant_null <= 6) & (df_evaluation.cant_meses_conectado >= 10)]
    df_evaluation['id'] = list(range(len(df_evaluation)))
    return df_evaluation