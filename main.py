import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sample import Sample

import warnings
warnings.filterwarnings("ignore")

target_test = pd.read_csv('y_predict.csv').convert_dtypes()
target_test['month'] = pd.to_datetime(target_test['month'])

dislok = pd.read_parquet('data/dislok_wagons.parquet').convert_dtypes()
tr_rem = pd.read_parquet('data/tr_rems.parquet').convert_dtypes()

df_test = Sample().get_sample(target_test, dislok, tr_rem, pred=True)

sfs_features = ['days_since_last_deprep',
 'days_to_planrep',
 'ost_prob',
 'isload',
 'probeg_changes_min',
 'last_fr',
 'most_freq_fr',
 'num_rem',
 'days_since_last_trem',
 'rodid',
 'cnsi_volumek',
 'kuzov',
 'norma_km']

test_pred = df_test[['wagnum', 'date']].rename(columns={'date': 'month'}).copy()

lgbm_best_day = joblib.load('models/lgdm_best_day_final.pkl')
lgbm_best_month = joblib.load('models/lgbm_best_month_final.pkl')


final_pred = df_test[['wagnum', 'date']].rename(columns={'date': 'month'}).copy()

final_pred['target_month'] = (lgbm_best_month.predict_proba(df_test[sfs_features])[:, 1] > 0.5) + 0
final_pred['target_day'] = (lgbm_best_day.predict_proba(df_test[sfs_features])[:, 1] > 0.5) + 0
final_pred['month'] = pd.to_datetime(final_pred['month'])
final_pred = df_test.iloc[:, :2].merge(final_pred, on=['wagnum', 'month'])

final_pred.to_csv("FINAL_PRED.csv")
