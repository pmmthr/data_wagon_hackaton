import numpy as np
import datetime

import pandas as pd
from tqdm import tqdm

class Sample:
    def __init__(self):
        wag_param = pd.read_parquet('data/wag_params.parquet').convert_dtypes()

        self.wag_to_rodid = dict(wag_param[['wagnum', 'rod_id']].values)
        self.wag_to_gruz = dict(wag_param[['wagnum', 'gruz']].values)
        self.wag_to_tara = dict(wag_param[['wagnum', 'tara']].values)
        self.wag_to_cnsi_volumek = dict(wag_param[['wagnum', 'cnsi_volumek']].values)
        self.wag_to_kuzov = dict(zip(wag_param['wagnum'].values, ((wag_param['kuzov'] == 2) + 0).values, ))
        self.wag_to_norma_km = dict(wag_param[['wagnum', 'norma_km']].values)

    def get_features(self, date, wagon_dislok, wagon_tr_rem):
        relevant_records = wagon_dislok[(wagon_dislok['plan_date'] < date)]
        relevant_tr_rem = wagon_tr_rem[wagon_tr_rem['rem_month'] < date]

        if len(relevant_records) == 0:
            return [np.nan] * 11

        last_row = relevant_records.iloc[-1]

        if (date - last_row['plan_date']).days < 180:
            relevant_records = relevant_records[relevant_records['plan_date'] > (date - datetime.timedelta(days=180))]

        days_since_last_kaprep = (date - relevant_records.iloc[-1]['date_kap']).days
        days_since_last_deprep = (date - relevant_records.iloc[-1]['date_dep']).days
        days_to_planrep = (last_row['date_pl_rem'] - date).days
        ost_prob = last_row['ost_prob']
        isload = last_row['isload']
        last_fr = last_row['fr_id']
        most_freq_fr = relevant_records['fr_id'].mode()[0]
        probeg_changes = relevant_records['ost_prob'].values[:-1] - relevant_records['ost_prob'].values[1:]

        probeg_changes = relevant_records['ost_prob'].diff(-1).values[:-1]
        probeg_changes = probeg_changes[(~np.isnan(probeg_changes))]
        probeg_changes = probeg_changes[np.where(probeg_changes > 0)]

        date_changes = relevant_records['plan_date'].diff()[1:].apply(lambda x: x.days).values
        date_changes = date_changes[np.where(probeg_changes > 0)]
        probeg_changes /= date_changes

        if len(relevant_tr_rem) > 0:
            num_rem = relevant_tr_rem.shape[0]
            days_since_last_trem = (date - relevant_tr_rem.iloc[-1]['rem_month']).days
        else:
            num_rem = 0
            days_since_last_trem = 1000000

        return [days_since_last_kaprep,
                days_since_last_deprep,
                days_to_planrep,
                ost_prob,
                isload,
                probeg_changes.max(),
                np.mean(probeg_changes),
                last_fr,
                most_freq_fr,
                num_rem,
                days_since_last_trem]


    def get_sample(self, target, dislok, tr_rem, pred=False):
        aggregated_targets = target.groupby('wagnum').agg(list)
        outs = []
        for wagnum in tqdm(aggregated_targets.index):
            wagon_dislok = dislok[dislok['wagnum'] == wagnum]
            wagon_tr_rem = tr_rem[tr_rem['wagnum'] == wagnum]
            row = aggregated_targets.loc[wagnum]

            for i in range(len(row['month'])):
                out = [wagnum, row['month'][i]]
                out += self.get_features(row['month'][i], wagon_dislok, wagon_tr_rem)

                if pred:
                    out.append(-1)
                    out.append(-1)
                else:
                    out.append(row['month'][i])
                    out.append(row['day'][i])

                outs.append(out)
        df = pd.DataFrame(outs, columns=['wagnum', 'date',
                                         'days_since_last_kaprep',
                                         'days_since_last_deprep',
                                         'days_to_planrep',
                                         'ost_prob',
                                         'isload',
                                         'probeg_changes_max',
                                         'probeg_changes_min',
                                         'last_fr',
                                         'most_freq_fr',
                                         'num_rem',
                                         'days_since_last_trem',
                                         'target_month',
                                         'target_day'])

        df['rodid'] = df['wagnum'].apply(lambda x: self.wag_to_rodid[x])
        df['gruz'] = df['wagnum'].apply(lambda x: self.wag_to_gruz[x])
        df['tara'] = df['wagnum'].apply(lambda x: self.wag_to_tara[x])
        df['cnsi_volumek'] = df['wagnum'].apply(lambda x: self.wag_to_cnsi_volumek[x])
        df['kuzov'] = df['wagnum'].apply(lambda x: self.wag_to_kuzov[x])
        df['norma_km'] = df['wagnum'].apply(lambda x: self.wag_to_norma_km[x])

        # каждый раз средние значения по выборке, так как характеристики со временем могут изменяться
        df['days_since_last_kaprep'].fillna(df['days_since_last_kaprep'].mean(), inplace=True)
        df['days_since_last_deprep'].fillna(df['days_since_last_deprep'].mean(), inplace=True)
        df['days_to_planrep'].fillna(df['days_to_planrep'].mean(), inplace=True)
        df['ost_prob'].fillna(df['ost_prob'].mean(), inplace=True)
        df['probeg_changes_max'].fillna(df['probeg_changes_max'].mean(), inplace=True)
        df['probeg_changes_min'].fillna(df['probeg_changes_min'].mean(), inplace=True)
        df['isload'].fillna(df['isload'].mode()[0], inplace=True)
        df['num_rem'].fillna(0, inplace=True)
        df['days_since_last_trem'].fillna(1000000, inplace=True)
        df['last_fr'].fillna(df['last_fr'].mode()[0], inplace=True)
        df['most_freq_fr'].fillna(df['most_freq_fr'].mode()[0], inplace=True)

        return df