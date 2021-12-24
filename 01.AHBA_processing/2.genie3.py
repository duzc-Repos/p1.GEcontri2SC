import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import datetime as dt

AHBA_exprs = pd.read_csv('./AHBA_exprs_noNorm.csv', index_col=0).T
target_names = AHBA_exprs.columns
tf_names = [ tf.strip() for tf in open('./0.ref/hs_hgnc_tfs.txt') ]
tf_names = target_names[target_names.isin(tf_names)]
tf_exprs = AHBA_exprs[tf_names]

ET_KWARGS = {
    'n_jobs': 4,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}

RF_KWARGS = {
    'n_jobs': 4,
    'n_estimators': 1000,
    'max_features': 'sqrt'
}
SGBM_KWARGS = {
    'learning_rate': 0.01,
    'n_estimators': 1000,  # can be arbitrarily large
    'max_features': 0.1,
    'subsample': 0.9
}

for k, t_name in enumerate(target_names):
    if t_name in tf_names:
        select_idx = tf_names.isin([t_name])
        train_x = tf_exprs.loc[:, ~select_idx]
    else:
        train_x = tf_exprs
    train_y = AHBA_exprs[t_name]

    #et = ExtraTreesRegressor(random_state=666, **ET_KWARGS)
    #et.fit(train_x, train_y)
    #links_df = pd.DataFrame({'TF': train_x.columns, 'importance': et.feature_importances_})

    rf = RandomForestRegressor(random_state=666, **RF_KWARGS)
    rf.fit(train_x, train_y)
    links_df = pd.DataFrame({'TF': train_x.columns, 'importance': rf.feature_importances_})

    #gbm = GradientBoostingRegressor(random_state=666, **SGBM_KWARGS)
    #gbm.fit(train_x, train_y)
    #links_df = pd.DataFrame({'TF': train_x.columns, 'importance': gbm.feature_importances_*gbm.n_estimators})

    links_df['target'] = t_name
    clean_links_df = links_df[links_df.importance > 0].sort_values(by='importance', ascending=False)
    clean_links_df = clean_links_df[['TF', 'target', 'importance']]
    clean_links_df.to_pickle(f'./2.grn/{t_name}_grn.pkl')
    if k % 10 == 0:
        print(dt.datetime.now(), k, t_name, (t_name in tf_names), sep="\t")
    #break




