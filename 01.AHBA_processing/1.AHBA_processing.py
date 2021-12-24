import sys
import os
import numpy as np
import pandas as pd
import pickle
import loompy as lp


path = sys.argv[1] ## path to AHBA data

## load probe information
file_probe_annot = os.path.join(path, "probes.csv")
probe_annot = pd.read_csv(file_probe_annot, sep=",", header=None)
probe_annot.columns = ["probe_id", "probe_name", "gene_id", "gene_symbol", "gene_name", "entrez_id", "chromosome"]

## load re-annotation information (from abagen package)
file_probe_reannot = os.path.join(path, "reannotated.csv.gz")
probe_reannot = pd.read_csv(file_probe_reannot)
# np.bincount(probe_reannot['compare']), probe_reannot.shape
# mismatch(0), match(1), introduced(2), not Given(3)


## load expression data
#normalized_microarray_donor10021  normalized_microarray_donor14380  normalized_microarray_donor15697
#normalized_microarray_donor12876  normalized_microarray_donor15496  normalized_microarray_donor9861
file_rawdata = './rawdata.pkl'
if os.path.exists(file_rawdata):
    probe_annot, probe_reannot, data = pickle.load(open(file_rawdata, 'rb'))
else:
    data = {}
    for subj in [10021, 14380, 15697, 12876, 15496, 9861]:
        file_microarray = os.path.join(path, f"microarray/normalized_microarray_donor{subj}/MicroarrayExpression.csv")
        file_noise = os.path.join(path, f"./microarray/normalized_microarray_donor{subj}/PACall.csv")
        file_annot = os.path.join(path, f"microarray/normalized_microarray_donor{subj}/SampleAnnot.csv")

        data[subj] = {"exprs" : pd.read_csv(file_microarray, header=None, index_col=0),
                      "noise" : pd.read_csv(file_noise, header=None, index_col=0),
                      "annot" : pd.read_csv(file_annot)}
        data[subj]["annot"].index = data[subj]["exprs"].columns
    pickle.dump([probe_annot, probe_reannot, data], open(file_rawdata, 'wb'))

##########################################
## 1. Filtering samples and probes
##########################################
## 1.0 re-annotaion
probe_reannot_filter1 = pd.merge(probe_reannot[['probe_name', 'gene_symbol', 'entrez_id']], 
                                 probe_annot[["probe_name", "probe_id"]], 
                                 on="probe_name", how="left").set_index("probe_id").dropna(subset=["entrez_id"])
probe_reannot_filter1.loc[:, "entrez_id"] = probe_reannot_filter1["entrez_id"].astype(int)

## 1.1 remove sample in Brainstem(BS) and Cerebellum(CB), select reannotated probes
index_probe_keep = probe_reannot_filter1.index
for subj in data:
    index_sample_keep   = ~data[subj]['annot'].slab_type.isin(["BS", "CB"])
    data[subj]['exprs'] = data[subj]['exprs'].loc[index_probe_keep, index_sample_keep]
    data[subj]['noise'] = data[subj]['noise'].loc[index_probe_keep, index_sample_keep]
    data[subj]['annot'] = data[subj]['annot'].loc[index_sample_keep, :]

## 1.2 probe: intensity-based filtering (IBF)
threshold = 0.5
signal_level, n_sample = np.zeros(probe_reannot_filter1.shape[0]), 0
for subj in data:
    signal_level += data[subj]['noise'].sum(axis=1)
    n_sample += data[subj]['noise'].shape[1]
index_probe_keep_IBF = (signal_level / n_sample ) > threshold

probe_reannot_filter2 = probe_reannot_filter1.loc[index_probe_keep_IBF, :]
for subj in data:
    data[subj]['exprs'] = data[subj]['exprs'].loc[index_probe_keep_IBF, :]
    data[subj]['noise'] = data[subj]['noise'].loc[index_probe_keep_IBF, :]

## 1.3 probe: select representative probe (DS method)
region_exprs = [ data[subj]['exprs'].groupby(data[subj]['annot'].structure_id, axis=1).mean().T.rank() for subj in data ]  # sample * probe

## calc DS score for each probes
ds_score = np.zeros(probe_reannot_filter2.shape[0])
for i in range(len(region_exprs)-1):
    exprs1_zscore = (region_exprs[i] - region_exprs[i].mean(axis=0)) / region_exprs[i].std(axis=0)
    for j in range(i+1, len(region_exprs)):
        exprs2_zscore = (region_exprs[j] - region_exprs[j].mean(axis=0)) / region_exprs[j].std(axis=0)
        samples = np.intersect1d(exprs1_zscore.index, exprs2_zscore.index)
        ds_score += (exprs1_zscore.loc[samples, :] * exprs2_zscore.loc[samples, :]).sum(axis=0) / (len(samples) - 1)
ds_score /= sum(range(len(region_exprs)))

## select probe
max_ds_idx = pd.DataFrame([ds_score, probe_reannot_filter2.entrez_id]).T.reset_index().set_index(keys=["entrez_id", "probe_id"]).groupby("entrez_id").idxmax()["Unnamed 0"]
index_probe_keep_DS =  pd.Index(max_ds_idx.apply(lambda x:x[1]).values)
probe_reannot_filter3 = probe_reannot_filter2.loc[index_probe_keep_DS, :]
for subj in data:
    data[subj]['exprs'] = data[subj]['exprs'].loc[index_probe_keep_DS, :]
    data[subj]['noise'] = data[subj]['noise'].loc[index_probe_keep_DS, :]

## 1.4 no normalization
exprs = []
sample_donor = []
sample_annot = []
for subj in data:
    a = data[subj]['exprs'][data[subj]['noise'].astype(bool)].fillna(0).astype(np.float32)
    print(a.shape, data[subj]['exprs'].shape)
    exprs.append( data[subj]['exprs'][data[subj]['noise'].astype(bool)].fillna(0).astype(np.float32) )
    sample_donor.append(np.repeat(subj, data[subj]['exprs'].shape[1]))
    sample_annot.append(data[subj]['annot'].values[:, [4, 5, 7, 8, 9, 10, 11, 12]])
exprs = np.concatenate(exprs, axis=1)
sample_donor = np.concatenate(sample_donor, axis=0)
sample_annot = np.concatenate(sample_annot, axis=0)

row_attrs = {"Gene" : probe_reannot_filter3['gene_symbol'].values,}
col_attrs = {"CellID" :  np.arange(exprs.shape[1])}
lp.create("AHBA_exprs_noNorm.loom", exprs, row_attrs, col_attrs)

exprs = pd.DataFrame(exprs)
exprs.index = probe_reannot_filter3['gene_symbol'].values
exprs.to_csv('AHBA_exprs_noNorm.csv')
print(exprs.shape)



## 1.5 normalization
for subj in data:
    # sample normalization across genes: SRS
    quantile = data[subj]['exprs'].quantile([0.25, 0.5, 0.75], axis=0)
    scale_iqr = quantile.loc[0.75, :] - quantile.loc[0.25, :]/1.35
    scale_robust_sigmoid = 1 / (1 + np.exp(-(data[subj]['exprs'] - quantile.loc[0.5, :])/scale_iqr))

    # gene unitu across samples:
    dat_range = scale_robust_sigmoid.max(axis=1)-scale_robust_sigmoid.min(axis=1)
    gene_norm = ((scale_robust_sigmoid.T - scale_robust_sigmoid.min(axis=1))/dat_range).T

    # set zero to background gene
    data[subj]['exprs'] = gene_norm[data[subj]['noise'].astype(bool)].fillna(0).astype(np.float32)

exprs = []
sample_donor = []
sample_annot = []
for subj in data:
    a = data[subj]['exprs'][data[subj]['noise'].astype(bool)].fillna(0).astype(np.float32)
    print(a.shape, data[subj]['exprs'].shape)
    exprs.append( data[subj]['exprs'][data[subj]['noise'].astype(bool)].fillna(0).astype(np.float32) )
    sample_donor.append(np.repeat(subj, data[subj]['exprs'].shape[1]))
    sample_annot.append(data[subj]['annot'].values[:, [4, 5, 7, 8, 9, 10, 11, 12]])
exprs = np.concatenate(exprs, axis=1)
sample_donor = np.concatenate(sample_donor, axis=0)
sample_annot = np.concatenate(sample_annot, axis=0)

row_attrs = {"Gene" : probe_reannot_filter3['gene_symbol'].values,}
col_attrs = {"CellID" :  np.arange(exprs.shape[1])}
lp.create("AHBA_exprs_noNorm.loom", exprs, row_attrs, col_attrs)

exprs = pd.DataFrame(exprs)
exprs.index = probe_reannot_filter3['gene_symbol'].values
exprs.to_csv('AHBA_exprs_noNorm.csv')
print(exprs.shape)


