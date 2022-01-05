import statsmodels.stats.multitest as multi
import pandas as pd
import numpy as np
import pingouin as pg
def pg_ttest(data, group_col, group1, group2, fdr=0.05, value_col='MS signal [Log2]'):
    '''
    data: long data format with ProteinID as index, one column of protein levels, other columns of grouping.
    '''
    df = data.copy()
    proteins = data.index.unique()
    columns = pg.ttest(x=[1,2], y=[3,4]).columns
    scores = pd.DataFrame(columns=columns)
    for i in proteins:
        df_ttest = df.loc[i]
        x=df_ttest[df_ttest[group_col]==group1][value_col]
        y=df_ttest[df_ttest[group_col]==group2][value_col]
        difference = y.mean()-x.mean()
        result = pg.ttest(x=x, y=y)
        result['protein']=i
        result['difference']=difference
        scores=scores.append(result)
    scores=scores.assign(new_column=lambda x: -np.log10(scores['p-val']))
    scores=scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['p-val'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    scores = scores.set_index('protein')
    return scores