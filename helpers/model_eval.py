import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick



def eval_model(ytrue, preds) -> pd.DataFrame:
    evaldf = pd.DataFrame({'real': ytrue, 'pred': preds, 'ape': (preds-ytrue)/ytrue})
    sns.set('notebook', font_scale=2)
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.kdeplot(evaldf.ape * 100, color='blue', fill='red', ax=ax)
    #sns.rugplot(evaldf.ape * 100, ax=ax, color='r')
    pct_lower = np.percentile(evaldf.ape, 5) * 100
    pct_upper = np.percentile(evaldf.ape, 95) * 100
    ax.axvline(evaldf.ape.min()*100, color='r', ls='--')
    ax.axvline(evaldf.ape.max()*100, color='g', ls='--')
    ax.axvline(pct_lower, color='k', ls='--')
    ax.axvline(pct_upper, color='k', ls='--')
    ax.set_axis_below = True
    ax.axvspan(pct_lower, pct_upper, color='k', alpha=0.1)
    ax.axvspan(evaldf.ape.min()*100, pct_lower, color='red', alpha=0.1)
    ax.axvspan(pct_upper, evaldf.ape.max()*100, color='green', alpha=0.1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set(yticks=[], xlabel=r"% Error of model", title=f"CI = [{pct_lower:.1f}; {pct_upper:.1f}]")
    sns.despine()
    plt.show()
    return evaldf
