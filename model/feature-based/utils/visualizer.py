import pandas as pd
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.metrics import r2_score

def plot_col(lcia_df, col_name, logx):
    ax =lcia_df[col_name].plot(kind="hist", bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9, logx=logx)
    # ax = lcia_df.hist(column=col_name, bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
    plt.rcParams['font.size'] = 30

    # Despine
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = ax.get_yticks()
    for tick in vals:
        ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    ax.set_title("")

    # Set x-axis label
    ax.set_xlabel(col_name, labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel("Counts", labelpad=20, weight='bold', size=25)

    # Format y-axis label
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    
    return ax

def plot_parity(ax, ytrain, result_train, ytest, result, title, mae):
    # plt.rcParams['ytick.labelsize'] = 25
    # plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['font.size'] = 20
    # plt.rcParams['font.family'] = 'Palatino'
    font_axis_publish = {
    # 'fontname':'Palatino',
    # 'color':  'black',
    # 'weight': 'bold',
    'size': 25}
    x=ytrain.reshape((-1,1))
    y=result_train.reshape((-1,1))
    bounds = (min(x.min(), y.min()) - int(0.1 * y.min()), max(x.max(), y.max())+ int(0.1 * y.max()))

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect("equal", adjustable="box")

    #reference parity line
    ax.plot([0, 1], [0, 1], "--",color='grey',transform=ax.transAxes)
    #train-val-test
    mae_train=np.abs(ytrain.reshape((-1,1))-result_train.reshape((-1,1))).mean()
    mae_test=mae.mean()#np.abs(ytest.reshape((-1,1))-result.reshape((-1,1))).mean()

    r2_train=r2_score(ytrain.reshape((-1,1)), result_train.reshape((-1,1)))
    r2_test=r2_score(ytest.reshape((-1,1)), result.reshape((-1,1)))

    ax.plot(ytrain.reshape((-1,1)), result_train.reshape((-1,1)),"o", color='#168aad',\
    alpha=1,ms=6, markeredgewidth=0.0,label='Train MAE: %0.2f'%(mae_train))

    ax.plot(ytest.reshape((-1,1)), result.reshape((-1,1)),"o", color='#f94144', \
    alpha=1 ,ms=6, markeredgewidth=0.0,label='Test MAE: %0.2f'%(mae_test))
    #legends and text
    L=ax.legend(ncol=1, loc=2, frameon=False,markerscale=3, prop={'size': 25}) #
    plt.setp(L.texts, family='Palatino')


    #axis
    ax.set_xlabel('Actual', fontdict=font_axis_publish)#, fontdict=font_axis_publish
    ax.set_ylabel('Predicted', fontdict=font_axis_publish)
    # ax.set_title(title)
    ax.tick_params(axis="y",direction="in", pad=5,size=4,width=1.5,which='both')
    ax.tick_params(axis="x",direction="in", pad=10,size=4,width=1.5,which='both')
    ax.tick_params(axis="y",direction="in", pad=5,size=6,width=2)
    ax.tick_params(axis="x",direction="in", pad=10,size=6,width=2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)