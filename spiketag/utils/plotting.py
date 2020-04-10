import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt



def cubehelix_cmap(scheme='dark'):
    if scheme == 'dark':
        cmap = sns.cubehelix_palette(as_cmap=True, dark=0.05, light=1.2, reverse=True);
    else:
        cmap = sns.cubehelix_palette(as_cmap=True, dark=0.05, light=1.2, reverse=False); 
    # sns.palplot(cmap.colors)
    return cmap


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0, ax=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    import matplotlib.collections as mcoll
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return ax



def colorvar(var):
    return (var-var.min())/(var.max()-var.min())



def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments




def colorbar(mappable, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, label=label)




def plot_epoches(time_list, time_span, ax, color=['g', 'r']):
    '''
    plot all trajectories within time_span seconds, starting from the time points in the time_list
    initial at `g` color and end at `r` color
    '''
    for _time in time_list:
        current_pos = pc.binned_pos[np.searchsorted(pc.ts, _time)]
        epoch = np.where((pc.ts<_time+time_span) & (pc.ts>=_time))[0]
        # trajectory
        ax.plot(pc.binned_pos[epoch, 0], pc.binned_pos[epoch, 1])
        # initial
        ax.plot(pc.binned_pos[epoch[0], 0], pc.binned_pos[epoch[0], 1],  color=color[0], marker='o', markersize=15)
        # end
        ax.plot(pc.binned_pos[epoch[-1], 0], pc.binned_pos[epoch[-1], 1], color=color[1], marker='o', markersize=15)



def unit_comparison_all_dim(mod_units, bmi_units, vq, label, unit_No, bg=True, ms=1, ms_scale=8):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(6,3, figsize=(16,5*6), sharex=True, sharey=True)
    for i, dim in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
        _vq = vq[label==unit_No]
        grp_No = np.unique(np.where(label==unit_No)[0])[0]
        fet_attr =['fet0', 'fet1', 'fet2', 'fet3']
        pca_attr =['pca0', 'pca1', 'pca2', 'pca3']

        if bg:
            ax[i][0].plot(mod_units.df[mod_units.df.group_id==grp_No][fet_attr[dim[0]]], 
                          mod_units.df[mod_units.df.group_id==grp_No][fet_attr[dim[1]]], 
                          '.', c='grey', ms=ms, label='other units', alpha=0.7);
            ax[i][2].plot(bmi_units.df[bmi_units.df.group_id==grp_No][fet_attr[dim[0]]], 
                          bmi_units.df[bmi_units.df.group_id==grp_No][fet_attr[dim[1]]], 
                          '.', c='grey', ms=ms, label='other units', alpha=0.7);
            ax[i][0].set_xlim([-0.6,0.6])
            ax[i][0].set_ylim([-0.6,0.6])
            ax[i][1].set_xlim([-0.6,0.6])
            ax[i][1].set_ylim([-0.6,0.6])
            ax[i][2].set_xlim([-0.6,0.6])
            ax[i][2].set_ylim([-0.6,0.6])

        ax[i][0].plot(mod_units.df[mod_units.df.spike_id==unit_No][fet_attr[dim[0]]], 
                      mod_units.df[mod_units.df.spike_id==unit_No][fet_attr[dim[1]]], 
                      'g.', ms=ms, label='unit {}'.format(unit_No));
        ax[i][0].plot(_vq[:,dim[0]], _vq[:,dim[1]], 
                      'r.', ms=ms*ms_scale, label='vq');
        ax[i][0].set_xlabel(pca_attr[dim[0]]); 
        ax[i][0].set_ylabel(pca_attr[dim[1]]);

        ax[i][1].plot(_vq[:,dim[0]], _vq[:,dim[1]], 
                      'r.', ms=ms*ms_scale, label='vq');
        ax[i][1].set_xlabel(pca_attr[dim[0]]);
        ax[i][1].set_ylabel(pca_attr[dim[1]]);

        ax[i][2].plot(bmi_units.df[bmi_units.df.spike_id==unit_No][fet_attr[dim[0]]], 
                      bmi_units.df[bmi_units.df.spike_id==unit_No][fet_attr[dim[1]]], 
                      'g.', ms=ms, label='unit {}'.format(unit_No));
        ax[i][2].plot(_vq[:,dim[0]], _vq[:,dim[1]], 
                      'r.', ms=ms*ms_scale, label='vq');
        ax[i][2].set_xlabel(pca_attr[dim[0]]); 
        ax[i][2].set_ylabel(pca_attr[dim[1]]);
        
        legend_elements = [Line2D([0], [0], color='grey', label='other units'),
                           Line2D([0], [0], color='g', label='unit {}'.format(unit_No)),
                           Line2D([0], [0], color='r', label='vq'),]
        ax[i][1].legend(handles=legend_elements, loc='upper left')
    return fig


def plot_unit_comparison(bmi_folder, model_foder, unit_No, bg=True, ms=2, ms_scale=4):
    '''
    example:
    ------------------------------------------------------------------
    bmi_folder = '/data2/wr93/04082020/BMI'
    model_folder = '/data2/wr93/04082020/manual_sort/spktag/shinsort'
    plot_unit_comparison(bmi_folder, model_folder, unit_No=2, ms=1, ms_scale=10);
    '''
    bmi_file = bmi_folder+'/fet.bin'
    model_file = model_foder+'.pd'
    param_file = model_foder+'.param'
    from spiketag.base import UNIT
    bmi_units = UNIT(); mod_units = UNIT()
    mod_units.load_unitpacket(model_file)
    bmi_units.load_unitpacket(bmi_file)
    param = torch.load(param_file)
    vq,label = param['vq'], param['label']
    fig = unit_comparison_all_dim(mod_units, bmi_units, vq, label, unit_No=unit_No, 
                                  bg=bg, ms=ms, ms_scale=ms_scale)