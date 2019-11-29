import seaborn as sns
import numpy as np
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
        linewidth=3, alpha=1.0):
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

    ax = plt.gca()
    ax.add_collection(lc)

    return lc



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

