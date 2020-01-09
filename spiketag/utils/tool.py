import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import seaborn as sns
# sns.set_style("darkgrid", {"axes.facecolor": ".85"})



def comet(pos, fs, pos_compare=None, start=1, stop=None, length=300, interval=1, markersize=25, blit=True, player=False, dpi=200, **kwargs):
    '''
    ani = comet2(pos=pos, pos_compare=pos[300:, :], start=300, stop=pos.shape[0], length=300, interval=1, 
                 blit=True)
    '''
    if stop is None:
        if pos_compare is not None:
            stop = min(pos.shape[0], pos_compare.shape[0])
        else:
            stop = pos.shape[0] 

    fig, ax = plt.subplots(dpi=dpi)
    range_min, range_max = pos.min(axis=0), pos.max(axis=0)
    margin = (range_min[0]+range_max[0])/2*0.2
    space  = (range_max[0]-range_min[0])//10
    ax.set_xlim((range_min[0]-margin, range_max[0]+margin))
    ax.set_ylim((range_min[1]-margin, range_max[1]+margin))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(space))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(space))

    time_text    = ax.text(.01, .96, '', transform=ax.transAxes)

    point1, = ax.plot([],[], marker="o", color="blue", ms=markersize, alpha=.5)
    line1,  = ax.plot([], [], lw=2, label='pos1')

    if pos_compare is not None:
        point2, = ax.plot([],[], marker="o", color="crimson", ms=markersize, alpha=.5)
        line2,  = ax.plot([], [], lw=2, label='pos2')

    plt.legend()

    def init():
        time_text.set_text('')
        point1.set_data([], []) 
        line1.set_data([], []) 
        if pos_compare is not None:
            point2.set_data([], []) 
            line2.set_data([], []) 
            return point1, point2, line1, line2, time_text 
        else:
            return point1, line1, time_text


    def update(i):
        time_text.set_text("time: {0:.3f} sec".format(i/fs)) # resolution 0.033 sec for 30 fps
        if i-length<0:
            a = 0
        else:
            a = i-length
        x1 = pos[a:i, 0]
        y1 = pos[a:i, 1]
        line1.set_data(x1, y1) 
        point1.set_data(x1[-1], y1[-1])

        if pos_compare is not None:
            x2 = pos_compare[a:i, 0]
            y2 = pos_compare[a:i, 1]
            line2.set_data(x2, y2) 
            point2.set_data(x2[-1], y2[-1])
            return point1, point2, line1, line2, time_text
        else:
            return point1, line1, time_text

    if player is True:
        ani = Player(fig, init_func=init, func=update, mini=start, maxi=stop, interval=interval, blit=blit, **kwargs)
    else:
        ani = animation.FuncAnimation(fig, init_func=init, func=update, frames=np.arange(start, stop), interval=interval, blit=blit, **kwargs)
    return ani



def decoder_viewer(pos, fs, pos_compare=None, mua_count=None, start=1, stop=None, length=300, interval=1, markersize=27, blit=True, player=False, dpi=100, **kwargs):
    '''
    ani = comet2(pos=pos, pos_compare=pos[300:, :], start=300, stop=pos.shape[0], length=300, interval=1, 
                 blit=True)
    '''
    if stop is None:
        if pos_compare is not None:
            stop = min(pos.shape[0], pos_compare.shape[0])
        else:
            stop = pos.shape[0] 

    fig= plt.figure(dpi=dpi, figsize=(10,15))
    gs = fig.add_gridspec(nrows=3, ncols=1, wspace=0.05)
    ax = [[],[]]
    ax[0] = fig.add_subplot(gs[:2, 0])
    ax[1] = fig.add_subplot(gs[2, 0])

    range_min, range_max = pos.min(axis=0), pos.max(axis=0)
    margin = (range_min[0]+range_max[0])/2*0.25
    space  = (range_max[0]-range_min[0])//10
    ax[0].set_xlim((range_min[0]-margin, range_max[0]+margin))
    ax[0].set_ylim((range_min[1]-margin, range_max[1]+margin))
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(space))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(space))
    
    ax[1].set_ylim([mua_count.min(), mua_count.max()])
    ax[1].axhline(np.median(mua_count), ls='-.', c='m')
    ax[1].set_ylabel('#total spikes/frame')

    time_text = ax[0].text(.01, .96, '', transform=ax[0].transAxes)
    time_text.set_alpha(.7)

    point1, = ax[0].plot([],[], marker="o", color="blue", ms=markersize, alpha=.5)
    line1,  = ax[0].plot([], [], lw=2, label='pos1')
    
    line_mua_rate, = ax[1].plot([], [], '-o', lw=2)

    if pos_compare is not None:
        point2, = ax[0].plot([],[], marker="o", color="crimson", ms=markersize, alpha=.5)
        line2,  = ax[0].plot([], [], lw=2, label='pos2')

    plt.legend()

    def init():
        time_text.set_text('')
        point1.set_data([], []) 
        line1.set_data([], []) 
        line_mua_rate.set_data([], [])
        if pos_compare is not None:
            point2.set_data([], []) 
            line2.set_data([], []) 
            return point1, point2, line1, line2, time_text, line_mua_rate
        else:
            return point1, line1, time_text, line_mua_rate


    def update(i):
        time_text.set_text("time: {0:.3f} sec; frame:{1}".format(i/fs, i)) # resolution 0.033 sec for 30 fps
        if i-length<0:
            a = 0
            ax[1].set_xlim([0, length])
        else:
            a = i-length
            ax[1].set_xlim([a, i])
        x1 = pos[a:i, 0]
        y1 = pos[a:i, 1]
        line1.set_data(x1, y1) 
        point1.set_data(x1[-1], y1[-1])
        
        line_mua_rate.set_data(np.arange(a,i), mua_count[a:i])

        if pos_compare is not None:
            x2 = pos_compare[a:i, 0]
            y2 = pos_compare[a:i, 1]
            line2.set_data(x2, y2) 
            point2.set_data(x2[-1], y2[-1])
            return point1, point2, line1, line2, time_text, line_mua_rate
        else:
            return point1, line1, time_text, line_mua_rate

    if player is True:
        ani = Player(fig, init_func=init, func=update, mini=start, maxi=stop, interval=interval, blit=blit, **kwargs)
    else:
        ani = animation.FuncAnimation(fig, init_func=init, func=update, frames=np.arange(start, stop), interval=interval, blit=blit, **kwargs)
    return ani


class _slider(matplotlib.widgets.Slider):
    def draw_val(self, val):
        """
        Set slider value to *val*

        Parameters
        ----------
        val : float
        """
        xy = self.poly.xy
        if self.orientation == 'vertical':
            xy[1] = 0, val
            xy[2] = 1, val
        else:
            xy[2] = val, 1
            xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        # if self.drawon:
        #     self.ax.figure.canvas.draw_idle()
        self.val = val
        # if not self.eventson:
        #     return



class Player(animation.FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, mini=0, maxi=100, loc=(0.125, 0.92), repeat=False, blit=True, **kwargs):
        self.i = mini+1
        self.min=mini
        self.max=maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.init_gui(loc)
        self.step_len=1
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        animation.FuncAnimation.__init__(self, self.fig, self.func, frames=self.play(), 
                                           init_func=init_func, fargs=fargs, blit=blit,
                                           save_count=save_count, **kwargs )    

    def press(self, event):
        # print('press', event.key)
        if event.key == ' ':
            if self.runs:
                self.stop()
            else:
                self.start()

    def play(self):
        while self.runs:
            self.i = self.i + (self.forwards-(not self.forwards)) * self.step_len
            if self.i > self.min and self.i < self.max:
                self.slider.draw_val(self.i)
                # self.fig.canvas.draw_idle()
                yield self.i
            else:
                self.stop()
                yield self.i

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()


    def forward(self, event=None):
        self.forwards = True
        self.start()
        
    def backward(self, event=None):
        self.forwards = False
        self.start()
        
    def oneforward(self, event=None):
        # self.forwards = True
        self.step_len += 1
        # self.onestep()
        
    def onebackward(self, event=None):
        if self.step_len > 1:
            self.step_len -= 1
        # self.forwards = False
        # self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i+=self.step_len
        elif self.i == self.max and not self.forwards:
            self.i-=self.step_len
        self.func(self.i)
        self.fig.canvas.draw_idle()

    def init_gui(self, loc):
        playerax = self.fig.add_axes([loc[0],loc[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = self.fig.add_axes([loc[0] + 0.33, loc[1], 0.35, 0.04])
        self.button_oneback = matplotlib.widgets.Button(playerax, label=u'$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label=u'$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label=u'$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label=u'$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label=u'$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)

        self.slider = _slider(sliderax, label='frame', valmin=self.min, valmax=self.max, valinit=self.i, valstep=1)
        self.slider.on_changed(self.set_frame)

    def set_frame(self, frame_No):
        self.frame = int(frame_No)
    
    @property
    def frame(self):
        return self.i
        
    @frame.setter
    def frame(self, frame_No):
        self.i = frame_No
        self.start()



def plot_err_2d(dec_pos, real_pos, err, dt, N=5000, err_percentile = 90, err_thr = 1/4): 
    '''
    plot and show the decoding performance in 2D maze

    Parameters
    ----------
    dec_pos : (n, 2) matrix
        decoded 2D trajectory 
    real_pos : (n, 2) matrix
        true 2D trajectory
    err : (n, 2) matrix
        normalized decoding error (0-1) in x-axis andy-axis
    dt : scalar
        time step of the decoder
    N : scalar
        the last N samples to display
    err_percentile : 
        the value that x% of the error is below
    err_thr : 
        plot a vertical red line indicating the goal of err_percentile.

    Returns
    -------
    The figure
    '''
    
    from sklearn.metrics import r2_score
    r2 = r2_score(real_pos, dec_pos, multioutput='raw_values')
    
    fig = plt.figure(1, figsize=(20,12))
    grid = plt.GridSpec(8, 4, wspace=.5, hspace=1)
    ax = {}
    
    # part I: last N examples of decoding vs true
    ax[0,0] = fig.add_subplot(grid[:3,:2])
    ax[0,1] = fig.add_subplot(grid[:3,2:])
    ax[1,0] = fig.add_subplot(grid[3:,:2])
    ax[1,1] = fig.add_subplot(grid[3:,2:])
    N = int(N)
    t = np.linspace(0, N*dt, N)
    ax[0,0].plot(t, real_pos[:,0][-N:]);
    ax[0,0].plot(t, dec_pos[:,0][-N:]);
    ax[0,0].set_xlabel('Time(secs)')
    ax[0,0].set_title('x-axis r2 score:{0:.2f}'.format(r2[0]), fontsize=18)
    ax[0,1].plot(t, real_pos[:,1][-N:]);
    ax[0,1].plot(t, dec_pos[:,1][-N:]);
    ax[0,1].set_title('y-axis r2 score:{0:.2f}'.format(r2[1]), fontsize=18)
    ax[0,1].set_xlabel('Time(secs)')
    ax[0,0].legend(['True', 'Decoded'], loc=[0.05,1], fontsize=15);

    # part II: error distribution
    # x-axis
    sns.distplot(err[:,0], kde=False, ax=ax[1,0]);
    ax[1,0].axvspan(xmin = 0,
                xmax = np.percentile(err[:,0], err_percentile),
                ymin=0, ymax=1, alpha=.2, color='c');
    ax[1,0].axvline(err_thr, color='r')
    ax[1,0].set_xlim([0,1])
    ax[1,0].legend(['{0:.2f}% of x-axis'.format(err_thr*100), 
                    '{0} percentile of error'.format(err_percentile)], fontsize=15)
    ax[1,0].set_xlabel('decoding error (x) distribution (normalized)')
    
    from scipy import signal
    win = signal.blackman(500)
    win /= win.sum()
    t = np.linspace(0, len(err)*dt, len(err))
    xerr_time_ax = fig.add_axes([.3, .34, .15, .1])
    xerr_time_ax.plot(t, err[:,0], alpha=.5, color='w')
    xerr_time_ax.plot(t, np.convolve(err[:,0].ravel(), win, mode='same'), 'm', lw=2)
    xerr_time_ax.set_title('decoding error (x) vs time')
    xerr_time_ax.set_xlabel('time(secs)')

    # y-axis
    sns.distplot(err[:,1], kde=False, ax=ax[1,1]);
    ax[1,1].axvspan(xmin = 0,
                xmax = np.percentile(err[:,0], err_percentile),
                ymin=0, ymax=1, alpha=.2, color='c');
    ax[1,1].axvline(err_thr, color='r')
    ax[1,1].set_xlim([0,1])
    ax[1,1].legend(['{0:.2f}% of y-axis'.format(err_thr*100), 
                    '{0} percentile of error'.format(err_percentile)], fontsize=15)
    ax[1,1].set_xlabel('decoding error (y) distribution (normalized)')
    
    yerr_time_ax = fig.add_axes([.72, .34, .15, .1])
    yerr_time_ax.plot(t, err[:,1], alpha=.5, color='w')
    yerr_time_ax.plot(t, np.convolve(err[:,1].ravel(), win, mode='same'), 'm', lw=2)
    yerr_time_ax.set_title('decoding error (y) vs time')
    yerr_time_ax.set_xlabel('time(secs)')
    return fig

