import numpy as np
import matplotlib.pyplot as plt

class spike_unit(object):
    """
    a unit of spike train
    """
    def __init__(self, spk_id, spk_time):
        self.id = spk_id
        self.t = spk_time

    def __len__(self):
        '''
        number of spikes
        '''
        return len(self.t)

    def isi(self):
        '''
        inter spike interval
        '''
        return np.diff(self.t)

    def align_by(self, event_time):
        """
        align the spike time of this unit to the event time

        will generate:
            self.spk_time_algined: the aligned spike time (s)
            self.n_events: the number of trials
        """
        self.event_time = event_time
        self.n_events = len(event_time)
        self.spk_time_algined = self.t - self.event_time.reshape(-1, 1)
        assert(self.spk_time_algined.shape[0] == self.n_events)
        return self.spk_time_algined

    def peth(self, event_time, window, binsize):
        """
        calculate the peri-event time histogram(peth) of this unit

        args (all in second):
            event_time: the time of event (s)
            window: the time window to calculate the peth (s)
            binsize: the size of bin to calculate spike counts in each bin (s) 
        
        return:
            peth_time: the time of peth (s)
            peth_mean: the mean of peth (spikes/s)
            peth_sem: the standard error of mean of peth (spikes/s)
        """
        bins = np.arange(window[0], window[1]+binsize, binsize)
        self.align_by(event_time)
        # TODO: np.apply_along_axis is slow, need to improve with JIT
        self.peth_spk_counts = np.apply_along_axis(func1d=np.histogram, axis=1, 
                                                   arr=self.spk_time_algined, 
                                                   bins=bins, range=(window[0], window[1]))[:, 0]
        self.peth_time = bins[:-1] + binsize/2
        self.peth_mean = self.peth_spk_counts.sum(axis=0) / binsize / self.n_events # spikes/s averaged over trials (n_events)
        self.peth_sem  = self.peth_spk_counts.std(axis=0) / binsize / np.sqrt(self.n_events) 
        return self.peth_time, self.peth_mean, self.peth_sem

    def eventplot(self, event_time):
        """
        plot the spike time of this unit (algined by event_time), considering each event as a trial
        """
        self.align_by(event_time)
        plt.eventplot(self.spk_time_algined, lineoffsets=range(self.n_events), linelengths=0.5);
        plt.axvline(0, c='r', ls='-.')
        plt.xlabel('Time (s)')
        plt.ylabel('Trials')
        plt.show()

    def plot_peth(self, event_time, window, binsize, event_name='trigger'):
        """
        plot the peth of this unit

        args:
            event_time: dict, {event_type: event_time} (event_time: s)
            window: the time window to calculate the peth (s)
            binsize: the size of bin to calculate spike counts in each bin (s)
            event_name: the name of event (trigger, cue_times, laser_times etc.)
        """
        plt.axvline(0, c='r', ls='-.', label=event_name)
        if type(event_time) != dict:
            event_time = {'': event_time}
        for _event_type, _event_time in event_time.items():
            self.peth(_event_time, window, binsize)
            plt.plot(self.peth_time, self.peth_mean, label=_event_type)
            plt.fill_between(self.peth_time, 
                             self.peth_mean - self.peth_sem, 
                             self.peth_mean + self.peth_sem, 
                             alpha=0.3)
        plt.legend(loc='best')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing rate (spikes/s)')
        plt.title('unit %d firing rate PETH' % self.id)
        plt.show()

    def to_neo(self, time_units='sec', t_start=None, t_stop=None):
        import neo
        if t_start is None:
            t_start = self.t[0]
        if t_stop is None:
            t_stop = self.t[-1]
        neo_st = neo.SpikeTrain(self.t, units=time_units, t_start=t_start, t_stop=t_stop)
        return neo_st
