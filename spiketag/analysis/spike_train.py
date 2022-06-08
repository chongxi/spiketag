import numpy as np
import matplotlib.pyplot as plt


class spike_unit(object):
    """
    a unit of spike train
    """
    def __init__(self, spk_id, spk_time):
        self.id = spk_id
        self.spk_time = spk_time

    def align_by(self, event_time):
        """
        align the spike time of this unit to the event time

        will generate:
            self.spk_time_algined: the aligned spike time (s)
            self.n_events: the number of trials
        """
        self.event_time = event_time
        self.n_events = len(event_time)
        self.spk_time_algined = self.spk_time - self.event_time.reshape(-1, 1)
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


class spike_train(object):
    """
    This class is used for spike train analysis.

    Args:

    spike_info: a numpy array of spike times of N neurons (first row:spk_time, second row:spk_id)
    For example:
        array([[ 0.   ,  0.   ,  0.   , ...,  0.398,  0.398,  0.399],
               [41.   , 43.   , 71.   , ..., 70.   , 77.   , 10.   ]],
        dtype=float32)

    Attributes:
        spike_time: a numpy array of spike times of N neurons (spk_time in #samples)
        spike_id  : a numpy array of spike id    of N neurons
        unit: a dictionary of spike_unit
        unit[i] is the spike_unit of neuron i
        unit[i].id: the id of neuron i
        unit[i].spk_time: the spike time of neuron i
    """
    
    def __init__(self, spike_info):
        """
        This class is used for spike train analysis.
        """
        self.spike_time = spike_info[0]
        self.spike_id = spike_info[1]
        self.unit = {}
        for i in np.unique(self.spike_id):
            self.unit[i] = spike_unit(spk_id = i, 
                                      spk_time = self.spike_time[self.spike_id == i])
    
    def __getitem__(self, i):
        return self.unit[i]
