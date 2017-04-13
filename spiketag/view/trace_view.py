import numpy as np
from .wave_view import wave_view
from phy import gui
from phy.gui.widgets import Table
from ..base.SPKTAG import SPKTAG
from ..utils.utils import get_config_dir

class channel_view(Table):
    '''
        Display the channels info
    '''
    def __init__(self, n_ch, wave_view):
        super(channel_view, self).__init__()

        self.channels = np.arange(n_ch)
        self.wave_view = wave_view    

        @self.connect_
        def on_select(ids):
            if len(ids) > 0:
                self.wave_view.select_ch(ids)

        self.set_rows(self.channels)

class group_view(Table):
    '''
        Display the group info
    '''
    def __init__(self, n_group, wave_view):
        super(group_view, self).__init__()

        self.groups = np.arange(n_group)
        self.wave_view = wave_view

        @self.connect_
        def on_select(ids):
            if len(ids) > 0:
                self.wave_view.select_group(ids)

        self.set_rows(self.groups)

class trace_view(object):
    '''
        Trace view, includes channel info, group info and waves.

        Parameters
        ----------
        probe : Probe
            The type of probe
        data  : array-like
            The raw data of trace. 
        spks  : array-like. Like pivotal, spks[0] is time, spks[1] is channel.
            The time of spks which want to be highlight, if spks is none, no spks will be highlight. 
    '''

    def __init__(self, probe, data, spks=None):
        
        assert probe is not None
        assert data is not None

        self.gui = gui.GUI(name = 'trace', config_dir=get_config_dir())

        self.wave_view = wave_view(fullscreen=True)
        spktag = SPKTAG()
        spktag.probe = probe
        if spks is not None:
            spktag.t = spks[0]
            spktag.ch = spks[1]
        self.wave_view.bind(data, spktag)

        self.channel_view = channel_view(probe.n_ch, self.wave_view)
        self.group_view = group_view(probe.n_group, self.wave_view)
        
        self.gui.add_view(self.wave_view)
        self.gui.add_view(self.channel_view)
        self.gui.add_view(self.group_view)

        # Disable the key of 'h'.
        self.gui.default_actions.disable('show_all_shortcuts')

    def show(self):
        self.gui.show()
        self.channel_view.show()
        self.group_view.show()
        
        
