from .MUA import MUA
from .SPK import SPK
from .FET import FET
from .CLU import CLU
import numpy as np
import json
import pickle


class SPKTAG(object):
	def __init__(self, probe=None, pivotal=None, spk=None, fet=None, clu=None, filename=None):
		'''
		pivotal: mua.pivotal_pos (a numpy array), s1st row is time, 2nd row is ch
		spk    : spk object
		fet    : fet object
		clu    : dictionary of clu object (each item is a channel based clu object)
		'''
		if filename is not None: # load from file
			self.fromfile(filename)
		elif probe is not None and pivotal is not None:                    # construct
		        self.probe  = probe	
			arg_piv     = np.lexsort((pivotal[0], pivotal[1])) 
			piv         = pivotal[:, arg_piv]
			self.nspk   = arg_piv.shape[0]
			self.t      = piv[0]
			self.ch     = piv[1]
                        self.group  = self._ch2group(self.ch)
			self.spk    = spk 
			self.fet    = fet 
			self.clu    = clu 
			self.meta   = {}
			self.spklen = spk.spklen
			self.fetlen = fet.fetlen
			self.dtype  = [('t', 'int32'),
                                       ('ch', 'int32'),
			               ('group','int32'),  
			               ('spk', 'f4', (self.spklen, self.probe.len_group)), 
			               ('fet','f4',(self.fetlen,)),
			               ('clu','int32')]
                        self.build_meta()
			self.build_spktag()
		else:
			pass


	def build_meta(self):
                self.meta["probe"] = pickle.dumps(self.probe)
		self.meta["nspk"] = self.nspk
		self.meta["fetlen"] = self.fetlen
		self.meta["spklen"] = self.spklen


	def build_spktag(self):
		spktag = np.zeros(self.nspk, dtype=self.dtype)
		spktag['t']  = self.t
                spktag['ch'] = self.ch
		spktag['group'] = self.group
		for g in range(self.probe.n_group):
		    spktag['spk'][spktag['group']==g] = self.spk[g]
		    spktag['fet'][spktag['group']==g] = self.fet[g]        
		    spktag['clu'][spktag['group']==g] = self.clu[g].membership
		self.spktag = spktag

        def fetch_spk_times(self, group):
            return np.sort(self.t[self.group == group]) 

	def update(self, spk, fet, clu):
		self.spk = spk
		self.fet = fet
		self.clu = clu
		self.build_spktag()	


	def tofile(self, filename):
		with open(filename+'.meta', 'w') as metafile:
			json.dump(self.meta, metafile)
		self.spktag.tofile(filename)


	def fromfile(self, filename):
		with open(filename+'.meta', 'r') as metafile:
			self.meta = json.load(metafile)
                self.probe = pickle.loads(self.meta['probe'])
		self.nspk   = self.meta['nspk']
		self.spklen = self.meta['spklen']
		self.fetlen = self.meta['fetlen']
		self.dtype = [('t', 'int32'), 
                              ('ch', 'int32'),
		              ('group', 'int32'),  
		              ('spk', 'f4', (self.spklen, self.probe.len_group)), 
		              ('fet', 'f4',(self.fetlen,)),
		              ('clu', 'int32')]
		self.spktag = np.fromfile(filename, dtype=self.dtype)
		self.t  = self.spktag['t']
                self.ch = self.spktag['ch']
		self.group = self.spktag['group']


	def tospk(self):
		spkdict = {}
		for g in range(self.probe.n_group):
			spkdict[g] = self.spktag['spk'][self.group==g]
		return SPK(spkdict)		


	def tofet(self):
		fetdict = {}
		for g in range(self.probe.n_group):
			fetdict[g] = self.spktag['fet'][self.group==g]
		return FET(fetdict)		


	def toclu(self):
		cludict = {}
		for g in range(self.probe.n_group):
			cludict[g] = CLU(self.spktag['clu'][self.group==g])
		return cludict

        def _ch2group(self, ch):
            ch2group_v = np.vectorize(self.probe.belong_group)
            return ch2group_v(ch)
