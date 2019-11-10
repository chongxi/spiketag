from spiketag.view import raster_view
import numpy as np

fet_packet = np.fromfile('./fet.bin', dtype=np.int32).reshape(-1,7)
spkid_packet = fet_packet[:, [0,-1]]
spkid_packet = np.delete(spkid_packet, np.where(spkid_packet[:,1]==0), axis=0)

rsview = raster_view(population_firing_count_ON=True)
rsview.title = 'spike raster'
rsview.set_data(spkid_packet)
rsview.show()
rsview.app.run()
