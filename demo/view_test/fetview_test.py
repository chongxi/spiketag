from spiketag.view import scatter_3d_view
import numpy as np

x = np.random.randn(20,4)
fet_view = scatter_3d_view()
fet_view.set_data(x)
fet_view.show()
fet_view.app.run()
