import pandas as pd
import geoplotlib
from geoplotlib import layers, core

class ColoredLayer(geoplotlib.layers.BaseLayer):
    def __init__(self, data):
        self.data = data

    def invalidate(self, proj):
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        self.painter = core.BatchPainter()
        self.painter.points(x, y)

    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()

dataset = pd.read_csv(
    'data/accidents.csv',
    header=0
)

dataset = dataset.drop('accident_id', axis=1)

dataset['lat'] = dataset['latitude']
dataset['lon'] = dataset['longitude']

geoplotlib.add_layer(ColoredLayer(dataset))
geoplotlib.show()
