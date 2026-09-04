/*
Optional Earth Engine companion.
Live Flood Hub/NWM REST calls stay in Python; upload resulting GeoTIFF/GeoJSON
to GCS/EE assets for GEE analysis.
*/

var basin = ee.FeatureCollection('projects/YOUR_PROJECT/assets/hunting_bayou_basin');
var hand = ee.Image('projects/YOUR_PROJECT/assets/hunting_bayou_hand_2m');
var streams = ee.FeatureCollection('projects/YOUR_PROJECT/assets/hgac_streams');
var depth = ee.Image('projects/YOUR_PROJECT/assets/latest_depth');

Map.centerObject(basin, 12);
Map.addLayer(basin.style({color: '555555', fillColor: '00000000', width: 2}), {}, 'Pilot basin');
Map.addLayer(hand.clip(basin), {min: 0, max: 5}, 'HAND', false);
Map.addLayer(streams.filterBounds(basin), {}, 'Streams', true);
Map.addLayer(depth.clip(basin), {min: 0.05, max: 2.0}, 'Prototype depth', true);

var water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
Map.addLayer(water.clip(basin), {min: 0, max: 100}, 'JRC water occurrence', false);

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(basin)
  .filterDate('2017-08-25', '2017-09-03')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .select('VV');
print('Harvey Sentinel-1 scenes', s1.size());
