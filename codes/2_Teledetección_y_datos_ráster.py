# Databricks notebook source
!pip install --upgrade geopandas

!pip install --upgrade pyshp

!pip install --upgrade shapely

!pip install --upgrade descartes

!pip install --upgrade rtree

!pip install --upgrade pygeos

!pip install --upgrade rasterio

!sudo apt install libspatialindex-dev

!sudo apt install tree

# COMMAND ----------

import geopandas as gpd

# COMMAND ----------

from google.colab import drive
drive.mount('/content/drive')

# COMMAND ----------

# MAGIC %md
# MAGIC Recortar por un polígono genérico

# COMMAND ----------

working_directory = "/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/"

# COMMAND ----------

gdf_rectangulo = gpd.read_file(working_directory + 'IMAGENES/PLANET SCOPE/Grilla_Planet_Scope_2020.shp',crs={'init' :'epsg:4326'})
gdf_rectangulo.crs='epsg:4326'

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Sesión 5 - Archivos ráster**

# COMMAND ----------

# MAGIC %md
# MAGIC Descarga de imágenes
# MAGIC 
# MAGIC En este notebook se describen los pasos para descargar las imágenes de Sentinel-2, para luego trabajar sobre el dataset creado en la sesión 4.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imágenes satelitales: Sentinel-2

# COMMAND ----------

# MAGIC %md
# MAGIC Las imágenes multiespectrales de Sentinel-2 tienen 13 bandas:
# MAGIC 
# MAGIC * 4 bandas de 10m: RGB y NIR
# MAGIC * 6 bandas de 20m: SWIR, red edge, etc.
# MAGIC * 3 bandas de 60m
# MAGIC 
# MAGIC La revisita de la constelación es de aproximadamente 5 días, por lo cual resultan muy útiles a la hora de analizar una serie de tiempo, o tener un monitoreo a gran escala de determinadas áreas de interés.
# MAGIC Para más información puede consultar la [Guía de usuario](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi).
# MAGIC 
# MAGIC | Sentinel-2 Bands              | Central Wavelength (µm) | Resolution (m) |
# MAGIC |-------------------------------|-------------------------|----------------|
# MAGIC | Band 1 - Coastal aerosol      |                   0.443 |             60 |
# MAGIC | Band 2 - Blue                 |                    0.49 |             10 |
# MAGIC | Band 3 - Green                |                    0.56 |             10 |
# MAGIC | Band 4 - Red                  |                   0.665 |             10 |
# MAGIC | Band 5 - Vegetation Red Edge  |                   0.705 |             20 |
# MAGIC | Band 6 - Vegetation Red Edge  |                    0.74 |             20 |
# MAGIC | Band 7 - Vegetation Red Edge  |                   0.783 |             20 |
# MAGIC | Band 8 - NIR                  |                   0.842 |             10 |
# MAGIC | Band 8A - Vegetation Red Edge |                   0.865 |             20 |
# MAGIC | Band 9 - Water vapour         |                   0.945 |             60 |
# MAGIC | Band 10 - SWIR - Cirrus       |                   1.375 |             60 |
# MAGIC | Band 11 - SWIR                |                    1.61 |             20 |
# MAGIC | Band 12 - SWIR                |                    2.19 |             20 |
# MAGIC 
# MAGIC 
# MAGIC Generalmente se trabaja con dos productos, con corrección radiométrica y ortorectificación aplicada.
# MAGIC 
# MAGIC 1. **L1C**: Top-of-atmosphere (ToA) reflectance
# MAGIC 2. **L2A**: Bottom-of-atmosphere (BoA) reflectance (también llamada Top-of-canopy)
# MAGIC 
# MAGIC L2A tiene **corrección atmosférica**, y es necesario para analizar índices de vegetación como NDVI o NDWI. La mayoría de las veces es preferible trabajar en ese nivel.  Uno puede generar un producto de nivel L2A a partir de L1C utilizando el software [Sen2Cor](https://step.esa.int/main/third-party-plugins-2/sen2cor/).

# COMMAND ----------

# MAGIC %md
# MAGIC Las imágenes son también de dominio público y se pueden descargar de diferentes fuentes. En este caso vamos a descargar directamente de [Copernicus Open Access Hub](https://scihub.copernicus.eu/), utilizando un paquete llamado [sentinelsat](https://sentinelsat.readthedocs.io/en/stable/).

# COMMAND ----------

# Instalamos sentinelsat, y folium para visualizar los resultados
!pip install sentinelsat folium

# COMMAND ----------

# MAGIC %md
# MAGIC Es necesario primero [registrarse en Copernicus](https://scihub.copernicus.eu/dhus/#/self-registration), dado que necesitamos ingresar usuario y contraseña para consultar y descargar productos.

# COMMAND ----------

from getpass import getpass
import os

username = os.getenv('DHUS_USER')
password = os.getenv('DHUS_PASSWORD')

if not (username and password):
    username = input('DHUS username: ')
    password = getpass('DHUS password: ')

# COMMAND ----------

# MAGIC %md
# MAGIC Inicializamos el API para consultar y descargar productos de Sentinel.

# COMMAND ----------

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

# COMMAND ----------

api = SentinelAPI(username, password)

# COMMAND ----------

# MAGIC %md
# MAGIC Leemos un archivo GeoJSON ubicado en `data/sen2/area.geojson` y leemos el polígono. Se asume que el GeoJSON tiene un único feature, con una geometría asociada.

# COMMAND ----------

# MAGIC %md
# MAGIC **Captura del area de interés**
# MAGIC ![](img/sen2_area.jpg)

# COMMAND ----------

footprint = gdf_rectangulo.geometry.values[0]
footprint

# COMMAND ----------

# MAGIC %md
# MAGIC Hacemos una primer consulta. Buscamos imágenes de los primeros días de agosto 2021, por lo que queremos filtrar por el campo `date`.

# COMMAND ----------

products = api.query(footprint,
                     date=('2021601', '20210810'),
                     platformname='Sentinel-2',
                     cloudcoverpercentage=(0, 40),
                     producttype='S2MSI2A')

# COMMAND ----------

len(products)

# COMMAND ----------

# MAGIC %md
# MAGIC Vemos que esto arroja 7 resultados. Cada producto pueden ser diferentes *tiles* o *teselas* de una misma escena o captura. En este caso, son dos teselas de una misma escena, dado que ambos tienen misma fecha y hora de captura.

# COMMAND ----------

# MAGIC %md
# MAGIC Ya que estamos, antes de seguir, vamos a visualizar los footprints de cada escena, para entender como son las imagenes antes de bajarlas. Podriamos descargar el geojson y visualizarlo en QGIS, pero tambien podemos usar `folium` para tener un mapa en el notebook.

# COMMAND ----------

import folium
import json

products_geo = api.to_geojson(products)

m = folium.Map(min_zoom=7, max_zoom=18)

area_layer = folium.GeoJson(gdf_rectangulo, name='area', style_function=lambda x: {'color': '#ff0000', 'fillColor': '#ff0000'}).add_to(m)
sen2_layer = folium.GeoJson(products_geo, name='sen2').add_to(m)

# Tomar los bounds de la capa sen2 layer para centrar el mapa
m.fit_bounds(sen2_layer.get_bounds())

m

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora descargamos las dos teselas:

# COMMAND ----------

#!wget --content-disposition --continue --user <usuario> --password <password> "https://apihub.copernicus.eu/apihub/odata/v1/Products('f0c4605e-e9cd-4bae-a8f4-27d7bc1e9144')/\$value"

# COMMAND ----------

list_of_items = list(products.items()) # items from ordered dictionary 
product_ID = list_of_items[0][0] # item corresponding to product id 
product_info = api.get_product_odata(product_ID) # info for given product ID 
print(product_info['Online']) # True/false check for online

# COMMAND ----------

dirname = 'sesion_4/images/'
os.makedirs(dirname, exist_ok=True)
api.download_all(api.to_dataframe(products).index, directory_path=dirname)

# COMMAND ----------

!ls 'sesion_4/images/'

# COMMAND ----------

# MAGIC %md
# MAGIC Cada producto viene en un Zip, que hay que extraer:

# COMMAND ----------

!for f in  sesion_4/images/*.zip; do unzip $f -d  sesion_4/images/; done

# COMMAND ----------

!tree -d  sesion_4/images/*.SAFE

# COMMAND ----------

# MAGIC %md
# MAGIC Para cada escena, las imágenes suelen encontrarse dentro del directorio `GRANULE/L2A_*/IMG_DATA` donde `*` suele ser un id de la escena. Para el caso de los productos de nivel L2A, dentro de `IMG_DATA` hay 3 directorios:
# MAGIC 
# MAGIC * R10m: Imágenes de resolucion 10m
# MAGIC * R20m: Imágenes de resolución 20m
# MAGIC * R60m: Imágenes de resolución 60m

# COMMAND ----------

# MAGIC %md
# MAGIC Generalmente cada banda viene separada en un archivo distinto:

# COMMAND ----------

!tree sesion_4/images/S2A_MSIL2A_20210722T152641_N0301_R025_T18PWR_20210722T193332.SAFE/GRANULE/*/IMG_DATA

# COMMAND ----------

# MAGIC %md
# MAGIC Si queremos solamente trabajar con RGB, podemos utilizar directamente el producto TCI (True Color Image), que corresponde a una imagen RGB de 8-bit, con los valores escalados correctamente para visualización.
# MAGIC 
# MAGIC > The TCI is an RGB image built from the B02 (Blue), B03 (Green), and B04 (Red) Bands. The reflectances are coded between 1 and 255, 0 being reserved for 'No Data'. The saturation level of 255 digital counts correspond to a level of 3558 for L1C products or 2000 for L2A products (0.3558 and 0.2 in reflectance value respectively.
# MAGIC 
# MAGIC Es decir, es la combinación de las bandas 4-3-2 (RGB), reescalado a uint8 (8-bits de precisión) tomando como valor máximo el 2000.

# COMMAND ----------

from glob import glob
from skimage.exposure import rescale_intensity
import rasterio

# COMMAND ----------

glob("sesion_4/images/S2A_MSIL2A_20210722T152641_N0301_R025_T18PWR_20210722T193332.SAFE/GRANULE/*/IMG_DATA/R10m/*.jp2")

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt


# COMMAND ----------

tci_path = list(glob("sesion_4/images/S2A_MSIL2A_20210722T152641_N0301_R025_T18PWR_20210722T193332.SAFE/GRANULE/*/IMG_DATA/R10m/*_TCI_*.jp2"))[0]

with rasterio.open(tci_path) as src:
    img = np.dstack(src.read())
    #img = img[0:4000,6000:10980,:]
    plt.figure(figsize=(10,10))
    plt.imshow(img)

# COMMAND ----------

# MAGIC %md
# MAGIC Si deseamos trabajar también con NIR, nos conviene usar las bandas por separado. Deberíamos reescalarlas entre 0 y 2000, tal como se hace en el caso de TCI: 

# COMMAND ----------

base_path = "sesion_4/images/S2A_MSIL2A_20210722T152641_N0301_R025_T18PWR_20210722T193332.SAFE/GRANULE/*/IMG_DATA/R10m/"
r_path = glob(f"{base_path}/*_B04_*.jp2")[0]
g_path = glob(f"{base_path}/*_B03_*.jp2")[0]
b_path = glob(f"{base_path}/*_B02_*.jp2")[0]
nir_path = glob(f"{base_path}/*_B08_*.jp2")[0]

def read_image(path):
    with rasterio.open(path) as src:
        return src.read(1)

rgbi = np.dstack([read_image(p) for p in [r_path, g_path, b_path, nir_path]])

# COMMAND ----------

rgbi.shape, rgbi.dtype, rgbi.min(), rgbi.max(), rgbi.mean()

# COMMAND ----------

# Clip image
rgbi = rgbi[0:4000,6000:10980,:]

# COMMAND ----------

rgbi = rescale_intensity(rgbi, in_range=(0, 2000), out_range='uint8')

# COMMAND ----------

rgbi.shape, rgbi.dtype, rgbi.min(), rgbi.max(), rgbi.mean()

# COMMAND ----------

# RGB
img = rgbi[:,:,:3]
plt.figure(figsize=(10, 10))
plt.imshow(img)

# COMMAND ----------

# False color (NIR, R, G)
img = np.dstack([rgbi[:,:,3], rgbi[:,:,0], rgbi[:,:,1]])
plt.figure(figsize=(10, 10))
plt.imshow(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Combinaciones de Bandas**

# COMMAND ----------

!sudo apt-get install python3-gdal

# COMMAND ----------

!gdalinfo -stats '/content/drive/MyDrive/sesion_4/rectangulo_cali.tif'

# COMMAND ----------

!gdalwarp -cutline '/content/drive/MyDrive/sesion_4/rectangulo_cali_2.shp' \
-crop_to_cutline '/content/drive/MyDrive/sesion_4/rectangulo_cali.tif' \
'/content/drive/MyDrive/sesion_4/rectangulo_cali_clip.tif' 

# COMMAND ----------

!gdal_translate -b 3 -b 2 -b 1 '/content/drive/MyDrive/sesion_4/rectangulo_cali_clip.tif' \
'/content/drive/MyDrive/sesion_4/rectangulo_cali_clip_rgb.tif' 

# COMMAND ----------

!gdalinfo -stats '/content/drive/MyDrive/sesion_4/rectangulo_cali_clip_rgb.tif' 

# COMMAND ----------

!gdal_calc.py \
-A '/content/drive/MyDrive/sesion_4/rectangulo_cali.tif' \
--A_band=8 \
-B '/content/drive/MyDrive/sesion_4/rectangulo_cali.tif' \
--B_band=3 \
--outfile='/content/drive/MyDrive/sesion_4/rectangulo_cali_ndvi.tif' \
--calc="((A-B)/(A+B))"

# COMMAND ----------

!gdalinfo -stats '/content/drive/MyDrive/sesion_4/rectangulo_cali_ndvi.tif'

# COMMAND ----------

!gdal_merge.py -separate -o '/content/drive/MyDrive/sesion_4/rectangulo_cali_merge.tif' '/content/drive/MyDrive/sesion_4/rectangulo_cali.tif' '/content/drive/MyDrive/sesion_4/rectangulo_cali_ndvi.tif'

# COMMAND ----------

!gdalinfo -stats '/content/drive/MyDrive/sesion_4/rectangulo_cali_merge.tif'

# COMMAND ----------

!gdal_calc.py \
-A '/content/drive/MyDrive/sesion_4/rectangulo_cali_merge.tif' \
--A_band=1 \
-B '/content/drive/MyDrive/sesion_4/rectangulo_cali.tif' \
--B_band=4 \
--outfile='/content/drive/MyDrive/sesion_4/rectangulo_cali_merge_bin.tif' \
--calc="logical_and(A>0.3,B>0.3)"

# COMMAND ----------

!gdalinfo -stats '/content/drive/MyDrive/sesion_4/rectangulo_cali_merge_bin.tif'