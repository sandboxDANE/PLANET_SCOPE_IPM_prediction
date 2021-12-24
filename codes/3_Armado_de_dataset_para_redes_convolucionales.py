# Databricks notebook source
!sudo apt-get install python3-gdal
!sudo apt-get install tree
!pip install rasterio
!pip install  keras

# COMMAND ----------

## Estos paquetes deberían funcionar INGE
#from skimage.io import imsave
import rasterio
import os
import logging
from rasterio.windows import Window
from shapely.geometry import box, shape
from tqdm import tqdm
import numpy as np
import cv2
import imageio
from tensorflow.keras.preprocessing.image import save_img

# COMMAND ----------

from google.colab import drive
drive.mount('/content/drive')

# COMMAND ----------

!ls /content/drive/MyDrive/sesion_6/

# COMMAND ----------

!gdalinfo -stats '/content/drive/MyDrive/sesion_4/rectangulo_cali.tif'

# COMMAND ----------

#working_directory = "/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/"

!gdalinfo -stats '/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/2016/20160820_133829_0c82/20160820_133829_0c82_3B_AnalyticMS.tif'


# COMMAND ----------



# COMMAND ----------

def sliding_windows(size, step_size, width, height, whole=False):
    """Slide a window of +size+ by moving it +step_size+ pixels
    Parameters
    ----------
    size : int
        window size, in pixels
    step_size : int
        step or *stride* size when sliding window, in pixels
    width : int
        image width
    height : int
        image height
    whole : bool (default: False)
        whether to generate only whole chips, or clip them at borders if needed
    Yields
    ------
    Tuple[Window, Tuple[int, int]]
        a pair of Window and a pair of position (i, j)
    """
    w, h = size
    sw, sh = step_size
    end_i = height - h if whole else height
    end_j = width - w if whole else width
    for pos_i, i in enumerate(range(0, end_i, sh)):
        for pos_j, j in enumerate(range(0, end_j, sw)):
            real_w = w if whole else min(w, abs(width - j))
            real_h = h if whole else min(h, abs(height - i))
            yield Window(j, i, real_w, real_h), (pos_i, pos_j)


# COMMAND ----------

#from keras.preprocessing.image import save_img
#import keras
#import tensorflow as tf


# COMMAND ----------

from PIL import Image

# COMMAND ----------

raster = '/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/2016/20161216_143523_0e2f/20161216_143523_0e2f_3B_AnalyticMS.tif'
bands = range(8)
#raster = '/content/drive/MyDrive/sesion_4/rectangulo_cali_clip_rgb.tif'
step_size = 100
size = 1000
output_dir = '/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/chips'

_logger = logging.getLogger(__name__)


with rasterio.open(raster) as ds:
        basename, _ = os.path.splitext(os.path.basename(raster))
        image_folder = os.path.join(output_dir, "chips_1000")
        os.makedirs(image_folder, exist_ok=True)
        _logger.info("Raster size: %s", (ds.width, ds.height))

        _logger.info("Building windows")
        win_size = (size, size)
        win_step_size = (step_size, step_size)
        windows = list(
            sliding_windows(win_size, win_step_size, ds.width, ds.height, whole=True)
        )
        _logger.info("Total windows: %d", len(windows))

        _logger.info("Building window shapes")
        window_shapes = [
            box(*rasterio.windows.bounds(w, ds.transform)) for w, _ in windows
        ]
        window_and_shapes = zip(windows, window_shapes)

        chips = []
        for (window, (i, j)) in tqdm(windows):
            _logger.debug("%s %s", window, (i, j))

            img_path = os.path.join(image_folder, f"{basename}_{i}_{j}.jpg")
            img = ds.read(window=window)
            img = np.nan_to_num(img)
            rgb = cv2.normalize(img[:, :, :], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            rgb = rgb.astype(np.uint8)
            rgb = np.moveaxis(rgb, [0, 1, 2], [2, 1, 0])
            #im = Image.new("RGBA",(100,100))
            rgb = Image.fromarray(rgb)
            rgb = rgb.convert("RGB")
            save_img(img_path, rgb)
            



# COMMAND ----------

!ls /dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/chips/chips/*
#!ls /content/drive/MyDrive/sesion_6/chips/chips/*

# COMMAND ----------


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/chips/chips_1000/20161216_143523_0e2f_3B_AnalyticMS_23_40.jpg')
imgplot = plt.imshow(img)
plt.show()

# COMMAND ----------

img = mpimg.imread('/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/2016/20161216_143523_0e2f/20161216_143523_0e2f_3B_AnalyticMS.tif')
imgplot = plt.imshow(img)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Apliquemos para un ráster entero

# COMMAND ----------

!unzip /content/drive/MyDrive/sesion_6/images/S2B_MSIL2A_20210806T152639_N0301_R025_T18PWR_20210806T203538.zip -d /content/drive/MyDrive/sesion_6/images/


# COMMAND ----------

!tree /content/drive/MyDrive/sesion_6/images/S2B_MSIL2A_20210806T152639_N0301_R025_T18PWR_20210806T203538.SAFE/GRANULE/*/IMG_DATA

# COMMAND ----------

!gdal_translate \
-of GTiff \
-co COMPRESS=JPEG \
/content/drive/MyDrive/sesion_6/images/S2B_MSIL2A_20210806T152639_N0301_R025_T18PWR_20210806T203538.SAFE/GRANULE/L2A_T18PWR_A023074_20210806T152640/IMG_DATA/R10m/T18PWR_20210806T152639_TCI_10m.jp2 \
/content/drive/MyDrive/sesion_6/images/S2B_MSIL2A_20210806T152639_N0301_R025_T18PWR_20210806T203538.SAFE/GRANULE/L2A_T18PWR_A023074_20210806T152640/IMG_DATA/R10m/T18PWR_20210806T152639_TCI_10m.tif

# COMMAND ----------

!gdalinfo -stats /content/drive/MyDrive/sesion_6/images/S2B_MSIL2A_20210806T152639_N0301_R025_T18PWR_20210806T203538.SAFE/GRANULE/L2A_T18PWR_A023074_20210806T152640/IMG_DATA/R10m/T18PWR_20210806T152639_TCI_10m.tif

# COMMAND ----------


bands = range(8)
raster = '/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/2016/20161216_143523_0e2f/20161216_143523_0e2f_3B_AnalyticMS.tif'
step_size = 1000
size = 100
output_dir = '/dbfs/mnt/pobreza/changeanalysis/func-mintic-pobrezamultidimensio/IMAGENES/PLANET SCOPE/chips'

_logger = logging.getLogger(__name__)


with rasterio.open(raster) as ds:
        basename, _ = os.path.splitext(os.path.basename(raster))
        image_folder = os.path.join(output_dir, "chips_v2_1000")
        os.makedirs(image_folder, exist_ok=True)
        _logger.info("Raster size: %s", (ds.width, ds.height))

        _logger.info("Building windows")
        win_size = (size, size)
        win_step_size = (step_size, step_size)
        windows = list(
            sliding_windows(win_size, win_step_size, ds.width, ds.height, whole=True)
        )
        _logger.info("Total windows: %d", len(windows))

        _logger.info("Building window shapes")
        window_shapes = [
            box(*rasterio.windows.bounds(w, ds.transform)) for w, _ in windows
        ]
        window_and_shapes = zip(windows, window_shapes)

        chips = []
        for (window, (i, j)) in tqdm(windows):
            _logger.debug("%s %s", window, (i, j))

            img_path = os.path.join(image_folder, f"{basename}_{i}_{j}.jpg")
            img = ds.read(window=window)
            img = np.nan_to_num(img)
            rgb = cv2.normalize(img[:, :, :], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            rgb = rgb.astype(np.uint8)
            rgb = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
            rgb = Image.fromarray(rgb)
            rgb = rgb.convert("RGB")
            save_img(img_path, rgb)
          

# COMMAND ----------

!ls '/content/drive/MyDrive/sesion_6/chips_S2B_MSIL2A_20210806T152639/chips'

# COMMAND ----------

# MAGIC 
# MAGIC %pylab inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC import matplotlib.image as mpimg
# MAGIC img = mpimg.imread('/content/drive/MyDrive/sesion_6/chips_S2B_MSIL2A_20210806T152639/chips/T18PWR_20210806T152639_TCI_10m_9_30.jpg')
# MAGIC imgplot = plt.imshow(img)
# MAGIC plt.show()