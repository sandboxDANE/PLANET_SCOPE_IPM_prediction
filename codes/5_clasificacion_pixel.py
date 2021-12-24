# Databricks notebook source
!wget https://www.orfeo-toolbox.org/packages/OTB-7.4.0-Linux64.run

# COMMAND ----------

!apt install file

# COMMAND ----------

!chmod +x OTB-7.4.0-Linux64.run
!./OTB-7.4.0-Linux64.run
!source OTB-7.4.0-Linux64/otbenv.profile

# COMMAND ----------

from google.colab import drive
drive.mount('/content/drive')

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Procedimiento para entrenar**

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Extraemos el conteo de píxeles por clase

# COMMAND ----------

!mkdir /content/drive/MyDrive/sesion_8/results/

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_PolygonClassStatistics \
-in /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-vec /content/drive/MyDrive/sesion_8/ground_truth.shp \
-field  uso \
-out /content/drive/MyDrive/sesion_8/results/classes_stat.xml'

# COMMAND ----------

!cat /content/drive/MyDrive/sesion_8/results/classes_stat.xml

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Seleccionamos las observaciones que vamos a usar para entrenar el modelo

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_SampleSelection \
-in /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-vec /content/drive/MyDrive/sesion_8/ground_truth.shp \
-instats /content/drive/MyDrive/sesion_8/results/classes_stat.xml \
-field uso \
-strategy all \
-outrates /content/drive/MyDrive/sesion_8/results/rates.csv \
-out /content/drive/MyDrive/sesion_8/results/samples.sqlite'

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Extraemos los atributos a partir de las posiciones de las observaciones

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_SampleExtraction \
-in /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-vec /content/drive/MyDrive/sesion_8/results/samples.sqlite \
-outfield prefix \
-outfield.prefix.name band_ \
-field uso'

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Cálculo de estadísticas de píxel para normalizar

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_ComputeImagesStatistics \
-il  /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-out /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml'

# COMMAND ----------

!cat /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Entrenamos un árbol de decisión con profundidad 4

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_TrainVectorClassifier \
-io.vd /content/drive/MyDrive/sesion_8/results/samples.sqlite \
-io.stats /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml \
-cfield uso \
-classifier dt \
-classifier.dt.max 4 \
-io.out /content/drive/MyDrive/sesion_8/results/dTModel.txt \
-io.confmatout /content/drive/MyDrive/sesion_8/results/ConfusionMatrixDT.csv \
-feat band_0 band_1 band_2 band_3 band_4 band_5'

# COMMAND ----------

# MAGIC %md
# MAGIC Corriendo todo junto, con una sola línea de OTB

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_TrainImagesClassifier \
-io.il /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-io.vd /content/drive/MyDrive/sesion_8/ground_truth.shp \
-io.imstat /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml \
-sample.bm 0.1 \
-sample.vtr 0.80 \
-sample.vfn uso \
-classifier dt \
-classifier.dt.max 4 \
-io.out /content/drive/MyDrive/sesion_8/results/DTModel.txt \
-io.confmatout /content/drive/MyDrive/sesion_8/results/ConfusionMatrixDT.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Aplicamos el clasificador a la imagen entera

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_ImageClassifier \
-in /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-imstat /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml \
-model /content/drive/MyDrive/sesion_8/results/DTModel.txt \
-out /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_labeled_DT.tif'

# COMMAND ----------

# MAGIC %md
# MAGIC Nos interesa ver entonces cómo podemos ensamblar diferentes clasificadores.

# COMMAND ----------

# MAGIC %md
# MAGIC 8. Entrenamos y aplicamos un clasificador Random Forest

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_TrainImagesClassifier \
-io.il /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-io.vd /content/drive/MyDrive/sesion_8/ground_truth.shp \
-io.imstat /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml \
-sample.bm 0.1 \
-sample.vtr 0.80 \
-sample.vfn uso \
-classifier rf \
-classifier.rf.nbtrees 20 \
-classifier.rf.max 4 \
-io.out /content/drive/MyDrive/sesion_8/results/RFModel.txt \
-io.confmatout /content/drive/MyDrive/sesion_8/results/ConfusionMatrixRF.csv'

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_ImageClassifier \
-in /content/drive/MyDrive/sesion_7/0000013568-0000013568_rgb-nir-sw-diffSW_compress.tif \
-imstat /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml \
-model /content/drive/MyDrive/sesion_8/results/RFModel.txt \
-out /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_labeled_RF.tif'

# COMMAND ----------

# MAGIC %md
# MAGIC 9. Por último, fusionemos los clasificadores.

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_FusionOfClassifications \
-il  /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_labeled_DT.tif /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_labeled_RF.tif \
-method  majorityvoting \
-nodatalabel    99 \
-undecidedlabel 98 \
-out /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_ensemble.tif'

# COMMAND ----------

# MAGIC %md
# MAGIC Apliquemos un filtro para reducir el efecto salt and pepper

# COMMAND ----------

!gdal_sieve.py -st 10 /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_ensemble.tif /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_ensemble_filtered.tif

# COMMAND ----------

# MAGIC %md
# MAGIC 10. Comparemos las matrices de confusión del filtrado versus el no filtrado.

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_ComputeConfusionMatrix \
-in /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_ensemble.tif \
-out /content/drive/MyDrive/sesion_8/results/ConfusionMatrix_ensemble.csv \
-ref vector \
-ref.vector.in /content/drive/MyDrive/sesion_8/ground_truth.shp \
-ref.vector.field uso \
-ref.vector.nodata 255'

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_ComputeConfusionMatrix \
-in /content/drive/MyDrive/sesion_8/results/0000013568-0000013568_rgb-nir-sw-diffSW_compress_ensemble_filtered.tif \
-out /content/drive/MyDrive/sesion_8/results/ConfusionMatrix_ensemble_filtered.csv \
-ref vector \
-ref.vector.in /content/drive/MyDrive/sesion_8/ground_truth.shp \
-ref.vector.field uso \
-ref.vector.nodata 255'

# COMMAND ----------

# MAGIC %md
# MAGIC Por último, clasificamos en la misma imagen que usamos para la red convolucional así podemos comparar resultados.

# COMMAND ----------

!bash -c 'source OTB-7.4.0-Linux64/otbenv.profile; otbcli_ImageClassifier \
-in /content/drive/MyDrive/sesion_7/0000013568-0000027136_rgb-nir-sw-diffSW_compress.tif \
-imstat /content/drive/MyDrive/sesion_8/results/images_statistics_0000013568-0000013568_rgb-nir-sw-diffSW_compress.xml \
-model /content/drive/MyDrive/sesion_8/results/RFModel.txt \
-out /content/drive/MyDrive/sesion_8/results/0000013568-0000027136_rgb-nir-sw-diffSW_compress_labeled_RF.tif'