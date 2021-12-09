El proceso de análisis de la información se puede dividir en cuatro grandes pasos que se dividen en cuatro archivos en este repo:

Procedimiento 1: Extracción de información, ingeniería de características y tratamiento de los datos.
Procedimiento 2: Tratamiento más riguroso de de los datos (Limpieza, Corte). Transformación de la información para el analísis transversal, el analísis en si, generación de resultados y subida de los resultados a s3.
Procedimiento 3: Tratamiento más riguroso de los datos (Limpieza, Corte) (Difiere en el procedimiento anterior en que este paso juega con la información desde el mismo punto de partida, pero hasta el último dato disponible, el paso anterior solo usa hasta el último año terminado) analisis de series de tiempo, tratamiento, transformación y resultados. Luego, subida de resultados a s3.
Procedimiento 4: Visualización y presentación de los resultados en Power BI.

Como el nombre del archivo sugiere, este pretende explicar el procedimiento que se debe seguir para reproducir los resultados:


Para este paso debe cerciorarse que tiene acceso al bucket en S3 llamado 'pi-alo-2021-2' o cambiar todas las variables "bucket" por el bucket a usar. El bucket debe tener otras tres zonas "raw", "trusted", "refined" y "logs" que esta ultima es usada para almacenar los registros de corrida de los clusters.

Dentro del bucket, la carpeta scripts debe contener el archivo "Clustering.py"

La fundamentación técnica y conclusiones se encuentran en el trabajo escrito, acá se pretende explicar solo los pasos requeridos para poder reproducir los resultados.

Clustering.py:

Este script trabaja con la zona trusted del datalake, es decir, la información entregada por Procedimiento1. 

En resumen, el script corta la información entre el 2000 y 2020 (Ultimo año terminado) escala los datos y hace una agrupación no supervisada por medio de dos métodos: KMeans  y Spectral Clustering, para ambos métodos se evalua su efectividad por medio de un indicador llamado Silhouette Index o indicador de silueta. 

Todos los archivos generados por este script son depositados en la zona refined del cluster. Refined tiene dos subzonas, Clustering y Time_Series, esta va hacia Clustering.