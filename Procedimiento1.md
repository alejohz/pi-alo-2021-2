El proceso de análisis de la información se puede dividir en cuatro grandes pasos que se dividen en cuatro archivos en este repo:

Procedimiento 1: Extracción de información, ingeniería de características y tratamiento de los datos.
Procedimiento 2: Tratamiento más riguroso de de los datos (Limpieza, Corte). Transformación de la información para el analísis transversal, el analísis en si, generación de resultados y subida de los resultados a s3.
Procedimiento 3: Tratamiento más riguroso de los datos (Limpieza, Corte) (Difiere en el procedimiento anterior en que este paso juega con la información desde el mismo punto de partida, pero hasta el último dato disponible, el paso anterior solo usa hasta el último año terminado) analisis de series de tiempo, tratamiento, transformación y resultados. Luego, subida de resultados a s3.
Procedimiento 4: Visualización y presentación de los resultados en Power BI.

Como el nombre del archivo sugiere, este pretende explicar el procedimiento que se debe seguir para reproducir los resultados:

Para este paso debe cerciorarse que tiene acceso al bucket en S3 llamado 'pi-alo-2021-2' o cambiar todas las variables "bucket" por el bucket a usar. El bucket debe tener otras tres zonas "raw", "trusted", "refined" y "logs" que esta ultima es usada para almacenar los registros de corrida de los clusters.

Dentro del bucket, la carpeta scripts debe contener los archivos "extract_files.py",  "raw_to_trusted.py" y "install_libraries.sh"

install_libraries.sh: 

Fundamental en el bootstrapping del cluster en EMR. Es el que actualiza e instala las librerias requeridas para el proyecto en el ambiente de procesamiento. 

extract_files.py:

Utiliza un método simple de extracción de la información desde yahoo finance por medio de una URL que según algunos parámetros se puede extraer la información requerida, de cualquier índice y en cualquier rango de tiempo disponible en Yahoo Finance. El archivo ya se entrega en formato csv y este es el archivo IndexData

También utiliza la interfase desarrollada por el world bank data para encontrar y extraer información disponible en el sitio oficial "https://data.worldbank.org/" de aquí extraemos información como porcentajes de cambio de GDP, población y cambios en la inflación por región o pais, la transformamos y filtramos a solo lo que necesitamos y se sube a S3.

Todos los archivos de este script son depositados en la zona raw del datalake.

raw_to_trusted.py:
Este archivo usa los 4 archivos producidos por el script anterior, a partir de IndexData crea dos archivos, que se llaman IndexProcessed que es los indices convertidos a USD e IndexData_2 que es los indices en el rango de tiempo de estudio, es decir, desde el primer dia disponible del 2000 hasta el ultimo dato disponible (Se extrae tambien 1999 para el analisis transversal, pero se corta en el tratamiento más especifico)

El formato de los archivos es index_data_2 más el día que se hace el query para terminar como .csv, lo mismo ocurre con el resto de archivos para diferenciar que dia fueron extraídos.

Todos los archivos de este script son depositados en la zona trusted del datalake.

Para continuar, leer Procedimiento2.