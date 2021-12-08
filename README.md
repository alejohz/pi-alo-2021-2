# pi-alo-2021-2
 Repo donde se encuentra toda la documentación alrededor del proyecto integrador para la Maestría en Ciencia de Datos  y Analítica (2021-2)


Esta es la documentación del proyecto integrador.
Los integrantes somos:
Alejandro Barrientos Osorio
Luis Miguel Caicedo Jimenez
Omar Alejandro Henao Zapata


Antes de hablar de los archivos necesarios, requerimos actualizar las llaves en tres scripts principales:

master.sh
Ingest_BI.py
Upload_s3.ipynb

Las llaves son aws_access_key_id, aws_secret_access_key y aws_session_token.

Los archivos requeridos son:

master.sh - Este es el script de bash maestro que corre el script EMR_Cluster.py con las llaves actualizadas.

EMR_Cluster.py - Este es el script que a través de boto3 corre un Cluster de EMR con 2 nodos y 1 nodo maestro con JupyterHub como aplicación.

El resto de archivos deberían estar montados en S3, incluidos los scripts en la zona de "s3://bucket-name/scripts/".

Sin embargo, nosotros los modificamos de manera local y por medio del notebook "Upload_s3.ipynb" actualizamos las versiones en el bucket de s3. Por esto es el único notebook dentro del trabajo, para poder escoger cual script actualizar  y optimizar las conexiones a S3.