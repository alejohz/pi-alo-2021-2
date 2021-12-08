#!/bin/bash


aws_access_key_id='ASIAWN43HLCG4N5AT3OC'
aws_secret_access_key='LJVnoz4hYZMJkzpGHNOmM5HhYyNSIknxhxL35i3y'
aws_session_token='FwoGZXIvYXdzEND//////////wEaDLQi81goWs+SOKQpuCKuAacIqlKGQbpFKEJv1k14zCRTMHDrVnvLFSJUferhh3zlxSH7kDvNMWvgHOW2p3EkNN+Pt3rgUCRZht5niYfRiJif92UOEFqOq2vw/eW2kDFiUij/ZoFJDOlPC4NI9g2Th7LOXmlPYQr6rCsAuqigpbo670dPj2jwwubiO35cJFfzLHQffkpU6pW+hl57jhMiS+WkA0enRPGv0ODuMgF12MrGIcOe91UvtalSwEO7OCjgpbqNBjIthoK31hHe3Sp6Vnjx0Ozxo+U0k0laLoYtVTBi7J9x1iyNagBI/mpKlMJswt7D'


#python3 /home/ahenao/scripts/PI/EMR_Cluster.py --key1 $aws_access_key_id --key2 $aws_secret_access_key --key3 $aws_session_token
python3 /home/ahenao/scripts/PI/Clustering.py --key1 $aws_access_key_id --key2 $aws_secret_access_key --key3 $aws_session_token