wget http://pai-vision-data.oss-cn-shanghai.aliyuncs.com/models/resnet_v1_50.tar.gz
tar -zxf resnet_v1_50.tar.gz

python image_feature_extract.py  resnet_v1_50  image_path
