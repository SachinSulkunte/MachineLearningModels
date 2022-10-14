# Requires "Image-ExifTool-12.28 2" folder in working directory - downloaded with exiftool

from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json
import subprocess
from boto3 import client
import botocore
from botocore.exceptions import ClientError

host = $host
region = $region

service = 'es'
credentials = boto3.Session(profile_name="lambda").get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

es = Elasticsearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

# remove special characters from coordinates
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

temp = {} # single video tags
coords = {}
exifToolPath = "exiftool" # downloaded from exiftool.org

bucket = $bucket # bucket name
def detect_loc(prefix):
    try:
        s3 = client('s3')
        for key in s3.list_objects(Bucket=bucket, Prefix=prefix)['Contents']:
            print (key['Key'])
            path = "./temp.mp4"
            try:
                s3.download_file(bucket, key['Key'], path)
                data = subprocess.Popen([exifToolPath, path],stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True) 
                for tag in data.stdout:
                    line = tag.strip().split(':', 1)
                    temp[line[0].strip()] = line[-1].strip()
                try:
                    gps = temp['GPS Position']
                    dic = {",":"", "'":"", "\"":""} # extract special chars
                    gps = replace_all(gps, dic)
                    gps = gps.split()
                    deg = gps[0]
                    minutes = gps[2]
                    seconds = gps[3]
                    direction = gps[4]
                    # convert to decimal format
                    lat = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)
                    
                    deg = gps[5]
                    minutes = gps[7]
                    seconds = gps[8]
                    direction = gps[9]
                    # convert to decimal format
                    lon = (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)
                except KeyError as e:
                    lat = 0
                    lon = 0
                
                try:
                    date = temp['Create Date']
                    if (date == '0000:00:00 00:00:00'): # invalid in elasticsearch date field
                        date = '1111:11:11 11:00:00' # filler valid date
                except KeyError as e:
                    date = '1111:11:11 11:00:00' # filler valid date
                
                try: # not currently used
                    cam = temp['Make'] + ' ' + temp['Model']
                except KeyError as e:
                    cam = "Unknown"
                
                # can add more fields here if desired

                document = {
                    "filename": key['Key'],
                    "date": date,
                    "location": {
                        "lat": lat,
                        "lon": lon
                    },
                    "camera": cam
                }
                #print(document)
                # write data to elasticsearch index
                response = es.index(index="vid_metadata", doc_type="_doc", body=document)
                print(response)
            except botocore.exceptions.ClientError as error:
                print(error)
                s3 = client('s3')
    except botocore.exceptions.ClientError as error:
        print("ClientError")
        print(error)

# S3 bucket folders to run code on
prefix = ["vids1", "vids2", "vids3", "vids4"]
for i in prefix:
    detect_loc(i)
