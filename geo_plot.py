# Plot geographical location of videos on map of the area

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json
import seaborn

host = 'search-vast-db-h3mq23otgi6hffagiqjpjsnsey.us-east-1.es.amazonaws.com'
region = 'us-east-1' # e.g. us-west-1

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

# query es for metadata results
all_files = json.dumps(es.search(index="vid_metadata", doc_type="_doc", q="*", filter_path=['hits.hits._source.location.lat', 'hits.hits._source.location.lon', 'hits.hits._source.date'], size=1000))
contents = json.loads(all_files)

lat = []
lon = []
times = []

#iterate through es query return
for hit in contents['hits']['hits']:
    try:
        times.append(hit['_source']['date'])
        lat.append(hit['_source']['location']['lat'])
        lon.append(hit['_source']['location']['lon'])
    except KeyError as e:
        print("No Source")

df = pd.DataFrame(columns=['lat', 'lon', 'date'])
df['date'] = times
df['lat'] = lat
df['lon'] = lon

#print(df.head())

# define bounding box containing all points
    # cannot use df min and max due to 0.0 invalid points
bbox = (-77.0575, -76.9991, 38.8715, 38.9122)

# get mag image from openstreetmap.org using set bbox boundaries
plot = plt.imread('./map.png')

fig, ax = plt.subplots(figsize = (8,7))
# can adjust color based on map composition
ax.scatter(df.lon, df.lat, zorder=1, alpha= 0.2, c='#01058B', s=30)
ax.set_title('Plotting Spatial Data on Vicinity Map')
ax.set_xlim(bbox[0],bbox[1])
ax.set_ylim(bbox[2],bbox[3])
ax.imshow(plot, zorder=0, extent = bbox, aspect= 'equal')
plt.show()

# future embellishment - account for time data (stored in "times" list) and
# color points with a gradient to indicate time