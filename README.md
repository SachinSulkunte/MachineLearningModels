# MachineLearning_AWS
All code operates on input data that is pulled from an AWS S3 bucket and once processed, placed into an Elasticsearch database. The processing was automated using AWS Lambda.

**extract_text.py**: Custom OCR implementation using the open-source Tesseract Engine and OpenCV

**metadata.py**: Extracts available metadata from video files for use in ML models
- **geoplot.py**: Plots location extracted from video metadata across map of the vicinity to determine spread
