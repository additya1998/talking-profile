# Talking Profile for Fake Video Detection

## Repository Structure

	- src/
  	  - feature_extractors/
	    - svm/
	- datasets/
	- features/

## Datasets

1. [**FaceForensics++**](https://github.com/ondyari/FaceForensics)
```shell
python download-FaceForensicspp.py
    <output path>
    -d <dataset type, e.g., Face2Face, original or all>
    -c <compression quality, e.g., c23 or raw>
    -t <file type, e.g., images, videos, masks or models> 
```		

2. **Barack Obama**
```shell
cd talking-profile/datasets/Obama
python download.py
``` 

3. **Donald Trump**
```shell
cd talking-profile/datasets/Trump
python download.py
``` 

## Feature Extraction
1. Build [Openface](https://github.com/TadasBaltrusaitis/OpenFace) at talking-profile/src/feature_extractors
2. Extraction
```shell
python talking-profile/feature_extractors/extract_features.py --source_dir <source_video_director> --dest_dir <output_directory>
```

## Classification

1. **How to run?**
```shell
python talking-profile/src/svm/code.py
```
2. **Functions**
	* **au_features** - takes the file path and of a video and returns the AU features extracted by Openface for each frame of the video 	
	* **face_features** - takes the file path and of a video and returns the following features extracted by Openface for each frame of the video
		* Location of the head with respect to camera in millimeters.  
		* Rotation is in radians around X,Y,Z axes.
		* Normalized Eye gaze features
		* Normalized Blink features
		* Normalized Lip gap features
	* **generate_tp** - generates the talking profile for a given array of features
		* tp_type : 'original' uses difference between feature vectors of frames to calculate talking profile while 'corr' uses correlation between features as described [here](http://openaccess.thecvf.com/content_CVPRW_2019/html/Media_Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.html)
		* frame_length : number of frames used to generate one talking profile feature vector
		* stride : used to control overlap between different talking profile feature vectors
		* fgap : used to control the number of frames to skip between two selected frames for one talking profile feature vector
		* feature_subset : use only a subset of features   
	* **get_classifier** - returns a classifier (simple SVM or AdaBoost) for the input training data