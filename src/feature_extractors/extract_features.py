import os, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', action='store', required=False, default='/home/additya/NUS/datasets/new/Obama_new/new/')
parser.add_argument('--dest_dir', action='store', required=False, default='/home/additya/NUS/features/new/Obama_new/new/')
parser.add_argument('--openface_dir', action='store', required=False, default='/home/additya/NUS/src/feature_extractors/')
args = parser.parse_args()

source_dir = args.source_dir
dest_dir = args.dest_dir

if os.path.exists(dest_dir):
	shutil.rmtree(dest_dir)

for root, dirs, files in os.walk(source_dir):
	for file in files:
		if file.endswith(('.mp4', '.avi')):
			print("=" * 100)
			print("File: ", os.path.join(root, file))
			print("=" * 100)
			out_path = os.path.join(dest_dir, os.path.join(root, file)[len(source_dir):]).replace('.mp4', '').replace('.avi', '')
			command = args.openface_dir + 'OpenFace/build/bin/FeatureExtraction -q' \
					+ ' -f "' + os.path.join(root, file) + '" -dest_dir "' + out_path + '"'
			if not os.path.exists(out_path):
				os.makedirs(out_path)
			os.system(command)
			print("==================Done: ", file, "==================")
