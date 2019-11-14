from sklearn.preprocessing import normalize
import csv, os, sys
import numpy as np
from sklearn.utils import shuffle
from random import randint
import pickle
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from config import *

class Experiment:
	def __init__(self, video_idx, fake_type, feature_type, tp_type, svm_type, frame_length, training_samples):
	
		self.video_idx = video_idx
		self.feature_path = os.path.join(FEATURE_PREFIX_PATH, fake_type, 'c23/videos', '0' * (3 - len(str(self.video_idx))) + str(self.video_idx), '0' * (3 - len(str(self.video_idx))) + str(self.video_idx) + '.csv')
		self.feature_type = feature_type
		self.svm_type = svm_type
		self.training_samples = training_samples
		self.frame_length = frame_length
		self.tp_type = tp_type
		self.fake_type = fake_type

		(self.fake_s, self.fake_e) = frame_d[self.fake_type][self.video_idx]
		
		self.features = self.get_features(self.feature_path)

		(frames_left, frames_right) = (self.frame_length // 2, (self.frame_length - 1) // 2)
		(train_first_center_frame, train_last_center_frame) = (self.frame_length // 2, (self.frame_length // 2) + self.training_samples - 1)

		self.train_positive_idx = np.array([i for i in range(train_first_center_frame, train_last_center_frame + 1)])
		self.train_positive = np.array([])
		for idx in self.train_positive_idx:
			(low, high) = (idx - frames_left, idx + frames_right + 1) 	# [low, high)
			assert(high - low == self.frame_length)
			tp = self.generate_tp(self.features[low:high, :])
			# assert(tp.shape[0] == 1)
			if self.train_positive.shape[0] == 0:
				self.train_positive = tp
			else:
				self.train_positive = np.vstack((self.train_positive, tp))
		assert(self.train_positive.shape[0] == self.training_samples)
		assert(self.fake_s > train_last_center_frame + frames_right)
		assert(self.train_positive.shape[0])
		print("Initial frames accessed for training:", train_last_center_frame + frames_right + 1)

		self.train_last_center_frame = train_last_center_frame
		self.test_positive_idx = np.array([i for i in range(train_last_center_frame + 1, self.fake_s)])
		self.test_positive_idx = np.hstack((self.test_positive_idx, np.array([i for i in range(self.fake_e + 1, self.features.shape[0] - 1 - frames_right)])))
		self.test_positive = np.array([])
		for idx in self.test_positive_idx:
			(low, high) = (idx - frames_left, idx + frames_right + 1) 	# [low, high)
			assert(high - low == self.frame_length)
			tp = self.generate_tp(self.features[low:high, :])
			# assert(tp.shape[0] == 1)
			if self.test_positive.shape[0] == 0:
				self.test_positive = tp
			else:
				self.test_positive = np.vstack((self.test_positive, tp))
		assert(self.test_positive.shape[0])

		self.test_negative_idx = np.array([i for i in range(self.fake_s, self.fake_e + 1)])
		self.test_negative = np.array([])
		for idx in self.test_negative_idx:
			(low, high) = (idx - frames_left, idx + frames_right + 1) 	# [low, high)
			assert(high - low == self.frame_length)
			tp = self.generate_tp(self.features[low:high, :])
			# assert(tp.shape[0] == 1)
			if self.test_negative.shape[0] == 0:
				self.test_negative = tp
			else:
				self.test_negative = np.vstack((self.test_negative, tp))
		assert(self.test_negative.shape[0])

		self.train_negative = self.generate_train_negative()
		if self.svm_type == 'two-class':
			assert(self.train_negative.shape[0])

		self.added_idx = []

		self.original_train_positive = self.train_positive
		self.original_train_negative = self.train_negative
		self.original_test_positive = self.test_positive
		self.original_test_negative = self.test_negative

	def generate_train_negative(self, video_count=100):
		if self.svm_type == 'one-class':
			return np.array([])

		train_negative = np.array([])
		idx_list = [i for i in range(1000) if i not in ALL_VIDEOS_IDX]
		np.random.shuffle(idx_list); 
		idx_list = idx_list[:video_count]

		for f in sorted(idx_list):
			f = '0' * (3 - len(str(f))) + str(f)
			feat_path = os.path.join('/home/additya/NUS/features/NewFaceForensics++/FaceForensics++/original_sequences/youtube/c23/videos/', f, f + '.csv')
			feat = self.get_features(feat_path)
			full_tp = self.generate_tp(feat)
			full_tp_less_overlap = full_tp
			for i in range(10, full_tp.shape[0], 10):
				full_tp_less_overlap = np.vstack((full_tp_less_overlap, full_tp[i]))
			full_tp_less_overlap = full_tp_less_overlap[np.random.permutation(full_tp_less_overlap.shape[0])[:20], :] 
			if full_tp_less_overlap.shape[0] == 0:
				continue
			if train_negative.shape[0] == 0:
				train_negative = full_tp_less_overlap
			else:
				train_negative = np.vstack((train_negative, full_tp_less_overlap))
		
		return train_negative

	def get_features(self, path):
		
		if self.feature_type == 'au':
			features = np.array([])
			with open(path, "r") as face_csv:
				csv_reader = csv.DictReader(face_csv)
				for row in csv_reader:
					feature_v = []
					feature_v += [float(row[' AU01_r'])]
					feature_v += [float(row[' AU02_r'])]
					feature_v += [float(row[' AU04_r'])]
					feature_v += [float(row[' AU05_r'])]
					feature_v += [float(row[' AU06_r'])]
					feature_v += [float(row[' AU07_r'])]
					feature_v += [float(row[' AU09_r'])]
					feature_v += [float(row[' AU10_r'])]
					feature_v += [float(row[' AU12_r'])]
					feature_v += [float(row[' AU14_r'])]
					feature_v += [float(row[' AU15_r'])]
					feature_v += [float(row[' AU17_r'])]
					feature_v += [float(row[' AU20_r'])]
					feature_v += [float(row[' AU23_r'])]
					feature_v += [float(row[' AU25_r'])]
					feature_v += [float(row[' AU26_r'])]
					
					# feature_v += [float(row[' AU45_r'])] -> Not using eye-blink AU
					
					feature_v += [float(row[' pose_Rx'])] # head rotation about the x-axis 
					feature_v += [float(row[' pose_Rz'])] # head rotation about the z-axis

					upper_lip = np.array([float(row[' X_62']), float(row[' Y_62']), float(row[' Z_62'])])
					lower_lip = np.array([float(row[' X_66']), float(row[' Y_66']), float(row[' Z_66'])])
					lip_gap_h = np.linalg.norm(upper_lip - lower_lip)

					lip_left = np.array([float(row[' X_60']), float(row[' Y_60']), float(row[' Z_60'])])
					lip_right = np.array([float(row[' X_64']), float(row[' Y_64']), float(row[' Z_64'])])
					lip_gap_w = np.linalg.norm(lip_left - lip_right)

					feature_v += [lip_gap_h] # 3-D horizontal distance between the corners of the mouth
					feature_v += [lip_gap_w] # 3-D vertical distance between the lower and upper lip
					
					feature_v = np.array(feature_v)
					if features.shape[0] == 0:
						features = feature_v
					else:
						features = np.vstack((features, feature_v))
		
		elif self.feature_type == 'original':
			features = np.array([])
			with open(path, "r") as face_csv:
				csv_reader = csv.DictReader(face_csv)
				for row in csv_reader:
					ht = [float(row[' pose_Tx']), float(row[' pose_Ty']), float(row[' pose_Tz'])]
					hr = [float(row[' pose_Rx']), float(row[' pose_Ry']), float(row[' pose_Rz'])]
					gaze = [float(row[' gaze_0_x']), float(row[' gaze_0_y']), float(row[' gaze_0_z']),
							float(row[' gaze_1_x']), float(row[' gaze_1_y']), float(row[' gaze_1_z'])]
					gaze_angle = [float(row[' gaze_angle_x']), float(row[' gaze_angle_y'])]
					blink_left = np.linalg.norm(np.array([float(row[' eye_lmk_x_17']), float(row[' eye_lmk_y_17'])])
							  - np.array([float(row[' eye_lmk_x_11']), float(row[' eye_lmk_y_11'])]))
					blink_right = np.linalg.norm(np.array([float(row[' eye_lmk_x_45']), float(row[' eye_lmk_y_45'])])
								- np.array([float(row[' eye_lmk_x_39']), float(row[' eye_lmk_y_39'])]))
					lip_gap_h = np.linalg.norm(np.array([float(row[' x_66']), float(row[' y_66'])])
						- np.array([float(row[' x_62']), float(row[' y_62'])]))
					lip_gap_w = np.linalg.norm(np.array([float(row[' x_64']), float(row[' y_64'])])
								- np.array([float(row[' x_60']), float(row[' y_60'])]))
					face_length = np.linalg.norm(np.array([float(row[' x_27']), float(row[' y_27'])])
						- np.array([float(row[' x_8']), float(row[' y_8'])])) + np.finfo(float).eps
					lip_gap_h = lip_gap_h / face_length # Normalize
					lip_gap_w = lip_gap_w / face_length # Normalize
					blink_left = blink_left / face_length # Normalize
					blink_right = blink_right / face_length # Normalize
					# feature_v = hr + [blink_left, blink_right] + [lip_gap_w, lip_gap_h]
					feature_v = ht + hr + [blink_left, blink_right] + [lip_gap_w, lip_gap_h]
					feature_v = feature_v + gaze
					# feature_v = feature_v + gaze_angle -> Gaze angle instead of gaze vector
					feature_v = np.array(feature_v)
					if features.shape[0] == 0:
						features = feature_v
					else:
						features = np.vstack((features, feature_v))

		return features

	def generate_tp(self, features):
	
		if features.shape[0] < self.frame_length:
			return np.array([])

		tp = np.array([])
		
		if self.tp_type == 'original':    
			for X in range(0, features.shape[0]):
				temp_f = np.array([])
				for Y in range(X, features.shape[0]):
					if (Y + 1 >= features.shape[0]) or (temp_f.shape[0] == self.frame_length - 1):
						break
					else:
						if temp_f.shape[0] == 0:
							temp_f = features[Y + 1, :] - features[Y, :]
						else:
							temp_f = np.vstack((temp_f, features[Y + 1, :] - features[Y, :]))
				if temp_f.shape[0] < self.frame_length - 1:
					break
				else:                
					if tp.shape[0] == 0:
						tp = temp_f.flatten()
					else:
						tp = np.vstack((tp, temp_f.flatten()))
	 
		else:
			for X in range(0, features.shape[0]):
				temp_f = features[X, :]
				for Y in range(X + 1, features.shape[0]):
					if temp_f.shape[0] == self.frame_length:
						break
					else:
						temp_f = np.vstack((temp_f, features[Y, :]))
				if temp_f.shape[0] == self.frame_length:
					corr = []
					for i in range(temp_f.shape[1]):
						for j in range(i, temp_f.shape[1]):
							corr.append(np.dot(temp_f[:, i], temp_f[:, j]))
					corr = np.array(corr)
					if tp.shape[0] == 0:
						tp = corr
					else:
						tp = np.vstack((tp, corr))
				else:
					break

		return tp

# Experiment(97, 'Deepfakes', 'au', 'corr', 'one-class', 100, 100)