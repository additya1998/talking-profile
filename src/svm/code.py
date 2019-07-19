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
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

frame_dict = {}
frame_dict['051'] = (717, 464, 622)
frame_dict['088'] = (665, 414, 564)
frame_dict['221'] = (718, 441, 619)
frame_dict['237'] = (757, 486, 660)
frame_dict['332'] = (695, 468, 610)
frame_dict['368'] = (761, 464, 627)
frame_dict['378'] = (761, 489, 658)
frame_dict['420'] = (1126, 724, 962)
frame_dict['423'] = (1814, 965, 1407)
frame_dict['494'] = (802, 537, 735)
frame_dict['515'] = (766, 427, 600)
frame_dict['912'] = (647, 491, 637)
frame_dict['060'] = (844, 530, 714)
frame_dict['206'] = (714, 534, 702)
frame_dict['236'] = (651, 409, 547)
frame_dict['257'] = (763, 473, 653)
frame_dict['367'] = (649, 458, 596)
frame_dict['371'] = (718, 537, 686)
frame_dict['417'] = (861, 522, 723)
frame_dict['421'] = (1038, 772, 986)
frame_dict['445'] = (1053, 771, 982)
frame_dict['496'] = (1496, 913, 1254)
frame_dict['519'] = (751, 503, 688)
frame_dict['927'] = (697, 506, 676)

def au_features(idx, prefix_path='/home/additya/NUS/features/FaceForensics++/LongVideos_new/merged_sequences/Deepfakes/c23/videos/'):
    feature_path = os.path.join(prefix_path, idx, idx + '.csv')
    features = np.array([])
    with open(feature_path, "r") as face_csv:
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
            feature_v += [float(row[' AU45_r'])]
            feature_v = np.array(feature_v)
            if features.shape[0] == 0:
                features = feature_v
            else:
                features = np.vstack((features, feature_v))
    (total_f, fake_s, fake_e) = frame_dict[idx]
    return features, total_f, fake_s, fake_e

def face_features(idx, prefix_path='/home/additya/NUS/features/FaceForensics++/LongVideos_new/merged_sequences/Deepfakes/c23/videos/'):
    feature_path = os.path.join(prefix_path, idx, idx + '.csv')
    features = np.array([])
    with open(feature_path, "r") as face_csv:
        csv_reader = csv.DictReader(face_csv)
        for row in csv_reader:
            ht = [float(row[' pose_Tx']), float(row[' pose_Ty']), float(row[' pose_Tz'])]
            hr = [float(row[' pose_Rx']), float(row[' pose_Ry']), float(row[' pose_Rz'])]
            gaze = [float(row[' gaze_0_x']), float(row[' gaze_0_y']), float(row[' gaze_0_z']),
                    float(row[' gaze_1_x']), float(row[' gaze_1_y']), float(row[' gaze_1_z'])]
            blink_left = np.linalg.norm(np.array([float(row[' eye_lmk_x_17']), float(row[' eye_lmk_y_17'])])
                      - np.array([float(row[' eye_lmk_x_11']), float(row[' eye_lmk_y_11'])]))
            blink_right = np.linalg.norm(np.array([float(row[' eye_lmk_x_45']), float(row[' eye_lmk_y_45'])])
                        - np.array([float(row[' eye_lmk_x_39']), float(row[' eye_lmk_y_39'])]))
            lip_gap_h = np.linalg.norm(np.array([float(row[' x_65']), float(row[' y_65'])])
                - np.array([float(row[' x_62']), float(row[' y_62'])]))
            lip_gap_w = np.linalg.norm(np.array([float(row[' x_64']), float(row[' y_64'])])
                        - np.array([float(row[' x_60']), float(row[' y_60'])]))
            face_length = np.linalg.norm(np.array([float(row[' x_27']), float(row[' y_27'])])
                - np.array([float(row[' x_8']), float(row[' y_8'])])) + np.finfo(float).eps
            lip_gap_h = lip_gap_h / face_length # Normalize
            lip_gap_w = lip_gap_w / face_length # Normalize
            blink_left = blink_left / face_length # Normalize
            blink_right = blink_right / face_length # Normalize
            feature_v = ht + hr + gaze + [blink_left, blink_right] + [lip_gap_w, lip_gap_h]
            feature_v = np.array(feature_v)
            if features.shape[0] == 0:
                features = feature_v
            else:
                features = np.vstack((features, feature_v))
    (total_f, fake_s, fake_e) = frame_dict[idx]
    assert(total_f == features.shape[0])
    return features, total_f, fake_s, fake_e

def generate_tp(features, tp_type, frame_length, stride=1, fgap=1, feature_subset = None):
    
    if features.shape[0] < frame_length:
        return np.array([])

    if feature_subset is not None:
        features = features[:, feature_subset]

    if tp_type == 'original':
        
        tp = np.array([])
        for X in range(0, features.shape[0], stride):
            temp_f = np.array([])
            for Y in range(X + fgap, features.shape[0], fgap):
                if temp_f.shape[0] == 0:
                    temp_f = features[Y, :] - features[Y - fgap, :]
                else:
                    temp_f = np.vstack((temp_f, features[Y, :] - features[Y - fgap, :]))
                if temp_f.shape[0] == frame_length - 1:
                    break
            if temp_f.shape[0] == frame_length - 1:
                if tp.shape[0] == 0:
                    tp = temp_f.flatten()
                else:
                    tp = np.vstack((tp, temp_f.flatten()))
            else:
                break

    else:
        tp = np.array([])
        for X in range(0, features.shape[0], stride):
            temp_f = features[X, :]
            for Y in range(X + fgap, features.shape[0], fgap):
                temp_f = np.vstack((temp_f, features[Y, :]))
                if temp_f.shape[0] == frame_length:
                    break
            if temp_f.shape[0] == frame_length:
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

def get_classifier(train_pos_X, train_neg_X):
    train_pos_Y = np.ones(train_pos_X.shape[0])
    train_neg_Y = np.zeros(train_neg_X.shape[0])
    
    train_X = np.vstack((train_pos_X, train_neg_X))
    train_Y = np.hstack((train_pos_Y, train_neg_Y))
    train_X, train_Y = shuffle(train_X, train_Y)

    
    clf = SVC(gamma='scale', probability=True)
#     clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    
    clf.fit(train_X, train_Y)
    return clf

def run_experiment(train_count, frame_length, pos_fname, ftype, tp_type, path, stride=1, fgap=1):
    
    if ftype == 'face':
        original_features, total_f, fake_s, fake_e = face_features(pos_fname, prefix_path=path)
    else:
        original_features, total_f, fake_s, fake_e = au_features(pos_fname, prefix_path=path)
        
    print('total frames: ', total_f)
    print('number of fake frames: ', fake_e - fake_s + 1)
    print('fake frames start:', fake_s, 'fake end:', fake_e)

    train_last_frame_start = stride * (train_count - 1) 
    train_last_frame_end = train_last_frame_start + (frame_length - 1) * fgap 
    print("Max frames accessed:", train_last_frame_start, train_last_frame_end)
    
    train_positive = generate_tp(original_features[:train_last_frame_end], tp_type=tp_type, frame_length=frame_length, stride=stride, fgap=fgap)

    test_positive = generate_tp(original_features[train_last_frame_start + 1:fake_s], tp_type=tp_type, frame_length=frame_length, stride=stride, fgap=fgap)
    if test_positive.shape[0] == 0:
        test_positive = generate_tp(original_features[fake_e + 1:], tp_type=tp_type, frame_length=frame_length, stride=stride, fgap=fgap)
    else:
        test_positive_new = generate_tp(original_features[fake_e + 1:], tp_type=tp_type, frame_length=frame_length, stride=stride, fgap=fgap)
        if test_positive_new.shape[0] != 0:
            test_positive = np.vstack((test_positive, test_positive_new))
        
    test_negative = generate_tp(original_features[fake_s:fake_e + 1], tp_type=tp_type, frame_length=frame_length, stride=stride, fgap=fgap)
    
    train_negative = np.array([])
    for f in frame_dict.keys():
        if f == pos_fname:
            continue
        if ftype == 'face':
            feat, _, _, _ = face_features(f, '/home/additya/NUS/features/FaceForensics++/LongVideos/original_sequences/c23/videos/')
        else:
            feat, _, _, _ = au_features(f, '/home/additya/NUS/features/FaceForensics++/LongVideos/original_sequences/c23/videos/')
        full_tp = generate_tp(feat, tp_type=tp_type, frame_length=frame_length, stride=stride, fgap=fgap)
        full_tp_less_overlap = full_tp
        if full_tp_less_overlap.shape[0] == 0:
            continue
        if train_negative.shape[0] == 0:
            train_negative = full_tp_less_overlap
        else:
            train_negative = np.vstack((train_negative, full_tp_less_overlap))
    
    
    added_idx = np.array([])
    total_correct_added, total_wrong_added = 0, 0
    initial_auc, final_auc = -1, -1
    
    prob_cutoff = 0.90
    frq_cutoff = 100
    
    auc_arr = []

    for it in range(50):

      print("#" * 100)
      print("Iteration:", it)

      print("Train Positive:", train_positive.shape[0])
      print("Train Negative:", train_negative.shape[0])
      print("Test Positive:", test_positive.shape[0])
      print("Test Negative:", test_negative.shape[0])

      clf = get_classifier(train_positive, train_negative)
      test_X = np.vstack((test_positive, test_negative))
      test_Y = np.hstack((np.ones(test_positive.shape[0]), np.zeros(test_negative.shape[0])))
      preds = clf.predict_proba(test_X)

      preds = preds[:, 1]
      
      fpr, tpr, thresholds = roc_curve(test_Y, preds, drop_intermediate=False)
      auc_score = np.round(roc_auc_score(test_Y, preds), 4)

      print(">>>AUC:", auc_score)
      if initial_auc == -1:
        initial_auc = auc_score
      final_auc = auc_score
      
      if (it > 0) and (it % 5 == 0):
          prob_cutoff -= 0.05
          frq_cutoff += 25
          prob_cutoff = max(prob_cutoff, 0.75)
          frq_cutoff = min(frq_cutoff, 1000)

      all_idx = preds.argsort()[::-1]
      pos_train_idx = np.array([i for i in all_idx if i not in added_idx and preds[i] > prob_cutoff])
      pos_train_idx = pos_train_idx[:min(pos_train_idx.shape[0], frq_cutoff)]

      if len(pos_train_idx) == 0:
          print("nothing to add breaking")
          break

      train_positive = np.vstack((train_positive, test_X[pos_train_idx]))
      print("Positive samples added =", pos_train_idx.shape[0])
      if added_idx.shape[0] == 0:
          added_idx = pos_train_idx
      else:
          added_idx = np.hstack((added_idx, pos_train_idx))

      CORR = (np.where(test_Y[pos_train_idx] == 1)[0].shape[0])
      WR = (np.where(test_Y[pos_train_idx] == 0)[0].shape[0])
      
      total_correct_added += CORR
      total_wrong_added += WR

      print("Correct Added:", CORR)
      print("Wrong Added:", WR)
      print("Total Correct Added:", total_correct_added)
      print("Total Wrong Added:", total_wrong_added)    

      print("#" * 100)
      auc_arr.append(auc_score)
      
    print('>>>>>>>>>>>>>>>>>>>>', initial_auc, final_auc)
    return np.array(auc_arr)


run_experiment(100, 100, '420', 'au', 'corr', '/home/additya/NUS/features/FaceForensics++/LongVideos_new/merged_sequences/Deepfakes/c23/videos/', stride=1, fgap=1)