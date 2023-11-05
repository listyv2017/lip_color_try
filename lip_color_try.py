import os
from skimage import io
import dlib
import cv2
import numpy as np



# import image
def read_image(img_path):
	if not os.path.isfile(img_path):
		print(f"file is not found {img_path}")

	#ndarray(1280*1920*3)
	img = cv2.imread(img_path)
	
	return img

def find_lip_landmark(img):

	img_copy = img.copy()
	# step1 faceを認識するインスタンスをたてる
	detector = dlib.get_frontal_face_detector()
	# step2 landmark pointを抽出する
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
 	#detectorをつかって顔認識
	faces = detector(img)
 
	if not faces:
		print("faces is not found")

	for face in faces:
    	# 顔のランドマーク検出
		landmark = predictor(img, face)

		lm_list = []
		for i in range(48, 67):
			x = landmark.part(i).x
			y = landmark.part(i).y
			lm_list.append([x, y])

		# 画像の幅と高さを取得
		height, width, _ = img.shape

		# ウィンドウサイズを計算（画面の半分の大きさにする場合）
		window_width = width // 2
		window_height = height // 2

    	# ランドマーク(小さな青い\)描画
		# iはインデックス landmarkは座標のリスト
		# 1は縁の半径 (255, 0, 0)は色の指定 -1は塗りつぶしの4オプション
		for (i, (x, y)) in enumerate(lm_list):
			cv2.circle(img_copy, (x, y), 1, (255, 0, 0), -1)

		print(lm_list)
		#sampleはwindow name

		
  
		return lm_list

#リップの色を変える
def change_lip_color(img, lm_list, r, g, b):

	img_copy = img.copy()
 
	rec1 = np.array(lm_list[:12]).reshape(-1,1,2)
	rec2 = np.array(lm_list[12:]).reshape(-1,1,2)

	
	# 色変更 
	image_changed_color = cv2.fillPoly(img_copy, [rec1, rec2], (b, g, r))
	cv2.imshow('sample', img_copy)
	# 平滑化 0は自動的に適切な標準偏差が計算される
	image_changed_color = cv2.GaussianBlur(image_changed_color, (5, 5), 0)

 
 	# 画像の幅と高さを取得
	height, width, _ = img_copy.shape
	window_width = width 
	window_height = height 

	# 元画像とカラー後の画像を組み合わせる
	blend=cv2.addWeighted(img, 0.7, image_changed_color, 0.3, 0)
 
	# 左: 元画像  右: カラー後blend
	conbine_img = np.hstack((img, blend))
 
 
	cv2.namedWindow('sample', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('sample', window_width, window_height)
	cv2.imshow('sample', conbine_img)
	cv2.waitKey(7500)
	cv2.destroyAllWindows()
 
	return 1



if __name__ == "__main__":
	img_path = "./Celebrity Faces Dataset/Angelina Jolie/007_cabbfcbb.jpg"
	img = read_image(img_path)
	print(img)
	print(img.shape)
	
	r, g, b = map(int, input("input color RGB: ").split())

	lm_list = find_lip_landmark(img)
	change_lip_color(img, lm_list, r, g, b)
