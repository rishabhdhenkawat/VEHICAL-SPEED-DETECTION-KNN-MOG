from flask import Flask, render_template, Response,request

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import os
# import magic
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask import Flask


UPLOAD_FOLDER = 'upload'
app = Flask(__name__)

import os
# import magic
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask import Flask
global name
UPLOAD_FOLDER = 'upload'
input=''

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['avi','mp4'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            global input
            input=filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            return render_template('index.html')
        else:
            flash('mp4','avi')
            return redirect(request.url)


def index():
    """Video streaming home page."""

        #return 'file uploaded successfully'
    return render_template('index.html')

@app.route('/', methods=['POST'])
def gen():
	import sys
	import time
	import json
	import re

	import cv2
	import numpy as np
	import cv2
	import dlib
	import time
	import threading
	import math
#	from vehicle_counter import VehicleCounter

	road = None
	WIDTH = 1280
	HEIGHT = 720


	def estimateSpeed(location1, location2):
		d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
		# ppm = location2[2] / carWidht
		ppm = 16.8
		d_meters = d_pixels / ppm
		# print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
		fps = 18
		speed = d_meters * fps * 3.6
		return speed




		# Write output to video file
	#	out = cv2.VideoWriter('./outpy.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))


	#if len(sys.argv) < 2:
	#	raise Exception("No road specified.")

	road_name = "80_donner_lake"

	with open('settings.json') as f:
		data = json.load(f)
		print(data)
		try:
			road = data[road_name]
		except KeyError:
			raise Exception('Road name not recognized.')

	WAIT_TIME = 1

	# Colors for drawing on processed frames
	DIVIDER_COLOR = (255, 255, 0)
	BOUNDING_BOX_COLOR = (255, 0, 0)
	CENTROID_COLOR = (0, 0, 255)

	# For cropped rectangles
	ref_points = []
	ref_rects = []

	def nothing(x):
		pass

	def click_and_crop (event, x, y, flags, param):
		global ref_points

		if event == cv2.EVENT_LBUTTONDOWN:
			ref_points = [(x,y)]

		elif event == cv2.EVENT_LBUTTONUP:
			(x1, y1), x2, y2 = ref_points[0], x, y

			ref_points[0] = ( min(x1,x2), min(y1,y2) )

			ref_points.append ( ( max(x1,x2), max(y1,y2) ) )

			ref_rects.append( (ref_points[0], ref_points[1]) )

	# Write cropped rectangles to file for later use/loading
	def save_cropped():
		global ref_rects

		with open('../Car-Speed-Detection-master/settings.json', 'r+') as f:
			data = json.load(f)
			data[road_name]['cropped_rects'] = ref_rects

			f.seek(0)
			json.dump(data, f, indent=4)
			f.truncate()

		print('Saved ref_rects to settings.json!')

	# Load any saved cropped rectangles
	def load_cropped ():
		global ref_rects

		ref_rects = road['cropped_rects']

		print('Loaded ref_rects from settings.json!')

	# Remove cropped regions from frame
	def remove_cropped (gray, color):
		cropped = gray.copy()
		cropped_color = color.copy()

		for rect in ref_rects:
			cropped[ rect[0][1]:rect[1][1], rect[0][0]:rect[1][0] ] = 0
			cropped_color[ rect[0][1]:rect[1][1], rect[0][0]:rect[1][0] ] = (0,0,0)


		return cropped, cropped_color

	def filter_mask (mask):
		# I want some pretty drastic closing
		kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
		kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
		kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

		# Remove noise
		opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
		# Close holes within contours
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close)
		# Merge adjacent blobs
		dilation = cv2.dilate(closing, kernel_dilate, iterations = 2)

		return dilation

	def get_centroid (x, y, w, h):
		x1 = w // 2
		y1 = h // 2

		return(x+x1, y+y1)

	def detect_vehicles (mask):

		MIN_CONTOUR_WIDTH = 10
		MIN_CONTOUR_HEIGHT = 10

		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		matches = []

		# Hierarchy stuff:
		# https://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy
		for (i, contour) in enumerate(contours):
			x, y, w, h = cv2.boundingRect(contour)
			contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

			if not contour_valid or not hierarchy[0,i,3] == -1:
				continue

			centroid = get_centroid(x, y, w, h)

			matches.append( ((x,y,w,h), centroid) )

		return matches

	def process_frame(frame_number, frame, bg_subtractor):
		processed = frame.copy()

		gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

		# remove specified cropped regions
		cropped, processed = remove_cropped(gray, processed)

		#if car_counter.is_horizontal:
		cv2.line(processed,(0,250),(1200,250), DIVIDER_COLOR, 1)
		#else:
		#	cv2.line(processed, (car_counter.divider, 0), (car_counter.divider, frame.shape[0]), DIVIDER_COLOR, 1)

		fg_mask = bg_subtractor.apply(cropped)
		fg_mask = filter_mask(fg_mask)

		matches = detect_vehicles(fg_mask)

		for (i, match) in enumerate(matches):
			contour, centroid = match

			x,y,w,h = contour

			#cv2.rectangle(processed, (x,y), (x+w-1, y+h-1), BOUNDING_BOX_COLOR, 1)
			cv2.circle(processed, centroid, 2, CENTROID_COLOR, -1)

	#	#.update_count(matches, frame_number, processed)

		cv2.imshow('Filtered Mask', fg_mask)

		return processed,matches

	# https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
	def lane_detection (frame):
		gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

		cropped = remove_cropped(gray)


	# I was going to use a haar cascade, but i decided against it because I don't want to train one, and even if I did it probably wouldn't work across different traffic cameras

	# I think KNN works better than MOG2, specifically with trucks/large vehicles

	bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
	car_counter = None

	load_cropped()

	cap = cv2.VideoCapture(input)
	#cap = cv2.VideoCapture(road['stream_url'])
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

	cv2.namedWindow('Source Image')
	cv2.setMouseCallback('Source Image', click_and_crop)

	frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

	frame_number = -1

	rectangleColor = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	fps = 0

	carTracker = {}
	carNumbers = {}
	carLocation1 = {}
	carLocation2 = {}
	speed = [None] * 1000
	while True:
		frame_number += 1
		ret, frame = cap.read()
		start_time = time.time()
		image=frame
		resultImage = image.copy()
		scale_percent = 60  # percent of original size
		width = int(frame.shape[1] * scale_percent / 100)
		height = int(frame.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

		if not ret:
			print('Frame capture failed, stopping...')
			break

		#if car_counter is None:
			#car_counter = VehicleCounter(frame.shape[:2], road, cap.get(cv2.CAP_PROP_FPS), samples=0)

		frame,matches = process_frame(frame_number, frame, bg_subtractor)

		#cv2.imshow('Source Image', frame)
		#cv2.imshow('Processed Image', processed)

		frameCounter = frameCounter + 1

		carIDtoDelete = []

		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(frame)

			if trackingQuality < 7:
				carIDtoDelete.append(carID)

		for carID in carIDtoDelete:
			print('Removing carID ' + str(carID) + ' from list of trackers.')
			print('Removing carID ' + str(carID) + ' previous location.')
			print('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			carLocation1.pop(carID, None)
			carLocation2.pop(carID, None)


		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		for (i, match) in enumerate(matches):
			contour, centroid = match

			x, y, w, h = contour
			#cv2.rectangle(gray, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
			x_bar = x + 0.5 * w
			y_bar = y + 0.5 * h

			matchCarID = None

			for carID in carTracker.keys():
				trackedPosition = carTracker[carID].get_position()

				t_x = int(trackedPosition.left())
				t_y = int(trackedPosition.top())
				t_w = int(trackedPosition.width())
				t_h = int(trackedPosition.height())

				t_x_bar = t_x + 0.5 * t_w
				t_y_bar = t_y + 0.5 * t_h

				if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
						x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
					matchCarID = carID

			if matchCarID is None:
				print('Creating new tracker ' + str(currentCarID))

				tracker = dlib.correlation_tracker()
				tracker.start_track(frame, dlib.rectangle(x, y, x + w, y + h))

				carTracker[currentCarID] = tracker
				carLocation1[currentCarID] = [x, y, w, h]

				currentCarID = currentCarID + 1

		# cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position()

			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())

			cv2.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

			# speed estimation
			carLocation2[carID] = [t_x, t_y, t_w, t_h]

		end_time = time.time()

		if not (end_time == start_time):
			fps = 1.0 / (end_time - start_time)

		# cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

		for i in carLocation1.keys():
			if frameCounter % 1 == 0:
				[x1, y1, w1, h1] = carLocation1[i]
				[x2, y2, w2, h2] = carLocation2[i]

				# print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
				carLocation1[i] = [x2, y2, w2, h2]

				# print 'new previous location: ' + str(carLocation1[i])
				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
						speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

					# if y1 > 275 and y1 < 285:
					if speed[i] != None and y1 >= 180:
						cv2.putText(frame, str(int(speed[i])) + " km/hr", (int(x1 + w1 / 2), int(y1 - 5)),
									cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

			# print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

			# else:
			#	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

			# print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
		cv2.imshow('result', frame)
		frame = cv2.imencode('.jpg', frame)[1].tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		# Write the frame into the file 'output.avi'
		# out.write(resultImage)

		if cv2.waitKey(33) == 27:
			break



	print('Closing video capture...')
	cap.release()
	cv2.destroyAllWindows()
	print('Done.')

@app.route('/video_feed')
def video_feed():
	"""Video streaming route. Put this in the src attribute of an frame tag."""
	return Response(gen(),
					mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0',debug=True)