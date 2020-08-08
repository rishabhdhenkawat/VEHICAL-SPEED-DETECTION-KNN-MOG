
## VECHICLE SPEED CALCULATION USING OPEN CV PYTHON

#### SUMMARY ####
This project estimates the real time vehical speed on roads from any real time stream or recorded video and done for the evaluation by synergy labs  pvt ltd.
Libraries and funtions used : KNN background subtraction and morphology to isolate the vehicles and detect their contours, dlib, open cv for image processing , numpy for mathmetical operations, flask as framework, docker for deployment.  

#### Vehicle Tracking ####

To find a car's speed, we need to know how its moving from frame to frame. We can already detect cars on any given frame, but we need a kind of permanence to detect as the move in the video. This is a rather long process, but in general we compare the current detections to the previous detections, and based on manually set parameters, we determine whether or not the new detections are valid movement
#### Speed Calculation ####

As the distance of road is not provided in the video so aveage mode is used for the calculation of speed where average of some initial vehicals 
speed with respect to  coordinates is taken  and converted into real world speeds using a defined parameter which depends on the deviation from 
calculated average speed.
*Average mode* samples a certain number of vehicles to find there average speed on screen (in pixels). Subsequent cars are compared to the average, and their speeds are reported as percent differences from the average. This mode is useful when you don't know the distance of the road in the video, so it can be applied to almost any road. It's important to note that speed is calculated once a vehicle passes the light blue line 

Final speed is provided by dividing the difference between initial and final coordinates by time until vehical crossed the light blue line
Optimization note: processing time is compansated with real time hence the time used for processing does not affect the actual time, vehical has taken.

#### INSTRUCTIONS TO RUN
1. USING FLASK
   a. Clone this repositry
   b. Install the required libraries from requirements.txt file
   c. Run app.py python3 file
   d. Open any browser type localhost:5000/
   e. Upload the video to be processed.
2. USING DOCKER
   a. Directly bulid the docker image using docker file and run the image.
    
    
    
### Refrences ###

StackOverflow post: (https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515), 

Research Paper :sDetection of Vehicle Position and Speed using Camera Calibration and Image Projection Methods
Author links open overlay panelAlexander A SGunawanaDeasy ApriliaTanjungaFergyanto E.Gunawan

###### Author
Rishabh Dhenkawat
