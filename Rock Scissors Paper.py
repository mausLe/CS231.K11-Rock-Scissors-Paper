import time
import cv2
import numpy as np
import imutils
import math

cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('new1.avi', fourcc, 20.0, (640, 480))

while 1:
    start_time = time.time()
    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # print(frame.shape)
        kernel = np.ones((3, 3), np.uint8)

        # define region of interest
        roi = frame[100:300, 100:300]

        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([0, 47, 24], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin colour image
        thresh = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        # mask = cv2.dilate(mask, kernel, iterations=4)

        # blur the image
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

        # find contours
        thresh = cv2.threshold(thresh, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)

        # find contours in thresholded image, then grab the largest
        # one
        Contr = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        Contr = imutils.grab_contours(Contr)
        cnt = max(Contr, key=cv2.contourArea)
        cv2.drawContours(roi, cnt, -1, (0, 0, 255), 3)
        # out.write(frame)

        # approx the contour a little
        # Calculates a contour perimeter or a curve length
        # 1st parameter: Input vector of 2D points
        # 2nd parameter: Flag indicating whether the curve is closed or not
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        # Approximates a polygonal curve(s) with the specified precision
        # cv2.approxPolyDP(curve, epsilon, closed[, approxCurve])
        #
        # epsilon: Parameter specific approx accuracy, max distance between original curve and its approximation
        # 3th parameter: True - approxed curve is closed, otherwise...
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Technically, approx is the flexible Max contour that scaling dependently Cam-hand distance
        # make convex hull around hand
        hull = cv2.convexHull(cnt)

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

        # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)  # pass returnPoints = False to find Convex defects
        defects = cv2.convexityDefects(approx, hull)
        # convexityDefects returns 4 values:
        # l = no. of defects
        l = 0
        # print(len(defects))
        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000:
                # cv2.putText(image,text, )
                cv2.putText(frame, '', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'rock', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 2:
            cv2.putText(frame, 'scissors', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 3:
            cv2.putText(frame, 'paper', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 4:
            cv2.putText(frame, 'paper', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 5:
            cv2.putText(frame, 'paper', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        end_time = time.time()
        print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    except:
        pass

    # out.write(frame)
    # show the windows
    cv2.imshow('thresh', thresh)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q') or k == ord('p'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
