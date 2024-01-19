import cv2

cap = cv2.VideoCapture(0)
cascade_model = cv2.CascadeClassifier('models/cascade.xml') 

if (cap.isOpened()== False): 
	print("Error opening video file") 

while(cap.isOpened()): 
	ret, frame = cap.read() 
	if ret == True: 
		gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		detect_rect = cascade_model.detectMultiScale(gray_img, 1.1, 9)
		# Membuat bounding box untuk setiap objek yang dideteksi
		for(x, y, w, h) in detect_rect: 
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
		cv2.imshow('Frame', frame) 
		
		if cv2.waitKey(25) & 0xFF == ord('q'): 
			break
	else: 
		break

cap.release() 
cv2.destroyAllWindows()