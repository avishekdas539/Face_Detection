import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



cap = cv2.VideoCapture(0)
_,frame = cap.read()
W, H, _ = frame.shape

model = tf.keras.models.load_model("last_100epochs.h5")

# W, H = 450, 450
# coordinates_x = []
# coordinates_y = []


# sec_counter = 0

# while sec_counter<10000:
while True:
    _ , frame = cap.read()
    # frame = frame[50:500, 50:500,:]
    frame = cv2.flip(frame, 1)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = model.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # x_min, y_min = tuple(np.multiply(sample_coords[:2], [H,W]).astype(int))
        # x_max, y_max = tuple(np.multiply(sample_coords[2:], [H,W]).astype(int))
        # coordinates_x.append((x_max+x_min)/2)
        # coordinates_y.append((y_max+y_min)/2)

        
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [H,W]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [H,W]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [H,W]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [H,W]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [H,W]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow('FaceTrack', frame)
    # sec_counter += 100
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# plt.plot(coordinates_x, coordinates_y)
# plt.show()