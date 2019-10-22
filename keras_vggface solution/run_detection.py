import cv2
import sys 
import keras_vggface
from mtcnn.mtcnn import MTCNN


# extract faces from a given image
def extract_face(image):
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(image)
    # extract the bounding box from the faces
    for result in results:
        (x1, y1, width, height) = result['box']
        cv2.rectangle(image, (x1, y1), (x1+width, y1+height), (200, 50, 50), 10)
    
    return image

video_capture = cv2.VideoCapture(0)

while True:

    # Capture frame-by-frame
    retval, frame = video_capture.read()
    output_frame = extract_face(frame)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the camera view
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit() 
