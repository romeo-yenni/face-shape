import cv2
import dlib

def detect_face_shape(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Initialize face detector
    detector = dlib.get_frontal_face_detector()
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = detector(gray)

    if len(faces) == 0:
        return "No faces found in the image."

    # Assuming there's only one face in the image, you can access it like this
    face = faces[0]

    # Load the shape predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Extract face landmarks for more detailed analysis
    landmarks = predictor(gray, face)
    
    # Calculate face width and height
    face_width = landmarks.part(16).x - landmarks.part(0).x
    face_height = landmarks.part(8).y - landmarks.part(27).y
    
    # Determine face shape based on proportions
    face_ratio = face_width / face_height

    if face_ratio < 0.85:
        return "Your face shape is round."
    elif 0.85 <= face_ratio < 1.05:
        return "Your face shape is oval."
    elif face_ratio >= 1.05:
        return "Your face shape is long."

# Replace 'path/to/your/image.jpg' with the actual path to your image file
image_path = 'face.jpg'
result = detect_face_shape(image_path)
print(result)
