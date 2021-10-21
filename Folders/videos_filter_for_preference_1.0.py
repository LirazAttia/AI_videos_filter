
#scintific packs
from pathlib import Path
import numpy as np
#AI model pack
import tensorflow.keras
#image, video and copy files packs
from PIL import Image, ImageOps
import cv2
import shutil
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

###########################################################################################
# Load the model
model_path = r"D:\YossiY4\Desktop\GitHub_Folder\AI_videos_filter\Folders\model\keras_model.h5"
model = tensorflow.keras.models.load_model(model_path)
###########################################################################################

original_files_path =  r"D:\YossiY4\Desktop\GitHub_Folder\AI_videos_filter\Folders\original_files"
converted_files_path= r"D:\YossiY4\Desktop\GitHub_Folder\AI_videos_filter\Folders\converted_files"
relevant_files_path = r"D:\YossiY4\Desktop\GitHub_Folder\AI_videos_filter\Folders\relevant_files"
img_path = r"D:\YossiY4\Desktop\GitHub_Folder\AI_videos_filter\Folders\temp_frame.jpg" #add \frame.jpg
 
##########################################################################################

def model_prediction(image):
    """ """
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    #image = Image.open('test_photo.jpg')
    image = Image.open(image)

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    try:
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
    except:
        print("exeption") # raise
        return False

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    ####image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return prediction

def delet_frame(img_path):
    """ """
    img_path = Path(img_path)
    img_path.unlink()
    img_path = str(img_path)

def prediction_count_add(prediction_count, prediction):
    """ """
    #print(prediction[0,0], prediction[0,1]) #how do I know for sure that 0 is bat and 1 is no bat
    if prediction[0,0] > 0.95:
        prediction_count += 1
    return prediction_count

def all_frames_check(file_path:str, img_path: str , prediction_tresh = 10, count = 0, prediction_count = 0 ):
    """ """
    prediction = np.array([0, 0]) ### ???
     
    file_path = str(file_path)
    video_cap = cv2.VideoCapture(file_path)
    if (video_cap.isOpened() == False):
        print("Error opening video stream or file") ###rise!!
        return False

    success, frame = video_cap.read()
    while success:
        img_file = cv2.imwrite(img_path, frame)  # save frame as JPEG file
        prediction = model_prediction(img_path)
        #prediction check
        prediction_count = prediction_count_add(prediction_count, prediction)
        if prediction_count >= prediction_tresh:
            delet_frame(img_path)
            return True
        #new_frame
        count += 1
        success, frame = video_cap.read()
    
    delet_frame(img_path)
    return False
    
def save_file_with_True_prediction(file_path: str, relevant_files_path: str, original_files_path: str):
    """ """
    file_path = Path(file_path)
    file_name = str(file_path.name)
    file_name = file_name[4:] #taking off "new_" from the start of the name
    original_file_path = original_files_path + "\\" + file_name
    print("Relevant File >>> ", original_file_path)
    shutil.copy(original_file_path, relevant_files_path)

def iteration_over_folder(original_files_path, converted_files_path, relevant_files_path, img_path, model):
    """ """
    print("Processing...  ",  converted_files_path)
    for file_path in Path(converted_files_path).iterdir():
        #file_start_str = str(file_path)
        #file_start_str = file_start_str
        if file_path.is_file() and file_path.suffix == '.avi' and str(file_path.name)[0:4] == "new_" :  
            file_prediction =  all_frames_check(file_path, img_path, prediction_tresh = 10)
            if file_prediction == True:
                save_file_with_True_prediction(file_path, relevant_files_path, original_files_path)

################################################################################################################################
################################################################################################################################

if __name__ == "__main__":
    session_name =input("Session name >>> ")
    iteration_over_folder(original_files_path, converted_files_path, relevant_files_path, img_path, model)
    print("\nFINSHED ", session_name)
    input("For approvel press enter or any other key >>> ")