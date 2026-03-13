import os
import shutil

print("starting dataset")
raw_images_folder= "data/facial_emotion_recognition/images/images"
train_output_folder ="data/facial_emotion_recognition/train"
emotion_lookup = {
    "Anger":"anger",
    "Contempt":"contempt",
    "Disgust":  "disgust" ,
    "Fear":"fear",
    "Happy": "happy",
    "Neutral": "neutral" ,
    "Sad":"sad",
    "Surprised": "surprised"}
for emotion_name in emotion_lookup.values( ):
    folder_path= os.path.join(train_output_folder , emotion_name)
    os.makedirs( folder_path ,exist_ok=True )



for person_id in  os.listdir( raw_images_folder ):
    person_directory= os.path.join(raw_images_folder, person_id)
    if not os.path.isdir(person_directory ):
        continue
    
    for img_file in os.listdir(person_directory):
        label_name =os.path.splitext( img_file )[0]
        if label_name not in emotion_lookup:
            continue
        emotion_category= emotion_lookup[ label_name]
        original_file_path =os.path.join(person_directory , img_file )
        new_file_name= person_id  + "_" + img_file

        new_file_path =os.path.join( train_output_folder,
            emotion_category,
            new_file_name )
        shutil.copy(original_file_path , new_file_path )

print("finished reorganizing dataset")