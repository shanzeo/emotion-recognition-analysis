import os
import shutil

print("Preparing dataset...")

source_dir = "data/facial_emotion_recognition/images/images"
target_dir = "data/facial_emotion_recognition/train"

emotion_map = {
    "Anger": "anger",
    "Contempt": "contempt",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "surprised"
}

# Make sure train emotion folders exist
for emotion_folder in emotion_map.values():
    os.makedirs(os.path.join(target_dir, emotion_folder), exist_ok=True)

# Loop through each numbered person folder
for person_folder in os.listdir(source_dir):
    person_path = os.path.join(source_dir, person_folder)

    if not os.path.isdir(person_path):
        continue

    for image_file in os.listdir(person_path):
        file_name_no_ext = os.path.splitext(image_file)[0]

        if file_name_no_ext not in emotion_map:
            continue

        emotion_folder = emotion_map[file_name_no_ext]

        old_path = os.path.join(person_path, image_file)
        new_name = f"{person_folder}_{image_file}"
        new_path = os.path.join(target_dir, emotion_folder, new_name)

        shutil.copy(old_path, new_path)

print("Dataset reorganized successfully.")