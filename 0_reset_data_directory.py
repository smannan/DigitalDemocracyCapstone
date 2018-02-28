import os, shutil

cleaned_folder = 'data/cleaned/'
training_folder = 'data/training/'
for the_file in os.listdir(cleaned_folder):
    file_path = os.path.join(cleaned_folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)
        
for the_file in os.listdir(training_folder):
    file_path = os.path.join(training_folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)