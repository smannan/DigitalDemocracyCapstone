import sys, os, shutil

temp_folder = 'temp'
training_folder = 'training_output'

if sys.argv[0] != '/opt/conda/lib/python3.5/site-packages/ipykernel_launcher.py':
    if len(sys.argv) > 1:
        temp_folder = sys.argv[1]
    if len(sys.argv) > 2:
        training_folder = sys.argv[2]

if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    
if not os.path.exists(training_folder):
    os.makedirs(training_folder)
    
for the_file in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, the_file)
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