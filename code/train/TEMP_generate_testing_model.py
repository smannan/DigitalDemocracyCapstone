from model import create_testing_model

withheld_video_ids = [4221, 4224, 4227, 4230, 7598, 6476, 6483, 5150, 7134, 6235]

# Change NB to NN to generate a neural network model 

create_testing_model("../../data/original/raw.txt", 
	"../../data/original/upleveled.txt", 
	"NN", withheld_video_ids)

create_testing_model("../../data/original/raw.txt", 
	"../../data/original/upleveled.txt", 
	"NB", withheld_video_ids)