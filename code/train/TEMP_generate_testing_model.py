from model import create_testing_model

withheld_video_ids = [4221, 4224, 4227, 4230]

create_testing_model("../../data/original/raw.txt", "../../data/original/upleveled.txt", "NB", withheld_video_ids)