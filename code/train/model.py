import sys
sys.path.insert(0, '../..')

from code.process_dataset.raw import raw_data_processing
from code.process_dataset.upleveled import upleveled_data_processing
from code.process_dataset.training import create_training_data
from code.train.train_models import train_model


def create_production_model(original_raw_filename, original_upleveled_filename, model_type,
                            cleaned_raw_output_filename="../../data/cleaned/raw_cleaned.csv",
                            cleaned_bill_times_table_output_filename="../../data/cleaned/upleveled_bill_times_table.csv",
                            training_data_output_filename="../../data/training/training.csv",
                            model_folder="../../model"):
    
    raw_data_processing(input_filename=original_raw_filename,
                        output_filename=cleaned_raw_output_filename)

    upleveled_data_processing(input_filename=original_upleveled_filename,
                              output_filename=cleaned_bill_times_table_output_filename)

    create_training_data(cleaned_raw_filename=cleaned_raw_output_filename,
                         bill_times_table_filename=cleaned_bill_times_table_output_filename,
                         output_filename=training_data_output_filename)

    train_model(input_file=training_data_output_filename,
                output_folder=model_folder,
                model_type=model_type)
    
    
def create_testing_model(original_raw_filename, original_upleveled_filename, model_type, withheld_video_ids,
                         cleaned_raw_output_filename="../../data/cleaned/raw_cleaned.csv",
                         cleaned_bill_times_table_output_filename="../../data/cleaned/upleveled_bill_times_table.csv",
                         training_data_output_filename="../../data/training/training.csv",
                         withheld_raw_output_filename="../../data/cleaned/raw_cleaned_withheld.csv",
                         withheld_bill_times_table_output_filename="../../data/cleaned/upleveled_bill_times_table_withheld.csv",
                         withheld_training_data_output_filename="../../data/training/training_withheld.csv",
                         model_folder="../../model"):

    raw_data_processing(input_filename=original_raw_filename,
                        output_filename=cleaned_raw_output_filename,
                        withheld_filename=withheld_raw_output_filename,
                        withheld_ids=withheld_video_ids)

    upleveled_data_processing(input_filename=original_upleveled_filename,
                              output_filename=cleaned_bill_times_table_output_filename,
                              withheld_filename=withheld_bill_times_table_output_filename,
                              withheld_ids=withheld_video_ids)

    create_training_data(cleaned_raw_filename=cleaned_raw_output_filename,
                         bill_times_table_filename=cleaned_bill_times_table_output_filename,
                         output_filename=training_data_output_filename)

    create_training_data(cleaned_raw_filename=withheld_raw_output_filename,
                         bill_times_table_filename=withheld_bill_times_table_output_filename,
                         output_filename=withheld_training_data_output_filename)
                     
    train_model(input_file=training_data_output_filename,
                output_folder=model_folder,
                model_type=model_type)
    
    