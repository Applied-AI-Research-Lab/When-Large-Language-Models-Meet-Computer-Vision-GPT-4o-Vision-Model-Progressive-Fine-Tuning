import os
import pandas as pd
import openai
from openai import OpenAI
import json
import logging
import re
import time

class GPTmethods:
    def __init__(self, params):
        """
        Initialize the class with the provided parameters.
        The constructor sets up the OpenAI API key, model configuration, and various other
        parameters needed for generating prompts and making predictions.

        Args:
            params (dict): A dictionary containing the configuration settings.
        """
        # Access the OpenAI API key from environment variables
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        # Initialize class variables using the provided parameters
        self.model_id = params['model_id']  # The model ID to use (e.g., gpt-4o)
        self.prediction_column = params['prediction_column']  # Specifies the column where predictions will be stored
        self.pre_path = params['pre_path']  # The path to datasets
        self.data_set = params['data_set']  # Defines the path to the CSV dataset file
        self.prompt_array = params['prompt_array']  # A dictionary with additional data
        self.system = params['system']  # System-level message for context in the conversation
        self.prompt = params['prompt']  # The base prompt template
        self.feature_col = params['feature_col']  # Column name for feature input
        self.label_col = params['label_col']  # Column name for the label
        self.json_key = params['json_key']  # Key for extracting relevant data from the model's response
        self.max_tokens = params['max_tokens']  # Maximum number of tokens to generate in the response
        self.temperature = params['temperature']  # Controls response randomness (0 is most deterministic)
        self.path_to_image = params['path_to_image'] # Only for specific cases

    """
    Generates a custom prompt
    """

    def generate_prompt(self, feature):
        # Read the JSON file
        with open(self.pre_path + self.prompt_array['json_file'], 'r') as file:
            data = json.load(file)

        # Create a new dictionary with the product title and existing categories
        new_data = {
            # "product_title": feature,
            "categories": data["categories"]
        }

        # Convert the dictionary to a JSON-formatted string
        replacement = json.dumps(new_data, indent=2)

        updated_prompt = self.prompt.replace('[json]', replacement)

        # If the prompt is simple you can avoid this method by setting updated_prompt = self.prompt + feature
        return updated_prompt  # This method returns the whole new custom prompt

    """
    Creates a training and validation JSONL file for GPT fine-tuning.
    The method reads a CSV dataset, generates prompt-completion pairs for each row, and formats the data into
    the required JSONL structure for GPT fine-tuning.
    The generated JSONL file will contain system, user, and assistant messages for each training || validation instance.
    """

    def create_jsonl(self, data_type, data_set, parent_path, save_path):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(self.pre_path + data_set)
        data = []  # List to store the formatted data for each row

        # Iterate over each row in the DataFrame to format the data for fine-tuning
        for index, row in df.iterrows():
            data.append(
                {
                    "messages": [

                        {
                            "role": "user",
                            "content": self.generate_prompt(feature=row[self.feature_col])  # Generate user prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": parent_path + str(row['id']) + ".jpg"
                                    }
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": f"{{{self.json_key}: {row[self.label_col]}}}"  # Assistant's response
                        }
                    ]
                }
            )

        # Define the output file path for the JSONL file
        output_file_path = self.pre_path + save_path + "ft_dataset_gpt_" + data_type + ".jsonl"  # Define the path
        # Write the formatted data to the JSONL file
        with open(output_file_path, 'w') as output_file:
            for record in data:
                # Convert each dictionary record to a JSON string and write it to the file
                json_record = json.dumps(record)
                output_file.write(json_record + '\n')

        # Return a success message with the file path
        return {"status": True, "data": f"JSONL file '{output_file_path}' has been created."}

    """
    Create a conversation with the GPT model by sending a series of messages and receiving a response.
    This method constructs the conversation and returns the model's reply based on the provided messages.
    """

    def gpt_conversation(self, conversation):
        # Instantiate the OpenAI client to interact with the GPT model
        client = OpenAI()
        # Send the conversation to the model and get the response
        completion = client.chat.completions.create(
            model=self.model_id,  # Specify the model to use for the conversation
            messages=conversation  # Pass the conversation history as input
        )
        # Return the message from the model's response
        return completion.choices[0].message

    """
    Cleans the response from the GPT model by attempting to extract and parse a JSON string.
    If the response is already in dictionary format, it is returned directly.
    If the response contains a JSON string, it will be extracted, cleaned, and parsed.
    If no valid JSON is found or a decoding error occurs, an error message is logged.
    """

    def clean_response(self, response, a_field):
        if isinstance(response, dict):
            return {"status": True, "data": response}
        try:
            start_index = response.find('{')
            end_index = response.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = response[start_index:end_index + 1]
                # Replace single quotes with double quotes
                json_str = re.sub(r"'", '"', json_str)
                # Handle missing quotes around keys
                json_str = re.sub(r'([a-zA-Z0-9_]+):', r'"\1":', json_str)
                # Modified regex to handle multi-word values
                json_str = re.sub(r':\s*([a-zA-Z0-9_\s]+)([\s,}\]])', r': "\1"\2', json_str)
                # Clean up any double spaces in the values
                json_str = re.sub(r'\s+', ' ', json_str)
                json_data = json.loads(json_str)
                return {"status": True, "data": json_data}
            else:
                logging.error(f"No JSON found in the response. The input '{a_field}', resulted in the "
                              f"following response: {response}")
                return {
                    "status": False,
                    "data": f"No JSON found in the response. The input '{a_field}', "
                            f"resulted in the following response: {response}"
                }
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                          f"resulted in the following response: {response}")
            return {
                "status": False,
                "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
                        f"resulted in the following response: {response}"
            }
    # def clean_response(self, response, a_field):
    #     # If the response is already a dictionary, return it directly
    #     if isinstance(response, dict):
    #         return {"status": True, "data": response}
    #
    #     try:
    #         # Attempt to extract the JSON part from the response string
    #         start_index = response.find('{')
    #         end_index = response.rfind('}')
    #
    #         if start_index != -1 and end_index != -1:
    #             # Extract and clean the JSON string
    #             json_str = response[start_index:end_index + 1]
    #
    #             # Replace single quotes with double quotes
    #             json_str = re.sub(r"\'", '"', json_str)
    #
    #             # Try parsing the cleaned JSON string
    #             json_data = json.loads(json_str)
    #             return {"status": True, "data": json_data}
    #         else:
    #             # Log an error if no JSON is found
    #             logging.error(f"No JSON found in the response. The input '{a_field}', resulted in the "
    #                           f"following response: {response}")
    #             return {"status": False, "data": f"No JSON found in the response. The input '{a_field}', "
    #                                              f"resulted in the following response: {response}"}
    #     except json.JSONDecodeError as e:
    #         # Handle JSON parsing errors
    #         logging.error(f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
    #                       f"resulted in the following response: {response}")
    #         return {"status": False,
    #                 "data": f"An error occurred while decoding JSON: '{str(e)}'. The input '{a_field}', "
    #                         f"resulted in the following response: {response}"}

    """
    Prompts the GPT model to generate a prediction based on the provided input.
    The method constructs a conversation with the model using the system message and user input, 
    and processes the model's response to return a clean, formatted prediction.
    """

    def gpt_prediction(self, input):
        conversation = []
        conversation.append({
            'role': 'user',
            "content": [
                {"type": "text", "text": self.generate_prompt(feature=input[self.feature_col])},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.path_to_image + str(input['id']) + ".jpg"
                    }
                }
            ]
        })  # Generate the prompt
        # Instead of replacing url with id, you can use the feature column directly
        # in some cases "url": input[self.feature_col]

        # Get the model's response by passing the conversation to gpt_conversation
        conversation = self.gpt_conversation(conversation)
        # Extract the content of the GPT model's response
        content = conversation.content

        # Clean and format the response before returning it
        return self.clean_response(response=content, a_field=input[self.feature_col])

    """
    Makes predictions for a specific dataset and append the predictions to a new column.
    This method processes each row in the dataset, generates predictions using the GPT model, 
    and updates the dataset with the predicted values in the specified prediction column.
    """

    def predictions(self):

        # Start measuring time
        start_time = time.time()

        # Read the CSV dataset into a pandas DataFrame
        df = pd.read_csv(self.pre_path + self.data_set)

        # Create a copy of the original dataset (with '_original' appended to the filename)
        file_name_without_extension = os.path.splitext(os.path.basename(self.data_set))[0]

        # Rename the original file by appending '_original' to its name
        original_file_path = self.pre_path + file_name_without_extension + '_original.csv'
        if not os.path.exists(original_file_path):
            os.rename(self.pre_path + self.data_set, original_file_path)

        # Check if the prediction_column is already present in the header
        if self.prediction_column not in df.columns:
            # If not, add the column to the DataFrame with pd.NA as the initial value
            df[self.prediction_column] = pd.NA

            # # Explicitly set the column type to a nullable integer
            # df = df.astype({prediction_column: 'Int64'})

        # Save the updated DataFrame back to CSV (if a new column is added)
        if self.prediction_column not in df.columns:
            df.to_csv(self.pre_path + self.data_set, index=False)

        # Set the dtype of the reason column to object
        # df = df.astype({reason_column: 'object'})

        # Iterate over each row in the DataFrame to make predictions
        for index, row in df.iterrows():
            # Make a prediction if the value in the prediction column is missing (NaN)
            if pd.isnull(row[self.prediction_column]):
                prediction = self.gpt_prediction(input=row)
                # If the prediction fails, log the error and break the loop
                if not prediction['status']:
                    print(prediction)
                    break
                else:
                    print(prediction)
                    # If the prediction data contains a valid value, update the DataFrame
                    if prediction['data'][self.json_key] != '':
                        # Update the CSV file with the new prediction values
                        df.at[index, self.prediction_column] = prediction['data'][self.json_key]
                        # for integers only
                        # df.at[index, prediction_column] = int(prediction['data'][self.json_key])

                        # Update the CSV file with the new values
                        df.to_csv(self.pre_path + self.data_set, index=False)
                    else:
                        logging.error(
                            f"No {self.json_key} instance was found within the data for '{row[self.feature_col]}', and the "
                            f"corresponding prediction response was: {prediction}.")
                        return {"status": False,
                                "data": f"No {self.json_key} instance was found within the data for '{row[self.feature_col]}', "
                                        f"and the corresponding prediction response was: {prediction}."}

                # break
            # Add a delay of 5 seconds (reduced for testing)

        # Change the column datatype after processing all predictions to handle 2.0 ratings
        # df[prediction_column] = df[prediction_column].astype('Int64')

        # End measuring time
        end_time = time.time()  # Record the end time

        # Calculate total time taken
        time_taken = end_time - start_time
        print(f"Total time taken for predictions: {time_taken:.2f} seconds")

        # After all predictions are made, return a success message
        return {"status": True, "data": 'Prediction have successfully been'}

    """
    Upload a dataset for GPT fine-tuning via the OpenAI API.
    The dataset file will be uploaded with the purpose of fine-tuning the model.
    """

    def upload_file(self, dataset):
        # Uploads the specified dataset file to OpenAI for fine-tuning.
        upload_file = openai.File.create(
            file=open(dataset, "rb"),  # Opens the dataset file in binary read mode
            purpose='fine-tune'  # Specifies the purpose of the upload as 'fine-tune'
        )
        return upload_file

    """
      Train the GPT model either through the API or by using the OpenAI UI for fine-tuning.
      Refer to the official OpenAI fine-tuning guide for more details: 
      https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model?ref=mlq.ai
      """

    def train_gpt(self, file_id):
        # Initiates a fine-tuning job using the OpenAI API with the provided training file ID and model ("gpt-4o").
        return openai.FineTuningJob.create(training_file=file_id, model="gpt-4o")
        # Optionally, check the status of the training job by calling:
        # openai.FineTuningJob.retrieve(file_id)

    """
    Delete a Fine-Tuned GPT model
    This method deletes a specified fine-tuned GPT model using OpenAI's API. 
    """

    def delete_finetuned_model(self, model):  # ex. model = ft:gpt-3.5-turbo-0613:personal::84kHoCN
        return openai.Model.delete(model)

    """
    Cancel Fine-Tuning Job
    This method cancels an ongoing fine-tuning job using OpenAI's API.
    """

    def cancel_gpt_finetuning(self, train_id):  # ex. id = ftjob-3C5lZD1ly5HHAleLwAqT7Qt
        return openai.FineTuningJob.cancel(train_id)

    """
    Retrieve All Fine-Tuned Models and Their Status
    This method fetches a list of fine-tuned models and their details using OpenAI's API. 
    The results include information such as the model IDs, statuses, and metadata.
    """

    def get_all_finetuned_models(self):
        return openai.FineTuningJob.list(limit=10)


# TODO: Before running the script:
#  Ensure the OPENAI_API_KEY is set as an environment variable to enable access to the OpenAI API.

"""
Configure the logging module to record error messages in a file named 'error_log.txt'.
"""
logging.basicConfig(filename='../error_log.txt', level=logging.ERROR)

"""
The `params` dictionary contains configuration settings for the AI model's prediction process. 
It includes specifications for the model ID, dataset details, system and task-specific prompts, 
and parameters for prediction output, response format, and model behavior.
"""
params = {
    'model_id': 'gpt-4o-mini',  # Specifies the GPT model ID for making predictions.
    'prediction_column': 'gpt_4o_mini_prediction',  # Specifies the column where predictions will be stored.
    'pre_path': 'Datasets/',  # Specifies the base directory path where dataset files are located.
    'data_set': 'test_set.csv',  # Defines the path to the CSV dataset file.
    'prompt_array': {'json_file': 'categories.json'},  # Can be an empty array for simple projects.
    # Defines the system prompt that describes the task.
    'system': 'You are an AI assistant specializing in image classification for an e-commerce platform.',
    # Defines the prompt for the model, instructing it to make predictions and return its response in JSON format.
    # You can pass anything within brackets [example], which will be replaced during generate_prompt().
    'prompt': 'Your task is to analyze the provided image using your computer vision capabilities and classify it into the most appropriate category from a predefined list. [json] Provide your final classification in the following JSON format without explanations: {"category": "chosen_category_name"}',
    'feature_col': 'Image',  # Specifies the column in the dataset containing the text input/feature for predictions.
    'label_col': 'Category',  # Used only for creating training and validation prompt-completion pairs JSONL files.
    'json_key': 'category',  # Defines the key in the JSON response expected from the model, e.g. {"category": "value"}
    'max_tokens': 1000,  # Sets the maximum number of tokens the model should generate in its response.
    'temperature': 0,  # Sets the temperature for response variability; 0 provides the most deterministic response.
    'path_to_image': '', # Only for specific cases
}

# Resolution: 400px
# gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'
# gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'
# Phase 1 - 400px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-400:AdE0O03C'
# params['prediction_column'] = 'Phase-1-Resolution-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'
# Phase 2 - 400px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-400:AdEr0CQE'
# params['prediction_column'] = 'Phase-2-Resolution-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'
# Phase 3 - 400px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-400:AdJK4XmZ'
# params['prediction_column'] = 'Phase-3-Resolution-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'
# Phase 4 - 400px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-400:AdK0aECc'
# params['prediction_column'] = 'Phase-4-Resolution-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'

# Resolution: 200px
# gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-200'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/200/'
# gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-200'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/200/'
# Phase 1 - 200px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-200:AdKpIXNZ'
# params['prediction_column'] = 'Phase-1-Resolution-200'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/200/'
# Phase 2 - 200px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-200:Adb6bVnW'
# params['prediction_column'] = 'Phase-2-Resolution-200'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/200/'
# Phase 3 - 200px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-200:Adcn7IKB'
# params['prediction_column'] = 'Phase-3-Resolution-200'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/200/'
# Phase 4 - 200px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-200:Ade0da8i'
# params['prediction_column'] = 'Phase-4-Resolution-200'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/200/'

# Resolution: 100px
# gpt-4o base model
# params['model_id'] = 'gpt-4o'
# params['prediction_column'] = 'GPT-4o-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'
# gpt-4o-mini base model
# params['model_id'] = 'gpt-4o-mini'
# params['prediction_column'] = 'GPT-4o-mini-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'
# Phase 1 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-1-resolution-100:Adertepk'
# params['prediction_column'] = 'Phase-1-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'
# Phase 2 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-2-resolution-100:AdfgUYdN'
# params['prediction_column'] = 'Phase-2-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'
# Phase 3 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-3-resolution-100:AdiOeb9k'
# params['prediction_column'] = 'Phase-3-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'
# Phase 4 - 100px - Predictions
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-100:AdvqLQ0k'
# params['prediction_column'] = 'Phase-4-Resolution-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'

# Trained on low-resolution images, making predictions for high-resolution images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-100:AdvqLQ0k'
# params['prediction_column'] = 'Low-to-Higher-Trained-100-Prediction-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-200:Ade0da8i'
# params['prediction_column'] = 'Low-to-Higher-Trained-200-Prediction-400'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/400/'

# Trained on high-resolution images, making predictions for low-resolution images
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-200:Ade0da8i'
# params['prediction_column'] = 'High-to-Lower-Trained-200-Prediction-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'
# params['model_id'] = 'ft:gpt-4o-2024-08-06:personal:phase-4-resolution-400:AdK0aECc'
# params['prediction_column'] = 'High-to-Lower-Trained-400-Prediction-100'
# params['path_to_image'] = 'https://applied-ai.gr/projects/computer-vision/100/'


"""
Create an instance of the GPTmethods class, passing the `params` dictionary to the constructor for initialization.
"""
GPT = GPTmethods(params)

"""
Call the `predictions` method of the GPTmethods instance to make predictions on the specified dataset.
"""
GPT.predictions()


"""
Create JSONL training files
"""
# parent_path = 'https://applied-ai.gr/projects/computer-vision/'
# Original Size
# GPT.create_jsonl(data_type='train_1_original_size', data_set='train_set_1.csv', parent_path=parent_path + 'original/', save_path='FineTuning/original/')
# GPT.create_jsonl(data_type='train_2_original_size', data_set='train_set_2.csv', parent_path=parent_path + 'original/', save_path='FineTuning/original/')
# GPT.create_jsonl(data_type='train_3_original_size', data_set='train_set_3.csv', parent_path=parent_path + 'original/', save_path='FineTuning/original/')
# GPT.create_jsonl(data_type='train_4_original_size', data_set='train_set_4.csv', parent_path=parent_path + 'original/', save_path='FineTuning/original/')

# 100px Width Size
# GPT.create_jsonl(data_type='train_1_100_size', data_set='train_set_1.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# GPT.create_jsonl(data_type='train_2_100_size', data_set='train_set_2.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# GPT.create_jsonl(data_type='train_3_100_size', data_set='train_set_3.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# GPT.create_jsonl(data_type='train_4_100_size', data_set='train_set_4.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')


# 200px Width Size
# GPT.create_jsonl(data_type='train_1_200_size', data_set='train_set_1.csv', parent_path=parent_path + '200/', save_path='FineTuning/200/')
# GPT.create_jsonl(data_type='train_2_200_size', data_set='train_set_2.csv', parent_path=parent_path + '200/', save_path='FineTuning/200/')
# GPT.create_jsonl(data_type='train_3_200_size', data_set='train_set_3.csv', parent_path=parent_path + '200/', save_path='FineTuning/200/')
# GPT.create_jsonl(data_type='train_4_200_size', data_set='train_set_4.csv', parent_path=parent_path + '200/', save_path='FineTuning/200/')


# 300px Width Size
# GPT.create_jsonl(data_type='train_1_300_size', data_set='train_set_1.csv', parent_path=parent_path + '300/', save_path='FineTuning/300/')
# GPT.create_jsonl(data_type='train_2_300_size', data_set='train_set_2.csv', parent_path=parent_path + '300/', save_path='FineTuning/300/')
# GPT.create_jsonl(data_type='train_3_300_size', data_set='train_set_3.csv', parent_path=parent_path + '300/', save_path='FineTuning/300/')
# GPT.create_jsonl(data_type='train_4_300_size', data_set='train_set_4.csv', parent_path=parent_path + '300/', save_path='FineTuning/300/')


# 400px Width Size
# GPT.create_jsonl(data_type='train_1_400_size', data_set='train_set_1.csv', parent_path=parent_path + '400/', save_path='FineTuning/400/')
# GPT.create_jsonl(data_type='train_2_400_size', data_set='train_set_2.csv', parent_path=parent_path + '400/', save_path='FineTuning/400/')
# GPT.create_jsonl(data_type='train_3_400_size', data_set='train_set_3.csv', parent_path=parent_path + '400/', save_path='FineTuning/400/')
# GPT.create_jsonl(data_type='train_4_400_size', data_set='train_set_4.csv', parent_path=parent_path + '400/', save_path='FineTuning/400/')

"""
Create JSONL validation files
"""
# Original Size
# GPT.create_jsonl(data_type='validation_original_size', data_set='validation_set.csv', parent_path=parent_path + 'original/', save_path='FineTuning/original/')
# 100px Width Size
# GPT.create_jsonl(data_type='validation_100_size', data_set='validation_set.csv', parent_path=parent_path + '100/', save_path='FineTuning/100/')
# 200px Width Size
# GPT.create_jsonl(data_type='validation_200_size', data_set='validation_set.csv', parent_path=parent_path + '200/', save_path='FineTuning/200/')
# 300px Width Size
# GPT.create_jsonl(data_type='validation_300_size', data_set='validation_set.csv', parent_path=parent_path + '300/', save_path='FineTuning/300/')
# 400px Width Size
# GPT.create_jsonl(data_type='validation_400_size', data_set='validation_set.csv', parent_path=parent_path + '400/', save_path='FineTuning/400/')
