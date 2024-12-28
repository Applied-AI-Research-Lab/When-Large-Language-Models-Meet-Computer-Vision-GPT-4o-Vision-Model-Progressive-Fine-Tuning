import pandas as pd

class ResNetDataProcessor:
    def __init__(self, file_path, base_url):
        """
        Initialize the processor with the CSV file path and base URL.
        :param file_path: Path to the CSV file
        :param base_url: Base URL for constructing image URLs
        """
        self.file_path = file_path
        self.base_url = base_url
        self.data = None

    def load_data(self):
        """
        Load the data from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"File not found at {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def update_image_urls(self):
        """
        Update the 'Image' column with new URLs.
        """
        if self.data is not None:
            try:
                self.data['Image'] = self.data['id'].astype(str).apply(lambda x: f"{self.base_url}{x}.jpg")
                print("Image URLs updated successfully.")
            except KeyError:
                print("Column 'id' not found in the data.")
            except Exception as e:
                print(f"Error updating image URLs: {e}")
        else:
            print("Data not loaded. Cannot update image URLs.")

    def save_data(self, output_path):
        """
        Save the updated data to a new CSV file.
        :param output_path: Path to save the updated CSV file
        """
        if self.data is not None:
            try:
                self.data.to_csv(output_path, index=False)
                print(f"Updated data saved to {output_path}")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("Data not loaded. Cannot save data.")


# Creating train_set with 100px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/train_set.csv', 'https://applied-ai.gr/projects/computer-vision/100/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/train_set_100.csv')

# Creating train_set with 200px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/train_set.csv', 'https://applied-ai.gr/projects/computer-vision/200/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/train_set_200.csv')

# Creating train_set with 400px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/train_set.csv', 'https://applied-ai.gr/projects/computer-vision/400/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/train_set_400.csv')

# Creating validation_set with 100px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/validation_set.csv', 'https://applied-ai.gr/projects/computer-vision/100/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/validation_set_100.csv')

# Creating validation_set with 200px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/validation_set.csv', 'https://applied-ai.gr/projects/computer-vision/200/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/validation_set_200.csv')

# Creating validation_set with 400px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/validation_set.csv', 'https://applied-ai.gr/projects/computer-vision/400/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/validation_set_400.csv')

# Creating test_set with 100px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/test_set.csv', 'https://applied-ai.gr/projects/computer-vision/100/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/test_set_100.csv')

# Creating test_set with 200px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/test_set.csv', 'https://applied-ai.gr/projects/computer-vision/200/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/test_set_200.csv')

# Creating test_set with 400px images urls
# processor = ResNetDataProcessor('../Datasets/ResNetFineTuning/test_set.csv', 'https://applied-ai.gr/projects/computer-vision/400/')
# processor.load_data()
# processor.update_image_urls()
# processor.save_data('../Datasets/ResNetFineTuning/test_set_400.csv')