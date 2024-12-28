import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import time
import re


class DatasetProcessor:
    """
    A class to process datasets by downloading images, creating thumbnails, and maintaining progress.

    Attributes:
        dataset_path (str): Path to the dataset file.
        batch_size (int): Number of images to process in a batch.
        stop_after (int): Maximum number of images to process before stopping.
        progress_file (str): Path to the progress file.
        images_folder (str): Folder to save original images.
        thumbnail_folders (dict): Folders to save thumbnails of varying widths.
    """

    def __init__(self, dataset_path, batch_size=10, stop_after=50, progress_file="progress.txt"):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.stop_after = stop_after
        self.progress_file = os.path.join('../Datasets', progress_file)
        self.images_folder = '../Images/Original/'
        self.thumbnail_folders = {
            100: '../Images/100/',
            200: '../Images/200/',
            300: '../Images/300/',
            400: '../Images/400/',
        }

        os.makedirs(self.images_folder, exist_ok=True)
        for folder in self.thumbnail_folders.values():
            os.makedirs(folder, exist_ok=True)

    def rename_original_file(self):
        """
        Rename the original dataset file to create a backup.

        Returns:
            str: Path to the renamed original file.
        """
        file_name_without_extension = os.path.splitext(os.path.basename(self.dataset_path))[0]
        original_file_path = f'../Datasets/{file_name_without_extension}_original3.csv'

        if not os.path.exists(original_file_path):
            os.rename(f'../Datasets/{self.dataset_path}', original_file_path)
            print(f"Backup of original file created: {original_file_path}")
        else:
            print("Backup already exists. Skipping backup.")

        return original_file_path

    def save_to_file(self, df):
        df.to_csv(f'../Datasets/{self.dataset_path}', index=False, encoding='utf-8')

    def is_image_live(self, url):
        """
        Check if an image URL is live and accessible.

        Args:
            url (str): The image URL to check.

        Returns:
            bool: True if the image is accessible, False otherwise.
        """
        try:
            response = requests.head(url, allow_redirects=True, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def remove_image_modifiers(self, url):
        """
        Remove image modifiers (e.g., ._AC_UL320_) from an Amazon image URL.
        This function keeps the base image name and appends the extension at the end.

        Args:
            url (str): The Amazon image URL.

        Returns:
            str: The cleaned URL without modifiers.
        """

        cleaned_url = re.sub(r'(_[^.]+)(\.[^.]+)$', r'\2', url)
        cleaned_url = re.sub(r'\.+', '.', cleaned_url)

        return cleaned_url

    def process_image_column(self, image_urls):
        """
        Process a column containing image URLs to validate and clean the URLs.

        Args:
            image_urls (str): Pipe-separated image URLs.

        Returns:
            str or None: The first valid and live image URL, or None if none are valid.
        """
        # Split the image URLs by '|'
        image_list = str(image_urls).split('|')

        for image_url in image_list:
            # Clean the URL by removing the Amazon-specific modifiers
            cleaned_url = self.remove_image_modifiers(image_url.strip())  # Ensure no leading/trailing whitespace

            # Test if the cleaned URL is live
            if self.is_image_live(cleaned_url):
                return cleaned_url  # Return the first valid image URL

        # If no valid URL is found, return None
        print("No valid image found.")  # Optional print for debugging
        return None

    def download_image(self, url, save_path):
        """
        Download an image from a URL.

        Args:
            url (str): The URL of the image.
            save_path (str): Path to save the downloaded image.

        Returns:
            bool: True if the download is successful, False otherwise.
        """
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return True
        except requests.exceptions.RequestException:
            print(f"Failed to download image: {url}")
        return False

    def create_thumbnail(self, image_path, thumbnail_path, max_width, quality=100):
        """
        Create a thumbnail for an image.

        Args:
            image_path (str): Path to the original image.
            thumbnail_path (str): Path to save the thumbnail.
            max_width (int): Maximum width for the thumbnail.
            quality (int): The quality of the saved thumbnail image (default 95).
        """
        try:
            with Image.open(image_path) as img:
                # Create the thumbnail
                img.thumbnail((max_width, max_width * img.height // img.width))

                # Save the thumbnail with high quality (default 100)
                img.save(thumbnail_path, "JPEG", quality=quality)

        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")

    def read_progress(self):
        """
        Read the last processed index from the progress file.

        Returns:
            int: The last processed index, or 0 if no progress file exists.
        """
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return int(f.read().strip())
        return 0

    def write_progress(self, last_index):
        """
        Write the last processed index to the progress file.

        Args:
            last_index (int): The last processed index to save.
        """
        with open(self.progress_file, 'w') as f:
            f.write(str(last_index))

    def process_images(self):
        """
        Process images from the dataset:
        - Validates image URLs.
        - Downloads images to the 'Original' folder.
        - Creates thumbnails for multiple resolutions.
        - Tracks progress and saves updates.

        Returns:
            pandas.DataFrame: The updated dataset.
        """
        try:
            df = pd.read_csv(f'../Datasets/{self.dataset_path}')
        except FileNotFoundError:
            print(f"Dataset file not found: ../Datasets/{self.dataset_path}")
            return
        except pd.errors.EmptyDataError:
            print("Dataset is empty.")
            return

        if 'Image' not in df.columns or 'id' not in df.columns:
            print("The dataset must contain 'Image' and 'id' columns.")
            return

        self.rename_original_file()

        start_index = self.read_progress()
        rows_to_drop = []
        processed_count = 0

        for idx in range(start_index, len(df)):
            row = df.iloc[idx]

            # Get the valid image URL
            valid_image = self.process_image_column(row['Image'])

            if valid_image:
                image_id = row['id']
                image_path = os.path.join(self.images_folder, f"{image_id}.jpg")

                # Download image to Original folder
                if self.download_image(valid_image, image_path):
                    # Create thumbnails for each resolution
                    for max_width, folder in self.thumbnail_folders.items():
                        thumbnail_path = os.path.join(folder, f"{image_id}.jpg")
                        self.create_thumbnail(image_path, thumbnail_path, max_width)

                    # Update the 'Image' column in the DataFrame with the valid image URL
                    df.at[idx, 'Image'] = valid_image
                else:
                    print(f"Can not download image for row {idx} {valid_image}")
                    # rows_to_drop.append(idx)
            else:
                print(f"No valid image URL for row {idx} {valid_image}")
                # rows_to_drop.append(idx)

            processed_count += 1

            # Save progress in batches
            if processed_count % self.batch_size == 0 or processed_count >= self.stop_after:
                self.write_progress(idx + 1)
                if rows_to_drop:
                    df.drop(rows_to_drop, inplace=True)
                    rows_to_drop.clear()
                self.save_to_file(df)
                print(f"Processed {processed_count} images. Saving progress.")
                time.sleep(5)

            if processed_count >= self.stop_after:
                print("Reached the stop limit. Saving progress and stopping.")
                break

        if rows_to_drop:
            df.drop(rows_to_drop, inplace=True)

        self.save_to_file(df)

        if processed_count < self.stop_after and os.path.exists(self.progress_file):
            os.remove(self.progress_file)

        print("Processing completed.")
        return df

    def count_files_in_folders(self):
        """
        Count the number of files in the Images/Original folder and each Thumbnails folder.

        Returns:
            dict: A dictionary with folder names as keys and file counts as values.
        """
        # Count files in the Original folder
        images_count = len(
            [f for f in os.listdir(self.images_folder) if os.path.isfile(os.path.join(self.images_folder, f))])

        print(f"Number of files in '{self.images_folder}': {images_count}")

        # Count files in each thumbnail folder
        thumbnail_counts = {}
        for max_width, folder in self.thumbnail_folders.items():
            count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            thumbnail_counts[max_width] = count
            print(f"Number of files in '{folder}': {count}")

        # Combine results into a single dictionary
        counts = {'Original': images_count,
                  **{f"Thumbnail_{width}": count for width, count in thumbnail_counts.items()}}
        return counts


# dataset_processor = DatasetProcessor('train_set.csv', batch_size=10, stop_after=50)
# dataset_processor = DatasetProcessor('test_set.csv', batch_size=10, stop_after=50)
dataset_processor = DatasetProcessor('validation_set.csv', batch_size=10, stop_after=50)


dataset_processor.process_images()
dataset_processor.count_files_in_folders()
