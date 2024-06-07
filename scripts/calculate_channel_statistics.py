import argparse
import os
import cv2
import numpy as np
import tqdm
import json

def update_statistics(statistics, image):
    """
    Update the statistics with the pixel values of the given image.

    Args:
    statistics (dict): The dictionary containing the statistics.
    image (np.ndarray): The image to update the statistics with.

    Returns:
    dict: The updated statistics.
    """
    num_channels = image.shape[2]

    statistics['means_per_image_full'].append(np.mean(np.mean(np.array(image), axis=1), axis=0))

    variances = np.zeros([num_channels])
    for i in range(num_channels):
        variances[i] = np.var(image[:, :, i])
    statistics['variances_per_image_full'].append(variances)

    return statistics

def disp_and_save_results(statistics, sensor='', save_as_json=False,
                          save_path=None, empty_list=False, no_file_list=False):
    """
    Display the results of the statistics.

    Args:
    statistics (dict): The dictionary containing the statistics.
    sensor (str): The name of the sensor.
    save_as_json (bool): Whether to save the results as a JSON file.
    empty_list (bool): Whether to display the list of images without points.
    no_file_list (bool): Whether to display the list of images without files.
    save_path (str): The path to save the JSON file.


    Returns:
    None
    """
    resutls = {}
    mean_dataset = np.mean(statistics['means_per_image_full'], axis=0)
    variance_of_means = np.var(statistics['means_per_image_full'],  axis=0)
    mean_of_variances = np.mean(statistics['variances_per_image_full'], axis=0)
    variance_dataset = variance_of_means + mean_of_variances
    std_dataset = np.sqrt(variance_dataset)
    for i in range(len(variance_dataset)):
        if variance_dataset[i] == 0.:
            variance_dataset[i] == 1.
    resutls['mean'] = mean_dataset.tolist()
    resutls['std'] = std_dataset.tolist()

    print(f'{sensor} mean per channel: ', mean_dataset)
    print(f'{sensor} std per channel: ', std_dataset)

    if empty_list:
        resutls['no_points_list'] = statistics['no_points_list']
        print(f'number of images without {sensor} points:', len(statistics['no_points_list']))
    if no_file_list:
        resutls['no_file_list'] = statistics['no_file_list']
        print(f'number of images without {sensor} file:', len(statistics['no_file_list']))

    if save_as_json:
        assert save_path is not None, 'Please provide a save path'
        file_name = os.path.join(save_path, f'channel_statistics_{sensor}.json')
        with open(file_name, 'w') as file:
            json.dump(resutls, file, indent=4)

def process_image(statistics, image_path, png_rescale_factors):
    """
    Process an image and update the statistics.

    Args:
    statistics (dict): The dictionary containing the statistics.
    image_path (str): The path to the image.
    png_rescale_factors (dict): The scale and shift factors for each sensor for loading the png files.

    Returns:
    dict: The updated statistics.
    """
    if image_path.endswith('.npy'):
        image = np.load(image_path)
    elif image_path.endswith('.npz'):
        image = np.load(image_path)['arr_0']
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = image.astype(np.float32)
        image = image / png_rescale_factors['scale_factor']
        image = image - png_rescale_factors['shift_factor']

    if image is not None and image.shape[2] == 3:
        statistics = update_statistics(statistics, image)
    else:
        raise Exception(f'[ERROR] Image needs to have 3 channels but is shape {image.shape}')

    return statistics

def calculate_and_display_statistics(projection_path, sensor_name, png_rescale_factors, save_as_json):
    """
    Calculate the mean and standard deviation of the pixel values for each channel in the images
    in the given directory.

    Args:
    projection_path (str): The root folder where the images are located.
    sensor_name (str): The name of the sensor.
    png_rescale_factors (dict): The scale and shift factors for each sensor for loading the png files.

    Returns:
    None
    """
    # Initialize statistics
    statistics = {
        'means_per_image_full': [],
        'variances_per_image_full': [],
        'no_file_list': [],
    }

    # Traverse through the directory and process images
    sensor_root = os.path.join(projection_path, sensor_name)
    files_list = []
    for root, dirs, files in os.walk(sensor_root):
        if not len(files) == 0:
            for file in files:
                files_list.append(os.path.join(root, file))

    for file in tqdm.tqdm(files_list, desc=f'Processing {sensor_name} images'):
        if file.endswith('.png') or file.endswith('.npy') or file.endswith('.npz'):
            # Load the image
            image_path = os.path.join(root, file)
            statistics = process_image(statistics, image_path, png_rescale_factors)
        else:
            raise Exception(f'[ERROR] Image needs to be a png or npy file but is {file}')

    # Calculate the mean and standard deviation per channel for all images
    # Use your disp_statistics_results function to display the results
    disp_and_save_results(statistics, sensor=sensor_name, save_as_json=save_as_json, save_path=projection_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate normalization parameters for projected images.')
    parser.add_argument('--muses_root', type=str, required=True, help='Root folder where your images are located')
    parser.add_argument('--lidar', action='store_true', help='Process lidar data')
    parser.add_argument('--radar', action='store_true', help='Process radar data')
    parser.add_argument('--event_camera', action='store_true', help='Process event camera data')
    parser.add_argument('--projection_folder', type=str, required=False, default='projected_to_rgb',
                        help='Folder where the projected images are located')
    parser.add_argument('--save_as_json', action='store_true',
                        help='Save the statistics as a JSON file in the projection folder.')
    args = parser.parse_args()

    # Define the scale and shift factors for each sensor
    png_rescale_factors_per_mod = dict(lidar=dict(
                        scale_factor=150.,
                        shift_factor=100.),
                    radar=dict(
                        scale_factor=150.,
                        shift_factor=100.),
                    event_camera=dict(
                        scale_factor=1.,
                        shift_factor=0.))

    projection_path = os.path.join(args.muses_root, args.projection_folder)

    if not args.event_camera and not args.lidar and not args.radar:
        raise ValueError("At least one sensor must be selected for processing. "
                         "Add any combination of --event_camera, --lidar, or --radar.")

    sensor_names = []
    if args.lidar:
        sensor_names.append('lidar')
    if args.radar:
        sensor_names.append('radar')
    if args.event_camera:
        sensor_names.append('event_camera')

    for sensor_name in sensor_names:
        if os.path.exists(os.path.join(projection_path, sensor_name)):
            print(f'[INFO] Calculating normalization parameters for all {sensor_name} images in {projection_path}')
            calculate_and_display_statistics(projection_path, sensor_name, png_rescale_factors_per_mod[sensor_name],
                                             args.save_as_json)
        else:
            print(f'[WARNING] {sensor_name} folder does not exist in {projection_path}')