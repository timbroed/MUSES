import argparse
import copy
import json
import logging
import numpy as np
import os
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
from matplotlib import colormaps as cm
from tkinter import messagebox

from processing.event_camera_processing import load_points_in_image_event_camera, \
    render
from processing.lidar_processing import load_lidar_data
from processing.radar_processing import load_radar_as_pcd
from processing.utils import load_muses_calibration_data, \
    filter_and_project_pcd_to_image, motion_compensate_pcd

DELETE_TOKEN = "all"


class ImageViewer:
    """
    ImageViewer class represents an image viewer application.

    Args:
        data_dict (dict): A dictionary containing the data for each image.
        complete_data_dict (dict): A dictionary containing the complete data for each image.
        data_root (str): The root directory where the muses dataset is located.
        image_idx (int, optional): The index of the initial image to display. Default is 1.
        scale_factor (float, optional): The scale factor to resize the images. Default is 0.8.
        motion_compensation (bool, optional): A flag to enable motion compensation. Default is True.
        radar_threshold (int, optional): The threshold for the radar data. Default is 0.

    Attributes:
        loaded_image (PIL.Image): The loaded image.
        data_root (str): The root directory where the image and data files are located.
        complete_data_dict (dict): A dictionary containing the complete data for each image.
        motion_compensation (bool): A flag to enable motion compensation.
        radar_threshold (int): The threshold for the radar data.
        root (tk.Tk): The main window of the application.
        results (list): A list to store the results.
        index (int): The index of the current image.
        total_reference_images (int): The total number of reference images.
        data_dict (dict): A dictionary containing the data for each image.
        total_images (int): The total number of images.
        image_id_loaded (int): The ID of the loaded image.
        image_names (list): A list of image names.
        last_event (str): The last event.
        scale_factor (float): The scale factor to resize the images.
        canvas_size (tuple): The size of the canvas.
        canvas (tk.Canvas): The canvas to display the images.
        status_label (tk.Label): The label to display the status.
        split_label (tk.Label): The label to display the split.
        split_combobox (ttk.Combobox): The Combobox to select the split.
        weather_combobox (ttk.Combobox): The Combobox to select the weather.
        time_of_day_combobox (ttk.Combobox): The Combobox to select the time of day.
        motion_compensation_label (tk.Label): The label to display the motion compensation.
        motion_compensation_combobox (ttk.Combobox): The Combobox to select the motion compensation.
        image_id_label (tk.Label): The label to display the image ID.
        image_id_entry (tk.Entry): The Entry to enter the image ID.
        apply_button (tk.Button): The Button to apply the changes.
        calib_data (dict): The calibration data.
    """
    def __init__(self,
                 data_dict,
                 complete_data_dict,
                 data_root,
                 image_idx=1,
                 scale_factor=0.8,
                 motion_compensation=False,
                 radar_threshold=0):
        """
        Initialize the ImageViewer object.

        Args:
            data_dict (dict): A dictionary containing the data for each image.
            complete_data_dict (dict): A dictionary containing the complete data for each image.
            data_root (str): The root directory where the image and data files are located.
            image_idx (int, optional): The index of the initial image to display. Default is 1.
            scale_factor (float, optional): The scale factor to resize the images. Default is 0.8.
            motion_compensation (bool, optional): A flag to enable motion compensation. Default is True.
            radar_threshold (int, optional): The threshold for the radar data. Default is 0.
        """
        self.loaded_image = None
        self.data_root = data_root
        self.complete_data_dict = complete_data_dict
        self.motion_compensation = motion_compensation
        if self.motion_compensation:
            if not os.path.exists(os.path.join(self.data_root, 'gnss')):
                self.show_missing_modality_message("GNSS", is_error=True)
        self.radar_threshold = radar_threshold if radar_threshold is not None else 0
        self.root = tk.Tk()
        self.root.title("MUSES Image Viewer")
        self.results = []
        self.index = image_idx - 1
        self.total_reference_images = 0
        self.data_dict = data_dict
        self.total_images = len(data_dict)
        self.image_id_loaded = -1
        self.image_names = list(data_dict.keys())
        self.last_event = None
        self.scale_factor = scale_factor
        self.canvas_size = (int(1920 * self.scale_factor), int(1080 * self.scale_factor))
        self.canvas = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1])
        self.canvas.grid(row=0, column=0, columnspan=11)
        self.status_label = tk.Label(self.root, text="")
        self.status_label.grid(row=1, column=0, columnspan=11)
        self.load_image()
        self.calib_data = load_muses_calibration_data(data_root)
        self.root.bind('<KeyPress>', self.handle_keypress)

        # Create a Combobox for the split
        self.split_label = tk.Label(self.root, text="Split:")
        self.split_label.grid(row=2, column=0)
        self.split_combobox = ttk.Combobox(self.root, values=["all", "train", "val", "test"])
        self.split_combobox.set(args.split if args.split else "all")
        self.split_combobox.grid(row=2, column=1)
        self.split_label = tk.Label(self.root, text="Weather:")
        self.split_label.grid(row=2, column=2)
        self.weather_combobox = ttk.Combobox(self.root, values=["all", "clear", "rain", "snow", "fog"])
        self.weather_combobox.set(args.weather if args.weather else "all")
        self.weather_combobox.grid(row=2, column=3)
        self.split_label = tk.Label(self.root, text="Time of Day:")
        self.split_label.grid(row=2, column=4)
        self.time_of_day_combobox = ttk.Combobox(self.root, values=["all", "day", "night"])
        self.time_of_day_combobox.set(args.time_of_day if args.time_of_day else "all")
        self.time_of_day_combobox.grid(row=2, column=5)
        self.motion_compensation_label = tk.Label(self.root, text="Motion Compensation:")
        self.motion_compensation_label.grid(row=2, column=6)
        self.motion_compensation_combobox = ttk.Combobox(self.root, values=["True", "False"])
        self.motion_compensation_combobox.set("True" if self.motion_compensation==True else "False")
        self.motion_compensation_combobox.grid(row=2, column=7)
        self.image_id_label = tk.Label(self.root, text="Scene Index:")
        self.image_id_label.grid(row=2, column=8)
        self.image_id_entry = tk.Entry(self.root)
        self.image_id_entry.insert(0, "0")
        self.image_id_entry.grid(row=2, column=9)

        # Create a Button to apply the split
        self.apply_button = tk.Button(self.root, text="Apply Changes", command=self.apply_button)
        self.apply_button.grid(row=2, column=10)

        # display Instructions
        self.show_message("Instructions", "Press [Number] to overlay the corresponding data on the frame image.\n\n"
                                          "Press a second time, to undo the overlay.\n\n"
                                          "Press [Left] or [Right] to navigate through the images.\n\n"
                                          "Press [Up] or [Down] to skip 100 images.\n\n"
                                        "1: Camera, 2: Lidar, 3: Event Camera, 4: Radar, 5: Reference Image, 6: Panoptic GT, 7: Uncertainty GT")

    def apply_button(self):
        # Get the split from the Combobox
        args.split = self.split_combobox.get()
        if args.split == "all":
            self.data_dict = self.complete_data_dict
        else:
            self.data_dict = {key: value for key, value in self.complete_data_dict.items() if value['split'] == args.split}

        args.time_of_day = self.time_of_day_combobox.get()
        if args.time_of_day in ["day", "night"]:
            self.data_dict = {key: value for key, value in self.data_dict.items() if value['time_of_day'] == args.time_of_day}

        args.weather = self.weather_combobox.get()
        if args.weather in ["clear", "rain", "snow", "fog"]:
            self.data_dict = {key: value for key, value in self.data_dict.items() if value['weather'] == args.weather}

        args.motion_compensation = self.motion_compensation_combobox.get()
        self.motion_compensation = args.motion_compensation == "True"

        self.total_images = len(self.data_dict)
        self.image_id_loaded = -1
        self.image_names = list(self.data_dict.keys())
        self.index = 0

        image_id = self.image_id_entry.get()
        try:
            image_id = int(image_id)
            if image_id > self.total_images:
                self.show_message("Error", f"Image ID {image_id} is greater than the total number of images.")
            else:
                self.index = image_id - 1
        except ValueError:
            self.show_message("Error", f"Image ID {image_id} is not a valid integer.")

        # Reload the image
        self.load_image()

    def move_index(self, increment):
        """
        Move the index by the given increment.

        Args:
            increment (int): The increment to move the index.
        """
        self.index += increment
        if self.index < 0:
            self.show_message("Start", "You are already at the first image.")
            self.index -= increment

        elif self.index >= len(self.image_names):
            self.show_message("End", "End of the list reached.")
            self.index -= increment

    def previous_image(self):
        # Function to go to the previous image
        self.move_index(-1)
        self.load_image()

    def next_image(self):
        # Function to go to the next image
        self.move_index(1)
        self.load_image()

    def get_image_path(self, image_name, key):
        return os.path.join(self.data_root, self.data_dict[image_name][key])

    def handle_keypress(self, event):
        """
        Handle the keypress event.

        Args:
            event (tk.Event): The keypress event.
        """
        if self.last_event == event.keysym:
            self.load_image('path_to_frame_camera')
            self.last_event = None
        elif event.keysym == 'Right':
            self.next_image()
            self.last_event = None
        elif event.keysym == 'Left':
            self.previous_image()
            self.last_event = None
        elif event.keysym == 'Up':
            self.move_index(100)
            self.load_image()
            self.last_event = None
        elif event.keysym == 'Down':
            self.move_index(-100)
            self.load_image()
            self.last_event = None
        elif event.keysym == '1':
            self.load_image('path_to_frame_camera')
            self.last_event = event.keysym
        elif event.keysym == '2':
            self.load_lidar_data()
            self.last_event = event.keysym
        elif event.keysym == '3':
            self.load_event_data()
            self.last_event = event.keysym
        elif event.keysym == '4':
            self.load_radar_data()
            self.last_event = event.keysym
        elif event.keysym == '5':
            if 'path_to_reference_frame' in self.data_dict[self.image_names[self.index]].keys():
                if not os.path.exists(os.path.join(self.data_root, 'reference_frame')):
                    self.show_missing_modality_message("Reference Frame")
                else:
                    self.load_image('path_to_reference_frame')
                    self.last_event = event.keysym
            else:
                self.show_message("INFO", f"This scene already has clear and daytime conditions. "
                                          f"No reference image available for: "
                                          f"{self.image_names[self.index]}")
        elif event.keysym == '6':
            if 'path_to_gt_panoptic' in self.data_dict[self.image_names[self.index]].keys():
                if not os.path.exists(os.path.join(self.data_root, 'gt_panoptic')):
                    self.show_missing_modality_message("GT Panoptic")
                else:
                    self.load_image('path_to_gt_panoptic')
                    self.last_event = event.keysym
            else:
                self.show_message("INFO", f"This scene is part of the test set. "
                                          f"No panoptic annotation available for: "
                                          f"{self.image_names[self.index]}")
        elif event.keysym == '7':
            if 'path_to_gt_uncertainty' in self.data_dict[self.image_names[self.index]].keys():
                if not os.path.exists(os.path.join(self.data_root, 'gt_uncertainty')):
                    self.show_missing_modality_message("GT Uncertainty")
                else:
                    self.load_image('path_to_gt_uncertainty')
                    self.last_event = event.keysym
            else:
                self.show_message("INFO", f"This scene is part of the test set. "
                                          f"No uncertainty map available for: "
                                          f"{self.image_names[self.index]}")

    def clear_canvas(self):
        # A function to delete elements from the canvas and clear image labels
        self.canvas.delete(DELETE_TOKEN)

    def load_image(self, key='path_to_frame_camera'):
        # A function to load images
        self.clear_canvas()
        image_name = self.image_names[self.index]
        image_path = self.get_image_path(image_name, key)
        self.check_and_display_image(image_path)

    def check_and_display_image(self, image_path):
        # Check if the image path exists and then display the image
        if os.path.exists(image_path):
            self.load_and_display_image(image_path)

        else:
            self.show_message("Error", f"File does not exist: {image_path}")

    def load_and_display_image(self, image_path):
        original_image = Image.open(image_path)
        self.loaded_image = original_image.resize(self.canvas_size, resample=Image.NEAREST)
        self.display_image(self.loaded_image)

    def display_image(self, image):
        self.clear_canvas()
        original_photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=original_photo, anchor=tk.NW, tags="image")
        self.canvas.image = original_photo
        self.update_label_status()

    def update_label_status(self, pixel_value=None):
        label = f"Image: {self.index + 1} / {self.total_images} ({self.image_names[self.index]})"
        image_data = self.data_dict[self.image_names[self.index]]

        label += (f"\nSplit: {image_data['split']}     Time of day: {image_data['time_of_day']}     "
                  f"Weather: {image_data['weather']}     Motion Compensation: {self.motion_compensation}     "
                  f"Radar Threshold: {self.radar_threshold}")
        label += "\n Legend:   1: Camera   2: Lidar   3: Event Camera   4: Radar   5: Reference Image   6: Panoptic GT   7: Uncertainty GT"

        self.status_label.config(text=label)

    def start_main_loop(self):
        self.root.mainloop()

    def show_message(self, title, message):
        messagebox.showinfo(title, message)

    def show_missing_modality_message(self, modality, is_error=False):
        message = (f"The {modality} data is not available in the muses root folder. "
                   f"Please download the data from https://muses.vision.ee.ethz.ch/download "
                   f"and unzip the sensors data before proceeding.")
        if is_error:
            raise FileNotFoundError(message)
        else:
            messagebox.showinfo("Missing Modality", message)

    def load_lidar_data(self):
        """
        Load the lidar data and display it on the image.
        """
        if not os.path.exists(os.path.join(self.data_root, 'lidar')):
            self.show_missing_modality_message("Lidar")
            return
        sample_data = self.data_dict[self.image_names[self.index]]
        lidar_path = os.path.join(self.data_root, sample_data['path_to_lidar'])
        pcd_points = load_lidar_data(lidar_path)

        # Check if motion compensation is enabled
        if self.motion_compensation:
            lidar2gnss = self.calib_data["extrinsics"]["lidar2gnss"]
            image_data = self.data_dict[self.image_names[self.index]]
            pcd_points = motion_compensate_pcd(self.data_root, image_data, pcd_points, lidar2gnss, ts_channel_num=5)

        K_rgb = self.calib_data["intrinsics"]["rgb"]["K"]
        lidar2rgb = self.calib_data["extrinsics"]["lidar2rgb"]
        uv_img_cords_filtered, filtered_pcd_points = filter_and_project_pcd_to_image(pcd_points, lidar2rgb, K_rgb,
                                                                                     self.canvas_size)
        colors = np.linalg.norm(filtered_pcd_points[:, 0:3], axis=1).astype(np.uint16)
        colors_clipped = np.clip(colors, a_min=None, a_max=200)
        rgb_colors = cm.get_cmap('hsv')(colors_clipped / 200)[:, :3] * 255

        # Map and store the channels in the image
        image = np.array(self.loaded_image)
        image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), :] = rgb_colors
        self.display_image(Image.fromarray(image))

    def load_radar_data(self):
        """
        Load the radar data and display it on the image.
        """
        if not os.path.exists(os.path.join(self.data_root, 'radar')):
            self.show_missing_modality_message("Radar")
            return
        sample_data = self.data_dict[self.image_names[self.index]]
        radar_path = os.path.join(self.data_root, sample_data['path_to_radar'])
        pcd_points = load_radar_as_pcd(radar_path, intensity_threshold=self.radar_threshold, image_fov_only=True)

        if self.motion_compensation:
            radar2gnss = np.array(self.calib_data["extrinsics"]["radar2gnss"])
            image_data = self.data_dict[self.image_names[self.index]]
            pcd_points = motion_compensate_pcd(self.data_root, image_data, pcd_points, radar2gnss, ts_channel_num=4)

        K_rgb = self.calib_data["intrinsics"]["rgb"]["K"]
        radar2rgb = self.calib_data["extrinsics"]["radar2rgb"]
        uv_img_cords_filtered, filtered_pcd_points = filter_and_project_pcd_to_image(pcd_points, radar2rgb, K_rgb,
                                                                                     self.canvas_size, max_distance=150)
        colors = filtered_pcd_points[:, 3]
        colors = np.clip(colors, a_min=None, a_max=150)
        colors /= colors.max()
        rgb_colors = cm.get_cmap('hsv')(colors)[:, :3] * 255

        # Map and store the channels in the image
        image = np.array(self.loaded_image)
        image[uv_img_cords_filtered[1, :].astype(int), uv_img_cords_filtered[0, :].astype(int), :] = rgb_colors
        self.display_image(Image.fromarray(image))


    def load_event_data(self):
        """
        Load the event data and display it on the image.
        """
        if not os.path.exists(os.path.join(self.data_root, 'event_camera')):
            self.show_missing_modality_message("Event Camera")
            return
        sample_data = self.data_dict[self.image_names[self.index]]
        event_path = os.path.join(self.data_root, sample_data['path_to_event_camera'])
        rgb_width, rgb_height = self.canvas_size
        x, y, p = load_points_in_image_event_camera(event_path, self.calib_data, rgb_width, rgb_height)
        overlay_image = render(x, y, p, rgb_height, rgb_width)

        loaded_image = np.array(self.loaded_image)
        final_image = np.where(overlay_image != 0, overlay_image, loaded_image)
        self.display_image(Image.fromarray(final_image))

# initialize the logger
logging.basicConfig(level=logging.INFO)

def visualization(args):
    """
    Visualize the image data.

    Args:
        args: The command line arguments.
    """
    # Create the full path to the JSON file
    json_file_path = os.path.join(args.muses_root, 'meta.json')

    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File '{json_file_path}' does not exist.")

    # Open and read the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    complete_data_dict = copy.deepcopy(data)

    if args.split:
        logging.info(f"Filtering data by split: {args.split}")
        data = {key: value for key, value in data.items() if value['split'] == args.split}
    if args.time_of_day:
        logging.info(f"Filtering data by time of day: {args.time_of_day}")
        data = {key: value for key, value in data.items() if value['time_of_day'] == args.time_of_day}
    if args.weather:
        logging.info(f"Filtering data by weather: {args.weather}")
        data = {key: value for key, value in data.items() if value['weather'] == args.weather}

    viewer = ImageViewer(data, complete_data_dict, args.muses_root, image_idx=args.image, scale_factor=args.canvas_scale,
                         motion_compensation=args.motion_compensation, radar_threshold=args.radar_threshold)
    viewer.start_main_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image based data visualization tool')
    parser.add_argument('--muses_root', default='data/muses/', help='Root path for data')
    parser.add_argument('--motion_compensation', action='store_true',
                        help='Enable motion compensation  (default: False)')
    parser.add_argument('--split', default=None,
                        help='Selected dataset split: train, val or test')
    parser.add_argument('--time_of_day', default=None,
                        help='Filter dataset by time of day: day or night')
    parser.add_argument('--weather', default=None,
                        help='Filter dataset by condition: clear, rain, snow or fog')
    parser.add_argument('--image', type=int, required=False, default=0,
                        help='Number starting image')
    parser.add_argument('--canvas_scale', type=float, required=False, default=0.8,
                        help='Scale factor for the program canvas. Change to adapt the size to your screen.')
    parser.add_argument('--radar_threshold', type=int, required=False, default=75,
                        help='Intensity threshold for the radar. Set to 0 to see the full radar image.')

    args = parser.parse_args()

    visualization(args)
