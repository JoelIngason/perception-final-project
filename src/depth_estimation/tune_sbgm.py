import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class StereoSGBMTuner:
    def __init__(self, master, left_image_path, right_image_path):
        self.master = master
        master.title("StereoSGBM Tuner")

        # Load stereo images
        self.left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        self.right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

        if self.left_img is None or self.right_img is None:
            raise ValueError("Error loading images. Check the paths.")

        # Initialize default StereoSGBM parameters
        self.min_disparity = 50
        self.num_disparities = 160  # Must be divisible by 16
        self.block_size = 11
        self.P1 = 3 * self.block_size**2 * 8
        self.P2 = 3 * self.block_size**2 * 32
        self.disp12_max_diff = 0
        self.uniqueness_ratio = 10
        self.speckle_window_size = 100
        self.speckle_range = 1
        self.pre_filter_cap = 0
        self.mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

        # Set up GUI elements
        self.setup_gui()

        # Initial disparity map
        self.update_disparity()

    def setup_gui(self):
        # Create frames
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        image_frame = ttk.Frame(self.master)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create sliders for parameters
        # minDisparity
        ttk.Label(control_frame, text="Min Disparity").pack()
        self.min_disp_slider = ttk.Scale(
            control_frame,
            from_=-100,
            to=500,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.min_disp_slider.set(self.min_disparity)
        self.min_disp_slider.pack(fill=tk.X, padx=5, pady=5)

        # numDisparities
        ttk.Label(control_frame, text="Num Disparities (16*X)").pack()
        self.num_disp_slider = ttk.Scale(
            control_frame,
            from_=1,
            to=200,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.num_disp_slider.set(self.num_disparities // 16)
        self.num_disp_slider.pack(fill=tk.X, padx=5, pady=5)

        # blockSize
        ttk.Label(control_frame, text="Block Size").pack()
        self.block_size_slider = ttk.Scale(
            control_frame,
            from_=3,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.block_size_slider.set(self.block_size)
        self.block_size_slider.pack(fill=tk.X, padx=5, pady=5)

        # P1
        ttk.Label(control_frame, text="P1").pack()
        self.p1_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=1000,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.p1_slider.set(self.P1)
        self.p1_slider.pack(fill=tk.X, padx=5, pady=5)

        # P2
        ttk.Label(control_frame, text="P2").pack()
        self.p2_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=10000,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.p2_slider.set(self.P2)
        self.p2_slider.pack(fill=tk.X, padx=5, pady=5)

        # disp12MaxDiff
        ttk.Label(control_frame, text="disp12MaxDiff").pack()
        self.disp12_max_diff_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=1000,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.disp12_max_diff_slider.set(self.disp12_max_diff)
        self.disp12_max_diff_slider.pack(fill=tk.X, padx=5, pady=5)

        # uniquenessRatio
        ttk.Label(control_frame, text="Uniqueness Ratio").pack()
        self.uniqueness_ratio_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.uniqueness_ratio_slider.set(self.uniqueness_ratio)
        self.uniqueness_ratio_slider.pack(fill=tk.X, padx=5, pady=5)

        # speckleWindowSize
        ttk.Label(control_frame, text="Speckle Window Size").pack()
        self.speckle_window_size_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=500,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.speckle_window_size_slider.set(self.speckle_window_size)
        self.speckle_window_size_slider.pack(fill=tk.X, padx=5, pady=5)

        # speckleRange
        ttk.Label(control_frame, text="Speckle Range").pack()
        self.speckle_range_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=50,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.speckle_range_slider.set(self.speckle_range)
        self.speckle_range_slider.pack(fill=tk.X, padx=5, pady=5)

        # preFilterCap
        ttk.Label(control_frame, text="Pre Filter Cap").pack()
        self.pre_filter_cap_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=500,
            orient=tk.HORIZONTAL,
            command=self.on_parameter_change,
        )
        self.pre_filter_cap_slider.set(self.pre_filter_cap)
        self.pre_filter_cap_slider.pack(fill=tk.X, padx=5, pady=5)

        # Mode (dropdown)
        ttk.Label(control_frame, text="Mode").pack()
        self.mode_var = tk.StringVar()
        self.mode_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            state="readonly",
        )
        self.mode_combobox["values"] = [
            "STEREO_SGBM_MODE_SGBM",
            "STEREO_SGBM_MODE_SGBM_3WAY",
            "STEREO_SGBM_MODE_HH",
            "STEREO_SGBM_MODE_HH4",
        ]
        self.mode_combobox.current(1)  # Default to SGBM_3WAY
        self.mode_combobox.bind("<<ComboboxSelected>>", self.on_parameter_change)
        self.mode_combobox.pack(fill=tk.X, padx=5, pady=5)

        # Disparity Map display
        self.disparity_label = ttk.Label(image_frame)
        self.disparity_label.pack()

    def on_parameter_change(self, event=None):
        # Retrieve current values from sliders
        self.min_disparity = int(self.min_disp_slider.get())
        self.num_disparities = int(self.num_disp_slider.get()) * 16
        self.num_disparities = max(self.num_disparities, 16)
        self.block_size = int(self.block_size_slider.get())
        if self.block_size % 2 == 0:
            self.block_size += 1  # blockSize must be odd
        self.P1 = int(self.p1_slider.get())
        self.P2 = int(self.p2_slider.get())
        self.disp12_max_diff = int(self.disp12_max_diff_slider.get())
        self.uniqueness_ratio = int(self.uniqueness_ratio_slider.get())
        self.speckle_window_size = int(self.speckle_window_size_slider.get())
        self.speckle_range = int(self.speckle_range_slider.get())
        self.pre_filter_cap = int(self.pre_filter_cap_slider.get())
        mode_str = self.mode_var.get()

        # Map mode string to OpenCV constant
        mode_mapping = {
            "STEREO_SGBM_MODE_SGBM": cv2.STEREO_SGBM_MODE_SGBM,
            "STEREO_SGBM_MODE_SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            "STEREO_SGBM_MODE_HH": cv2.STEREO_SGBM_MODE_HH,
            "STEREO_SGBM_MODE_HH4": cv2.STEREO_SGBM_MODE_HH4,
        }
        self.mode = mode_mapping.get(mode_str, cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        self.update_disparity()

    def update_disparity(self):
        # Initialize StereoSGBM with current parameters
        stereo = cv2.StereoSGBM.create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.P1,
            P2=self.P2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_range,
            preFilterCap=self.pre_filter_cap,
            mode=self.mode,
        )

        # Compute disparity map
        disparity = stereo.compute(self.left_img, self.right_img).astype(np.float32) / 16.0

        # Normalize the disparity map for display
        disp_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_display = np.uint8(disp_display)

        # Convert to RGB for Tkinter
        disp_color = cv2.applyColorMap(disp_display, cv2.COLORMAP_JET)
        disp_image = Image.fromarray(disp_color)
        disp_photo = ImageTk.PhotoImage(image=disp_image)

        # Update image in the label
        self.disparity_label.configure(image=disp_photo)
        self.disparity_label.image = disp_photo  # Keep a reference


def main():
    # Paths to the left and right images
    left_image_path = "../../data/34759_final_project_rect/seq_01/image_02/data/000000.png"
    right_image_path = "../../data/34759_final_project_rect/seq_01/image_03/data/000000.png"

    root = tk.Tk()
    app = StereoSGBMTuner(root, left_image_path, right_image_path)
    root.mainloop()


if __name__ == "__main__":
    main()
