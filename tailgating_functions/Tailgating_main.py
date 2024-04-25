"""
This file is used to identify the speed differences between pairs of cars above which tailgating occurs. Note,
this file assumes you have generated predictions on the kitti dataset using the SMOKE network.

To deploy, change the paths_to_labels and paths_to_images to the appropriate paths. Also, in line 40, you can choose
an image to visualise the process on. Some examples are included in the comments above, so just change the number in
chosen_image = image_names[NUMBER]

Ultimately, the dictionary tailgate_parameters contains all the car pairs noted in the images, with the maximum
speed difference calculated where applicable.
"""

try:
    import os
    import numpy as np

    from LabelIO import LabelIO
    from LabelLoading import LabelLoader
    from TailgateDetection import TailgateDetector
    from TailgateVisualisation import TailgateVisualisation
    from TailgatingStorage import TailgatingParametersCSVWriter

except ImportError as e:
    raise e


def load_images_and_labels():

    path_to_labels = "/home/spyros/Spyros/temp_repos/SMOKE/outputs/smoke_predictions/predictions"
    path_to_images = "/home/spyros/Spyros/temp_repos/SMOKE/datasets/kitti/testing/image_2"
    label_loader = LabelIO(labels_directory=path_to_labels, image_directory=path_to_images)

    inference_dict = label_loader.get_inference_dictionary()
    image_names = label_loader.get_valid_image_names()

    # 16 is a good example of cars initially "moving" in opposite directions
    # 6 is a good example of 2 cars on same lane in same ...
    # ...direction - USE TO CATCH ERRORS IN PLOTTING, E.G. IF YOU USE THRESHOLD OF 2 YOU GET NO PAIRS AT ALL
    # 7 is good example of multiple cars in same direction - USE AS PRIMARY EXAMPLE
    # 25 is cars all over and kinda breaks the directions - USE TO EXPLORE FILTERING BASED ON ANGLES.
    # 24 is a good example of one car moving towards the camera
    # 46 is a single pedestrian
    chosen_image = image_names[7]
    img_name = label_loader.get_image_path(chosen_image)
    print(f'examined image is {img_name}')

    # You can load the image path of one image or both the image and label path
    # example_image = '000001'
    # some_image_name = label_loader.get_image_path(example_image)
    # some_image_name_and_label = label_loader.get_image_label_paths(example_image)

    print('Loaded all Labelled images')

    # Example usage assuming `label_loader` has been defined and instantiated as before
    label_processor = LabelLoader(label_mapping=inference_dict)
    arranged_inferences = label_processor.get_parsed_data()

    image_data = label_processor.get_image_data(chosen_image)  # Prints all labels of chosen image
    print(image_data)

    return chosen_image, arranged_inferences


def main():

    chosen_image, arranged_inferences = load_images_and_labels()

    # Visualise BEV and arrange car locations based on Z dimension
    tailgate_analysis = TailgateVisualisation(labeled_data=arranged_inferences)  # initialise class
    tailgate_analysis.plot_BEV_vectors(chosen_image, plot_car_boxes=True)  # Show BEV of cars with motion vectors
    tailgate_analysis.plot_BEV_arranged(chosen_image)  # Show arranged cars based on distance from camera

    # Split cars into pairs and plot the pairs with direction of motion
    # tailgate_analysis.find_cases()  # Finds pairs of cars based on direction of motion, TODO: complete
    tailgate_analysis.find_cases_along_Z()  # Detect pairs of cars based on Z distance
    tailgate_analysis.plot_paired_cases(chosen_image)  # Plots subfigures with car pairs

    # Creates tailgating dictionaries and filters direction of motion to artificially make cars move away from camera
    tailgate_analysis.construct_tailgating_dictionaries()  # Make sure to run this to copy the paired_data
    tailgate_analysis.filter_direction_of_motion()
    # tailgate_analysis.plot_paired_cases_filtered_directions(chosen_image)

    # Filters out pairs of cars whose directions of motion exceed an angular difference of angular_threshold
    tailgate_analysis.filter_by_relative_rotation(angular_threshold=np.pi / 6)

    # This is where tailgating checks begin, first by checking if the cars are in the same "lane"
    tailgate_analysis.filter_tailgating_by_lane(threshold=1)  # Filters out pairs not in the same "lane"
    tailgate_analysis.plot_lane_distances(chosen_image)

    # Detects the distance between the pairs of cars that can be tailgating
    tailgate_analysis.detect_tailgating_distance()
    paired_data = tailgate_analysis.get_paired_cases(chosen_image)
    tailgate_cases = tailgate_analysis.get_tailgating_cases(chosen_image)
    tailgate_parameters = tailgate_analysis.get_tailgating_parameters(chosen_image)
    tailgate_analysis.plot_tailgating_distances(chosen_image)  # Plots the tailgating distances

    tailgate_analysis.calculate_tailgating_speed_limits()  # Calculates the maximum speed difference between cars
    tailgate_parameters = tailgate_analysis.get_tailgating_parameters(chosen_image)
    print(tailgate_parameters)

    # Save tailgating data to csv. TODO: include image name
    # tailgating_storer = TailgatingParametersCSVWriter(tailgate_analysis.tailgating_parameters,
    #                                                   output_dir='/home/spyros/Spyros/temp_repos/SMOKE/output_tailgating')
    # tailgating_storer.write_to_csv(filename='Tailgating_Results')


if __name__ == "__main__":
    main()
