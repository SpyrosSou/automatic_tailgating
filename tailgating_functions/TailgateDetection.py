try:
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import copy

    from helper_functions import angles_between_angles_radians, calculate_max_speed_difference_two_second_rule

except ImportError as e:
    raise e


class TailgateDetector:
    """
    This is a parent class that accepts the 3D bounding bxo inferences and performs tailgate detection. The data
    of the 3D boxes include x y z locations of each car, along with length-width-height of each box. Some assumptions
    are made for this class to function:

    1) All detected cars are moving away from the camera
    2) There are no stationary cars, since each image is a unique instance, the distances between cars are all examined
    3) Velocities are not considered, instead the maximum speed difference between cars is calculated
    """
    def __init__(self, labeled_data):
        self.labeled_data = labeled_data

        # Create a dictionary that includes the car labels in arrange form, from lower Z (i.e. nearest to the camera)
        self.arranged_labels = self.arranged_cars()

        # Create a dictionary that splits the cars in each image in cases. For example, an image with three cars will
        # have 2 cases, since these are the possible tailgating events that should be explored
        self.paired_data = {}

        # Create the final 2 dictionaries, one containing the cases of tailgating, the other used to store the
        # parameters of tailgating situations
        # Initialize tailgating_cases as a copy of paired_data
        self.tailgating_cases = {}

        # Initialize tailgating_parameters dictionary
        self.tailgating_parameters = {}

    def arranged_cars(self):
        """
        Arrange cars in each image by their Z coordinate and rename them as Car1, Car2, Car3, etc. The car with
        the lowest number (Car1) is closest to the Z axis of the camera. This method is useful as it generates a
        dictionary that will subsequently be used to find "pairs" of cars (i.e. potential tailgating cases)
        Returns
        -------
        A dictionary similar to the input labels, but any non-Car detections are omitted, and the Cars are arranged
        in order of increasing Z, meaning Car1 is the nearest car, etc
        """
        arranged_labels = {}

        # Make a copy of the original labeled_data dictionary
        labeled_data_copy = copy.deepcopy(self.labeled_data)

        for image_name, labels in labeled_data_copy.items():
            # Filter out pedestrians and cyclists, and sort cars by Z coordinate
            cars = [obj for obj in labels if obj['object_type'] == 'Car']
            cars.sort(key=lambda x: x['coordinates']['z'])

            # Rename cars as Car1, Car2, Car3, etc.
            for i, car in enumerate(cars, start=1):
                car['object_type'] = 'Car{}'.format(i)

            # Add arranged labels to the dictionary
            arranged_labels[image_name] = cars

        return arranged_labels

    def find_cases(self):
        """
        Following the arranging of cars in increasing Z, this method splits the cars of each image into pairs. For
        example, if Car1 and Car3 are nearest each other along the direction of motion of the car nearest to the camera.
        They constitute a pair. This is applied to all cars
        in the image.
        Returns
        -------
        Updates self.paired_data
        """

        # Make a copy of the original labeled_data dictionary
        labeled_data_copy = copy.deepcopy(self.arranged_labels)

        # Iterate over each image in the arranged_labels dictionary
        for image_name, data in labeled_data_copy.items():
            # Initialize a list to store tailgating cases for the current image
            cases = []

            # Iterate over each car in the current image
            for i, car1 in enumerate(data):
                # Initialize variables to store the nearest car and its distance along the direction of motion
                nearest_car = None
                min_distance = float('inf')

                # Iterate over other cars in the current image
                for j, car2 in enumerate(data):
                    if i != j:  # Skip comparing car1 with itself
                        # Calculate the distance along the direction of motion from car1 to car2
                        distance_along_direction = self.calculate_distance_along_direction(car1, car2)

                        # Check if car2 is the nearest car along the direction of motion for car1
                        if distance_along_direction < min_distance:
                            min_distance = distance_along_direction
                            nearest_car = car2

                # Add the pair of cars to the paired_data list for the current image
                cases.append((car1, nearest_car))

            # Update the paired_data dictionary with the paired data for the current image
            self.paired_data[image_name] = cases

    @staticmethod
    def calculate_distance_along_direction(car1, car2):
        """
        Calculate the distance along the direction of motion between two cars.
        """
        # Extract the position and direction of motion vectors for car1
        x1, z1 = car1['coordinates']['x'], car1['coordinates']['z']
        dx1 = np.cos(-car1['rotation_y'])
        dz1 = np.sin(-car1['rotation_y'])

        # Extract the position vectors for car2
        x2, z2 = car2['coordinates']['x'], car2['coordinates']['z']

        # Calculate the distance along the direction of motion from car1 to car2
        distance_along_direction = (x2 - x1) * dx1 + (z2 - z1) * dz1

        return distance_along_direction

    def find_cases_along_Z(self):
        """
        Following the arranging of cars in increasing Z, this method splits the cars of each image into pairs. For
        example, if Car1 and Car3 are nearest each other along Z, they constitute a pair. This is applied to all cars
        in the image.
        Returns
        -------
        Updates self.paired_data
        """

        # Make a copy of the original labeled_data dictionary
        labeled_data_copy = copy.deepcopy(self.arranged_labels)

        for image_name, cars in labeled_data_copy.items():
            cases = []

            # Iterate through the cars to find tailgating cases
            for i in range(len(cars) - 1):
                if cars[i]['coordinates']['z'] < cars[i + 1]['coordinates']['z']:
                    cases.append((cars[i], cars[i + 1]))

            # Store tailgating cases for the image
            self.paired_data[image_name] = cases

    def construct_tailgating_dictionaries(self):
        """
        Constructs the sef.tailgating_cases dictionary as a copy of self.paired_data, which will then be
        filtered to only include tailgating pairs
        Returns
        -------

        """
        self.tailgating_cases = self.paired_data.copy()

    def filter_direction_of_motion(self):
        """
        This function edits self.tailgating_cases dictionary. When the dictionary is first created, it is a copy of
        paired_Data dictionary. However, for the sake of this exercise, all cars are assumed to be moving away from the
        camera on the car that detects them. Therefore, this is some artificial data editing to ensure all cars
        detected are moving towards increasing Z.
        Returns
        -------

        """
        for image_name, cases in self.tailgating_cases.items():
            for case in cases:
                car1_rotation = case[0]['rotation_y']
                car2_rotation = case[1]['rotation_y']

                # Check if rotation_y indicates cars are pointing in the direction of decreasing Z
                if car1_rotation > 0:
                    car1_rotation += np.pi  # Invert by adding 180 degrees (pi radians)
                if car2_rotation > 0:
                    car2_rotation += np.pi  # Invert by adding 180 degrees (pi radians)

                # Update rotation_y in the tailgating_cases dictionary
                case[0]['rotation_y'] = car1_rotation
                case[1]['rotation_y'] = car2_rotation

    def filter_tailgating_by_lane(self, threshold: float):
        """
        Filter tailgating cases based on the lane alignment of cars. Specifically, in a pair of cars, the one furthest
        away is projected along the direction of motion of the car closer to the camera. This vector that is normal
        to the motion vector of Car1 is used as the distance between the two cars along the direction of motion.

        It can be approximated to be used in order to check if the cars are moving in the same lane. The threshold
        is used, and if the lane distance between the two cars is greater than the thresholds, the cars are
        considered to move in different lanes.
        Parameters
        ----------
        threshold: A float, in meters, above which the cars are considered to move in different lanes

        Returns
        -------
        Updates the self.tailgating_cases and self.tailgating_parameters to include lane distances.
        """
        # Create new dictionaries to store filtered tailgating cases and parameters
        filtered_tailgating_cases = {}
        filtered_tailgating_parameters = {}

        # Iterate over each image in the tailgating_cases dictionary
        for image_name, pairs in self.tailgating_cases.items():
            # Initialize lists to store filtered pairs and parameters for the current image
            filtered_pairs = []
            filtered_params = []

            # Iterate over each pair of cars in the current image
            for i, (car1, car2) in enumerate(pairs, start=1):
                # Calculate the perpendicular distance between car2 and the direction of motion of car1
                distance = self.calculate_perpendicular_distance(car1, car2)

                # Update tailgating_parameters with relevant information for all pairs
                car1_name = car1['object_type']
                car2_name = car2['object_type']
                params = {'pair': f'{car1_name}-{car2_name}', 'possible_tailgating': 'Yes', 'same_lane': 'Yes',
                          'lane_distance': distance}

                # Check if the distance is above the threshold
                if distance >= threshold:
                    # If the distance is above the threshold, exclude the pair from the filtered pairs list
                    params['possible_tailgating'] = 'No'
                    params['same_lane'] = 'No'
                else:
                    # Add the pair to the filtered pairs list
                    filtered_pairs.append((car1, car2))

                # Add the parameters for the current pair to the filtered_params list
                filtered_params.append(params)

            # Add the filtered pairs for the current image to the filtered tailgating_cases dictionary
            filtered_tailgating_cases[image_name] = filtered_pairs

            # Add the filtered parameters for the current image to the filtered tailgating_parameters dictionary
            filtered_tailgating_parameters[image_name] = filtered_params

        # Update the tailgating_cases and tailgating_parameters dictionaries
        self.tailgating_cases = filtered_tailgating_cases
        self.tailgating_parameters = filtered_tailgating_parameters

    def filter_by_relative_rotation(self, angular_threshold: float = np.pi / 4):
        """
        Compares the direction of motion of two cars, and if the angle between the two vectors exceeds the angular
        threshold, the cars are considered to be moving in different directions and thus no tailgating can occur.

        Note, the smallest angle between the two vectors is used as the angle difference between the two vectors of
        motion. Any car pairs above the angular threshold are filtered out of tailgating_cases.
        Parameters
        ----------
        angular_threshold: the threshold, in rads, above which two cars are considered to move in different directions

        Returns
        -------
        Updates the self.tailgating_cases and self.tailgating_parameters to include lane distances.
        """
        # Create new dictionaries to store filtered tailgating cases and parameters
        filtered_tailgating_cases = {}
        filtered_tailgating_parameters = {}

        # Iterate over each image in the tailgating_cases dictionary
        for image_name, pairs in self.tailgating_cases.items():
            # Initialize lists to store filtered pairs and parameters for the current image
            filtered_pairs = []
            filtered_params = []

            # Iterate over each pair of cars in the current image
            for i, (car1, car2) in enumerate(pairs, start=1):
                # Need to normalise angles to be within 0 and pi in order to apply threshold
                angle_1 = car1['rotation_y']
                angle_2 = car2['rotation_y']

                rotation_diff = min(angles_between_angles_radians(angle_1, angle_2))

                # Update tailgating_parameters with relevant information for all pairs
                car1_name = car1['object_type']
                car2_name = car2['object_type']
                params = {'pair': f'{car1_name}-{car2_name}', 'possible_tailgating': 'Yes',
                          'angular_threshold_between_cars': 'Maintained',
                          'rotational_difference': rotation_diff}

                # Check if the distance is above the threshold
                if rotation_diff >= angular_threshold:
                    # If the distance is above the threshold, exclude the pair from the filtered pairs list
                    params['possible_tailgating'] = 'No'
                    params['angular_threshold_between_cars'] = 'Exceeded'
                else:
                    # Add the pair to the filtered pairs list
                    filtered_pairs.append((car1, car2))

                # Add the parameters for the current pair to the filtered_params list
                filtered_params.append(params)

            # Add the filtered pairs for the current image to the filtered tailgating_cases dictionary
            filtered_tailgating_cases[image_name] = filtered_pairs

            # Add the filtered parameters for the current image to the filtered tailgating_parameters dictionary
            filtered_tailgating_parameters[image_name] = filtered_params

        # Update the tailgating_cases and tailgating_parameters dictionaries
        self.tailgating_cases = filtered_tailgating_cases
        self.tailgating_parameters = filtered_tailgating_parameters

    def detect_tailgating_distance(self):
        """
        Calculate the distance between Car1 and the projection of Car2 on the direction of motion of Car1.
        Update tailgating_parameters dictionary with the calculated distances.
        Returns
        -------
        updates self.tailgating_parameters
        """
        for image_name, pairs in self.tailgating_cases.items():
            # Iterate over each pair of cars in the current image
            for i, (car1, car2) in enumerate(pairs, start=1):
                # Calculate the perpendicular distance between car2 and the direction of motion of car1
                distance = self.calculate_distance_along_direction(car1, car2)

                # Update tailgating_parameters with the current distance for the current pair
                params = {'pair': f'Car{i}-Car{i + 1}', 'current_distance': distance}

                # Append the updated parameters to the tailgating_parameters dictionary
                self.tailgating_parameters[image_name][i - 1].update(params)

    def calculate_tailgating_speed_limits(self):
        """
        Used to calculate the maximum speed difference between two cars, above which tailgating occurs
        Returns
        -------
        updates self.tailgating_parameters
        """
        updated_parameters = {}
        for image_name, data in self.tailgating_parameters.items():
            updated_data = []
            for pair_data in data:
                if 'current_distance' in pair_data:
                    # Calculate the maximum speed difference
                    tailgating_distance = pair_data['current_distance']
                    max_speed_difference_kmh = calculate_max_speed_difference_two_second_rule(tailgating_distance)

                    # Update the entry with the maximum speed difference
                    pair_data['max_speed_difference_kmh'] = max_speed_difference_kmh
                updated_data.append(pair_data)
            updated_parameters[image_name] = updated_data

        # Update self.tailgating_parameters with the modified data
        self.tailgating_parameters = updated_parameters

    def get_paired_cases(self, image_name: str):
        """
        Get tailgating cases for a specific image.
        """
        return self.paired_data.get(image_name, None)

    def get_tailgating_cases(self, image_name: str):
        """
        Get tailgating cases for a specific image.
        """
        return self.tailgating_cases.get(image_name, None)

    def get_tailgating_parameters(self, image_name: str):
        """
        Get tailgating parameters for a specific image.
        """
        return self.tailgating_parameters.get(image_name, None)

    @staticmethod
    def calculate_perpendicular_distance(car1, car2):
        """
        Calculates the distance between a car and the vector of motion of another car
        Parameters
        ----------
        car1
        car2

        Returns
        -------

        """
        # Calculate the direction vector of motion for car1
        dx1 = np.cos(-car1['rotation_y'])
        dz1 = np.sin(-car1['rotation_y'])

        # Vector from car1 to car2
        dx2 = car2['coordinates']['x'] - car1['coordinates']['x']
        dz2 = car2['coordinates']['z'] - car1['coordinates']['z']

        # Dot product between vectors
        dot_product = dx2 * dx1 + dz2 * dz1

        # Projection of car2 onto the direction vector of car1
        projection_x = car1['coordinates']['x'] + dot_product * dx1
        projection_z = car1['coordinates']['z'] + dot_product * dz1

        # Calculate the distance between car2 and its projection onto the direction vector of car1
        distance = np.sqrt(
            (car2['coordinates']['x'] - projection_x) ** 2 + (car2['coordinates']['z'] - projection_z) ** 2)

        return distance

