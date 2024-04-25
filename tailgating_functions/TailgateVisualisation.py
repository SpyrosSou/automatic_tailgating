try:
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    from TailgateDetection import TailgateDetector

except ImportError as e:
    raise e


class TailgateVisualisation(TailgateDetector):
    """
    Child class of TailgateDetector, used for plotting figures in BEV, along with directions of motion and car
    bounding boxes.
    """

    def __init__(self, labeled_data):
        super().__init__(labeled_data)

        # Define the lists that will be populated in each image for plotting (and reset in each method) that will
        # contain the necessary parameters for plotting
        self.car_x = []
        self.car_z = []
        self.car_rotation_y = []
        self.car_length = []
        self.car_width = []

    def plot_BEV_vectors(self, image_name: str, plot_car_boxes: bool = True):
        """
        Plots a standard BEV diagram of all detected cars on the chosen image based on the inference results from
        the SMOKE network

        Parameters
        ----------
        image_name: A string indicating the image name (e.g. 000006)
        plot_car_boxes: Set to true if you want to plot the car bounding boxes in BEV

        Returns
        -------
        Plots a BEV figure
        """
        # Ensure labels exist for chosen image
        data = self.labeled_data.get(image_name, None)
        if data is None:
            print("No data available for image {}".format(image_name))
            return

        # Extract x, z, and rotation_y for cars. Also use flag in case cars should be plotted
        # Reset the class variables for each image
        self.car_x = []
        self.car_z = []
        self.car_rotation_y = []
        self.car_length = []
        self.car_width = []

        for obj in data:
            if obj['object_type'] == 'Car':
                self.car_x.append(obj['coordinates']['x'])
                self.car_z.append(obj['coordinates']['z'])
                self.car_rotation_y.append(obj['rotation_y'])

                self.car_length.append(obj['3Dbox_dimensions']['length'])
                self.car_width.append(obj['3Dbox_dimensions']['width'])

        # Plot the bird's-eye view with direction vectors
        plt.figure(figsize=(8, 6))
        plt.scatter(self.car_x, self.car_z, color='blue', marker='o')
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')
        plt.title('BEV of Detected Cars with Motion Vectors')
        plt.grid(True)
        # plt.gca().invert_yaxis()  # Invert y-axis to align with the direction of motion

        # Plot direction vectors
        for x, z, rotation_y in zip(self.car_x, self.car_z, self.car_rotation_y):
            # Convert rotation angle to unit vector
            dx = np.cos(-rotation_y)  # Use this for unit vector according to KITTI dataset orientations
            dz = np.sin(-rotation_y)
            plt.arrow(x, z, dx, dz, head_width=0.5, head_length=0.5, fc='red', ec='red')

        # Use the flag in the arguments to decide if the box that illustrate cars should be visualised.
        if plot_car_boxes:
            # Plot bounding boxes for cars
            for x, z, length, width, rotation_y in zip(self.car_x, self.car_z, self.car_length, self.car_width,
                                                       self.car_rotation_y):
                rotated_corners = self.define_car_boxes_in_BEV(x, z, length, width, rotation_y)

                # Plot the rotated bounding box
                plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], color='black')
                plt.fill(rotated_corners[:, 0], rotated_corners[:, 1], color='none', edgecolor='black')

        # TODO: Fix the dynamic figure boundaries
        xlim1, xlim2, zlim1, zlim2 = self.define_BEV_plot_ranges()  # Dynamically adjust axes limits
        # Address cases where only one car is in the image
        if xlim1 == xlim2:
            xlim1 += 5
            xlim2 -= 5

        plt.xlim(xlim2, xlim1)
        plt.ylim(zlim2, zlim1)  # Add some padding to the z axis
        plt.show()

    def plot_BEV_arranged(self, image_name: str, plot_car_boxes: bool = True):
        """
        Plots a BEV figure, but the cars have different colours based on their distance from the camera along Z. The
        car nearest is named Car1, then Car2 etc in increasing Z.

        Parameters
        ----------
        image_name: A string indicating the image name (e.g. 000006)
        plot_car_boxes: Set to true if you want to plot the car bounding boxes in BEV

        Returns
        -------
        Plots a BEV figure with color-coded cars based on distance from camera
        """
        data = self.arranged_labels.get(image_name, None)
        if data is None:
            print("No data available for image {}".format(image_name))
            return

        # Extract x, z, and rotation_y for cars. Also use flag in case cars should be plotted
        # Reset the class variables for each image
        self.car_x = []
        self.car_z = []
        self.car_rotation_y = []
        self.car_length = []
        self.car_width = []

        for obj in data:
            self.car_x.append(obj['coordinates']['x'])
            self.car_z.append(obj['coordinates']['z'])
            self.car_rotation_y.append(obj['rotation_y'])

            self.car_length.append(obj['3Dbox_dimensions']['length'])
            self.car_width.append(obj['3Dbox_dimensions']['width'])

        # Create a list of colors for each car
        colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))

        # Plot the bird's-eye view
        plt.figure(figsize=(8, 6))
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')
        plt.title('BEV of Arranged Cars in Increasing Z distance')
        plt.grid(True)

        # Plot each car with a different color
        for i, (car, color) in enumerate(zip(data, colors), start=1):
            plt.scatter(car['coordinates']['x'], car['coordinates']['z'], color=color, label=car['object_type'])

        car_labels = ['Car{}'.format(i) for i in range(1, len(data) + 1)]

        # Use the flag in the arguments to decide if the box that illustrate cars should be visualised.
        if plot_car_boxes:
            # Plot bounding boxes for cars
            for x, z, length, width, rotation_y in zip(self.car_x, self.car_z, self.car_length, self.car_width,
                                                       self.car_rotation_y):
                rotated_corners = self.define_car_boxes_in_BEV(x, z, length, width, rotation_y)

                # Plot the rotated bounding box
                plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], color='black')
                plt.fill(rotated_corners[:, 0], rotated_corners[:, 1], color='none', edgecolor='black')

        xlim1, xlim2, zlim1, zlim2 = self.define_BEV_plot_ranges()
        if xlim1 == xlim2:
            xlim1 += 5
            xlim2 -= 5

        plt.xlim(xlim2, xlim1)
        plt.ylim(zlim2, zlim1)  # Add some padding to the z axis
        plt.legend()
        plt.show()

    def plot_paired_cases(self, image_name: str, plot_car_boxes: bool = True, plot_motion_vectors: bool = True):
        """
        This function creates subplots that isolate car pairs, i.e. the only possible tailgating cases. Each pair
        will be shown in its own subplot.
        Parameters
        ----------
        image_name: A string indicating the image name (e.g. 000006)
        plot_car_boxes: Set to true if you want to plot the car bounding boxes in BEV
        plot_motion_vectors: Set to true to plot motion vectors of all cars

        Returns
        -------
        Subfigures of car pairs that could potentially tailgate
        """
        cases = self.paired_data.get(image_name, None)
        if cases is None:
            print("No paired cases found for image {}".format(image_name))
            return

        # Create subplots for each paired case
        num_cases = len(cases)
        try:
            fig, axes = plt.subplots(1, num_cases, figsize=(6 * num_cases, 6))
        except ValueError as e:
            print('Fewer than 2 cars detected in the image')
            return

        # Plot each tailgating case
        for i, (car1, car2) in enumerate(cases, start=1):
            ax = axes[i - 1] if num_cases > 1 else axes
            ax.set_title("Paired Case {}".format(i))
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Z Coordinate')
            ax.grid(True)

            # Plot cars
            ax.scatter(car1['coordinates']['x'], car1['coordinates']['z'], color='blue', label=car1['object_type'])
            ax.scatter(car2['coordinates']['x'], car2['coordinates']['z'], color='red', label=car2['object_type'])
            ax.legend()

            # Define the x-axis limits based on the range of z axis, with xlim2 being the low limit etc
            # Note, the z value of car1 will always be less than that of car2 since it is arranged to be closer
            z_range = np.abs(car2['coordinates']['z'] - car1['coordinates']['z'])
            x_mid = np.mean((car1['coordinates']['x'], car2['coordinates']['x']))
            xlim1 = x_mid + z_range / 2
            xlim2 = x_mid - z_range / 2

            ax.set_xlim(xlim2, xlim1)

            # Plot car bounding boxes based on input flag
            if plot_car_boxes:
                for car in (car1, car2):
                    rotated_corners = self.define_car_boxes_in_BEV(car['coordinates']['x'], car['coordinates']['z'],
                                                                   car['3Dbox_dimensions']['length'],
                                                                   car['3Dbox_dimensions']['width'], car['rotation_y'])

                    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color='black')
                    ax.fill(rotated_corners[:, 0], rotated_corners[:, 1], color='none', edgecolor='black')

            # Plot motion vectors dependingo on input flag
            if plot_motion_vectors:
                for car in (car1, car2):
                    # Convert rotation angle to unit vector
                    dx = np.cos(-car['rotation_y'])
                    dz = np.sin(-car['rotation_y'])
                    ax.arrow(car['coordinates']['x'], car['coordinates']['z'], dx, dz, head_width=0.5, head_length=0.5,
                             fc='red', ec='red')

        plt.tight_layout()
        plt.show()

    def plot_paired_cases_filtered_directions(self, image_name: str, plot_car_boxes: bool = True,
                                              plot_motion_vectors: bool = True):
        """
        This function creates subplots that isolate car pairs, i.e. the only possible tailgating cases. Each pair
        will be shown in its own subplot. Unlike the previous method, however, this uses the tailgating_cases dict
        which has filtered the direction of motion of each car to ensure it is pointing in increasing Z.
        Parameters
        ----------
        image_name: A string indicating the image name (e.g. 000006)
        plot_car_boxes: Set to true if you want to plot the car bounding boxes in BEV
        plot_motion_vectors: Set to true to plot motion vectors of all cars

        Returns
        -------
        Subfigures of car pairs that could potentially tailgate, however their motions have been artificially filtered
        to point towards increasing Z
        """
        cases = self.tailgating_cases.get(image_name, None)
        if cases is None:
            print("No paired cases found for image {}".format(image_name))
            return

        # Create subplots for each paired case
        num_cases = len(cases)
        try:
            fig, axes = plt.subplots(1, num_cases, figsize=(6 * num_cases, 6))
        except ValueError as e:
            print('Fewer than 2 cars detected in the image')
            raise e

        # Plot each tailgating case
        for i, (car1, car2) in enumerate(cases, start=1):
            ax = axes[i - 1] if num_cases > 1 else axes
            ax.set_title("Paired Case {}".format(i))
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Z Coordinate')
            ax.grid(True)

            # Plot cars
            ax.scatter(car1['coordinates']['x'], car1['coordinates']['z'], color='blue', label=car1['object_type'])
            ax.scatter(car2['coordinates']['x'], car2['coordinates']['z'], color='red', label=car2['object_type'])
            ax.legend()

            # Define the x-axis limits based on the range of z axis, with xlim2 being the low limit etc
            # Note, the z value of car1 will always be less than that of car2 since it is arranged to be closer
            z_range = np.abs(car2['coordinates']['z'] - car1['coordinates']['z'])
            x_mid = np.mean((car1['coordinates']['x'], car2['coordinates']['x']))
            xlim1 = x_mid + z_range / 2
            xlim2 = x_mid - z_range / 2

            ax.set_xlim(xlim2, xlim1)

            if plot_car_boxes:
                for car in (car1, car2):
                    rotated_corners = self.define_car_boxes_in_BEV(car['coordinates']['x'], car['coordinates']['z'],
                                                                   car['3Dbox_dimensions']['length'],
                                                                   car['3Dbox_dimensions']['width'], car['rotation_y'])

                    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], color='black')
                    ax.fill(rotated_corners[:, 0], rotated_corners[:, 1], color='none', edgecolor='black')

            if plot_motion_vectors:
                for car in (car1, car2):
                    # Convert rotation angle to unit vector
                    dx = np.cos(-car['rotation_y'])
                    dz = np.sin(-car['rotation_y'])
                    ax.arrow(car['coordinates']['x'], car['coordinates']['z'], dx, dz, head_width=0.5, head_length=0.5,
                             fc='red', ec='red')

        plt.tight_layout()
        plt.show()

    def plot_lane_distances(self, image_name):
        """
        Creates subfigures indicating the "lane" distance for each pair of cars, i.e. the normal distance between
        the care potentially being tailgated and the motion vector of the tailgating car
        Parameters
        ----------
        image_name: A string indicating the image name (e.g. 000006)

        Returns
        -------
        A BEV set of subfigures with the distances between car and direction of motion being displayed
        """
        # Get the tailgating cases for the specified image
        pairs = self.paired_data.get(image_name, None)
        if pairs is None:
            print("No tailgating cases found for image {}".format(image_name))
            return

        # Create a subplot for each pair of cars
        num_pairs = len(pairs)
        try:
            fig, axs = plt.subplots(1, num_pairs, figsize=(6 * num_pairs, 6))
        except ValueError as e:
            print('Fewer than 2 cars detected in the image - No pair is present')
            return

        for i, (car1, car2) in enumerate(pairs):

            if num_pairs == 1:
                ax = axs
            else:
                ax = axs[i]

            if not car1 or not car2:
                continue
            # Plot car1
            ax.scatter(car1['coordinates']['x'], car1['coordinates']['z'], color='blue',
                       label=car1['object_type'])

            # Plot car2
            ax.scatter(car2['coordinates']['x'], car2['coordinates']['z'], color='red',
                       label=car2['object_type'])

            # Plot a line representing the direction of motion of car1
            ax.plot([car1['coordinates']['x'], car1['coordinates']['x'] + 20 * np.cos(-car1['rotation_y'])],
                    [car1['coordinates']['z'], car1['coordinates']['z'] + 20 * np.sin(-car1['rotation_y'])],
                    color='green', linestyle='--', label='Direction of motion')

            # Calculate the perpendicular distance between car2 and the direction of motion of car1
            distance = self.calculate_perpendicular_distance(car1, car2)

            # Calculate the endpoint of the line segment from car1 to car2
            dx1 = np.cos(-car1['rotation_y'])
            dz1 = np.sin(-car1['rotation_y'])
            endpoint_x = car1['coordinates']['x'] + dx1 * distance
            endpoint_z = car1['coordinates']['z'] + dz1 * distance

            # Plot the line segment from car1 to car2
            ax.plot([car1['coordinates']['x'], endpoint_x],
                    [car1['coordinates']['z'], endpoint_z],
                    color='black', label='Perpendicular Distance')

            # Add a text annotation with the distance value
            ax.text((car1['coordinates']['x'] + endpoint_x) / 2,
                    (car1['coordinates']['z'] + endpoint_z) / 2,
                    'Distance: {:.2f}'.format(distance),
                    fontsize=10, ha='center', va='bottom')

            ax.legend()
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Z Coordinate')
            car1_name = car1['object_type']
            car2_name = car2['object_type']

            ax.set_title(f'Tailgating Case {car1_name} - {car2_name} - Image: {image_name}')
        plt.tight_layout()
        plt.show()

    def plot_tailgating_distances(self, image_name):
        """
        Plots the distances between the two potentially tailgating cars along the direction of motion of the
        tailgating car
        Parameters
        ----------
        image_name: A string indicating the image name (e.g. 000006)

        Returns
        -------
        A BEV plot showing the current distance along the motion of the tailgating car
        """
        # Get the tailgating cases for the specified image
        pairs = self.tailgating_cases.get(image_name, None)
        parameters = self.tailgating_parameters.get(image_name, [])

        if pairs is None:
            print("No tailgating cases found for image {}".format(image_name))
            return

        # Create a subplot for each pair of cars
        num_pairs = len(pairs)
        try:
            fig, axs = plt.subplots(1, num_pairs, figsize=(6 * num_pairs, 6))
        except ValueError as e:
            print('Fewer than 2 cars detected in the image - No tailgating pair')
            return

        fig.suptitle(f'Tailgating Distance Plot for Image: {image_name}', fontsize=16)

        # Iterate over each pair of cars
        for i, ((car1, car2), params) in enumerate(zip(pairs, parameters)):
            if num_pairs == 1:
                ax = axs
            else:
                ax = axs[i]

            if not car1 or not car2:
                continue

            car1_name = car1['object_type']
            car2_name = car2['object_type']

            ax.set_title(f'Tailgating Distance {car1_name} - {car2_name} - Image: {image_name}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Z Coordinate')

            # Plot Car1
            ax.scatter(car1['coordinates']['x'], car1['coordinates']['z'], color='blue', label=car1_name)

            # Plot Car2
            ax.scatter(car2['coordinates']['x'], car2['coordinates']['z'], color='red', label=car2_name)

            # Calculate the endpoint of the distance vector
            dx = np.cos(-car1['rotation_y'])
            dz = np.sin(-car1['rotation_y'])
            endpoint_x = car2['coordinates']['x'] - dx * params['current_distance']
            endpoint_z = car2['coordinates']['z'] - dz * params['current_distance']

            # Plot the distance vector
            ax.arrow(car2['coordinates']['x'], car2['coordinates']['z'],
                     endpoint_x - car2['coordinates']['x'], endpoint_z - car2['coordinates']['z'],
                     head_width=0.1, head_length=0.1, fc='green', ec='green')
            # Add the distance value as text
            ax.text(car2['coordinates']['x'] + (endpoint_x - car2['coordinates']['x']) / 2,
                    car2['coordinates']['z'] + (endpoint_z - car2['coordinates']['z']) / 2,
                    f'{params["current_distance"]:.2f}', fontsize=8, ha='center', va='bottom')

            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def define_car_boxes_in_BEV(x, z, length, width, rotation_y):
        """
        Provides parameters used for plotting car boxes in BEV
        Parameters
        ----------
        x: Location of examined car in x
        z: Location of examined car in z
        length: Length of car 3D box (Length is defined along x)
        width: Width of car 3D box (Width is defined along z)
        rotation_y: Rotation of car's axis of motion with respect to camera Z frame

        Returns
        -------
        rotated_corners: Numpy array that includes the locations of the four corners of the car following rotation
        """

        car_center = np.array([x, z])
        # Calculate corner points of the bounding box
        corners = np.array([
            [-length / 2, -width / 2],  # Front-left
            [length / 2, -width / 2],  # Front-right
            [length / 2, width / 2],  # Back-right
            [-length / 2, width / 2],  # Back-left
        ])
        # Rotate the corners based on the rotation angle
        rotation_matrix = np.array([
            [np.cos(-rotation_y), -np.sin(-rotation_y)],
            [np.sin(-rotation_y), np.cos(-rotation_y)]
        ])
        rotated_corners = np.dot(corners, rotation_matrix.T) + car_center

        return rotated_corners

    def define_BEV_plot_ranges(self):
        """
        Need a way to ensure x and z axes are comparable, otherwise car boxes look very strange. To do so,
        find the range of z = abs(max - min), and then use the same range to define the xlimits. Note, add a padding
        of 5m in z to better visualise edge cases.
        Returns
        -------
        X and Y axes limits for dynamic range adjustments
        """

        try:
            min_z = min(self.car_z)
            max_z = max(self.car_z)
            z_range = np.abs(max_z - min_z)
        except ValueError as e:
            print('Image contains no Cars')
            raise e

        # Find the mean x coordinate and add half the range on each side
        central_x_coordinate = np.mean(self.car_x)
        x_axis_top_limit = central_x_coordinate + z_range / 2
        x_axis_bottom_limit = central_x_coordinate - z_range / 2

        z_axis_top_limit = max_z + 5  # Add 5 meters to better visualise
        z_axis_bottom_limit = min_z - 5

        return x_axis_top_limit, x_axis_bottom_limit, z_axis_top_limit, z_axis_bottom_limit
