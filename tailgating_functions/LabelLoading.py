try:
    import os
except ImportError as e:
    raise e


class LabelLoader:
    """
    This class takes in the loaded label paths and images and reads them according to the kitti data structure.
    Details of this structure can be found here:
    https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb

    The labels look like this:

    <object_type> <truncation> <occlusion> <alpha> <left> <top> <right> <bottom> <height> <width> <length>
    <x> <y> <z> <rotation_y>

    object_type is Car, Pedestrian, or Cyclist. Truncation is a float between 0 and 1. Occlusion is an integer between
    0 and some other value, possibly 3. Alpha is the observation angle, between -pi and pi. Left, top, right, bottom
    are 4 numbers used for 2D bounding box generation, and are pixel coordinates. Height, width, and length are
    the parameters for 3D box generation, along y, z, and x axis of camera respectively.
    x, y, z are the 3D coordinates of the detected object in camera coordinates. Rotation_y is a camera rotation
    between -pi and pi about the camera y axis.

    All these have been loaded using LabelIO class, and the paths exist in a dictionary passed as an input in
    this class. The output is a dictionary with the keys being the image names used for inference. Each entry in
    this dictionary includes a list of dictionaries. The number of dictionaries is equal to the objects detected
    in the examined image. For example, image 000002.png has two cars, and therefore the list has 2 dictionaries, one
    for each line. These data will be used in following classes to detect tailgating.
    """

    def __init__(self, label_mapping: dict):
        self.label_mapping = label_mapping
        self.label_data = self.process_labels()  # Dictionary of all labels

    def process_labels(self):
        data = {}
        for image_name, paths in self.label_mapping.items():
            if paths['label_path']:  # Ensure there is a label path
                with open(paths['label_path'], 'r') as file:
                    detections = [self.parse_line(line.strip()) for line in file if line.strip()]
                data[image_name] = detections
        return data

    def parse_line(self, line):
        fields = line.split()
        return {
            'object_type': fields[0],
            'truncation': float(fields[1]),
            'occlusion': int(fields[2]),
            'alpha': float(fields[3]),
            'bbox': [float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7])],
            '3Dbox_dimensions': {'height': float(fields[8]), 'width': float(fields[9]), 'length': float(fields[10])},
            'coordinates': {'x': float(fields[11]), 'y': float(fields[12]), 'z': float(fields[13])},
            'rotation_y': float(fields[14])
        }

    def get_image_data(self, image_name):
        """ Returns the processed label data for a specific image. """
        return self.label_data.get(image_name, None)

    def get_parsed_data(self):
        return self.label_data
