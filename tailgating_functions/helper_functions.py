
try:
    import os

    import numpy as np
    import math

except ImportError as e:
    raise e


def normalize_angle(angle_deg):
    # Normalize angle to the range 0 to 360 degrees
    normalized_angle = angle_deg % 360
    return normalized_angle


def normalize_angle_rad(angle_rad):
    # Normalize angle to the range 0 to 2*pi radians
    normalized_angle = angle_rad % (2 * np.pi)
    return normalized_angle


def angles_between_angles_degrees(angle1_deg, angle2_deg):
    """
    Converts angles (degrees) into unit vectors and calculates the two angles between them, allowing to find the
    minimum angular difference between two paths.
    Parameters
    ----------
    angle1_deg: Angle of car 1 in degrees
    angle2_deg: Angle of car 2 in degrees

    Returns
    -------
    angle_radians_1, angle_radians_2: The two possible angles between two unit vectors in rads
    """
    # Normalize angles to the range 0 to 360 degrees
    angle1_deg = normalize_angle(angle1_deg)
    angle2_deg = normalize_angle(angle2_deg)

    # Convert angles to radians
    angle1_rad = np.radians(angle1_deg)
    angle2_rad = np.radians(angle2_deg)

    # Convert angles to unit vectors
    vector1 = np.array([np.cos(angle1_rad), np.sin(angle1_rad)])
    vector2 = np.array([np.cos(angle2_rad), np.sin(angle2_rad)])

    # Calculate the angles between the vectors (in radians)
    angle_radians_1 = np.arccos(np.dot(vector1, vector2))
    angle_radians_2 = np.arccos(np.dot(-vector1, vector2))

    return angle_radians_1, angle_radians_2


def angles_between_angles_radians(angle1_rad, angle2_rad):
    """
    Converts angles (rads) into unit vectors and calculates the two angles between them, allowing to find the
    minimum angular difference between two paths.
    Parameters
    ----------
    angle1_rad: Angle of car 1 in rads
    angle2_rad: Angle of car 2 in rads

    Returns
    -------
    angle_radians_1, angle_radians_2: The two possible angles between two unit vectors in rads
    """
    # Normalize angles to the range 0 to 360 degrees
    angle1_deg = np.degrees(angle1_rad)
    angle2_deg = np.degrees(angle2_rad)

    angle1_deg = normalize_angle(angle1_deg)
    angle2_deg = normalize_angle(angle2_deg)

    # Convert angles to radians
    angle1_rad = np.radians(angle1_deg)
    angle2_rad = np.radians(angle2_deg)

    # Convert angles to unit vectors
    vector1 = np.array([np.cos(angle1_rad), np.sin(angle1_rad)])
    vector2 = np.array([np.cos(angle2_rad), np.sin(angle2_rad)])

    # Calculate the angles between the vectors (in radians)
    angle_radians_1 = np.arccos(np.dot(vector1, vector2))
    angle_radians_2 = np.arccos(np.dot(-vector1, vector2))

    return angle_radians_1, angle_radians_2


def calculate_max_speed_difference_two_second_rule(distance_meters):
    """
    Applies the 2 second rule given a distance in order to determine the speed threshold above which tailgating occurs
    Parameters
    ----------
    distance_meters: The distance between two cars in meters

    Returns
    -------
    max_speed_difference_kmh: The speed difference in km per h above which tailgating occurs
    """
    # Convert distance from meters to kilometers
    distance_km = distance_meters / 1000

    # Assuming the maximum allowable time gap is 2 seconds
    time_gap_seconds = 2

    # Calculate the maximum allowable speed difference in kilometers per hour
    max_speed_difference_kmh = (distance_km / time_gap_seconds) * 3600

    return max_speed_difference_kmh
