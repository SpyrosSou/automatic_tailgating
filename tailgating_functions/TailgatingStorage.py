try:
    import os
    import csv

except ImportError as e:
    raise e


class TailgatingParametersCSVWriter:
    def __init__(self, tailgating_parameters: dict, output_dir: str):
        self.tailgating_parameters = tailgating_parameters
        self.output_dir = output_dir

    def write_to_csv(self, filename):
        # Collect all unique parameter names
        all_parameters = set()
        for data_list in self.tailgating_parameters.values():
            for data in data_list:
                all_parameters.update(data.keys())

        # Write data to CSV
        with open(f"{self.output_dir}/{filename}.csv", mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=sorted(all_parameters), extrasaction='ignore')
            writer.writeheader()
            for image_name, data_list in self.tailgating_parameters.items():
                for data in data_list:
                    writer.writerow(data)
