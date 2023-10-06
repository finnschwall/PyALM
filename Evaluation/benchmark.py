import os

def find_small_files(folder_path, threshold_gb):
    threshold_bytes = threshold_gb * (1024 ** 3)  # Convert GB to bytes
    small_files = []

    for foldername, subfolders, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            try:
                # Get file size in bytes
                file_size = os.path.getsize(file_path)
                if file_size < threshold_bytes:
                    # If file size is smaller than the threshold, add its absolute path to the list
                    small_files.append(os.path.abspath(file_path))
            except OSError:
                # Handle permission errors or other file access issues
                pass

    return small_files

# Example usage:
folder_path = '/path/to/your/folder'
threshold_gb = 1  # Specify the threshold size in gigabytes
small_files_list = find_small_files(folder_path, threshold_gb)

# Print the list of absolute paths to small files
print(small_files_list)