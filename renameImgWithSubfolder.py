import os

def add_suffix_to_files_in_folder(folder_path, chosen_suffix, suffix=""):
    try:
        # Iterate over all the files and directories in the current folder
        for filename in os.listdir(folder_path):
            # Get the full path of the file or directory
            full_path = os.path.join(folder_path, filename)
            
            # If it's a directory, call the function recursively
            if os.path.isdir(full_path):
                # Add the subfolder name to the suffix
                new_suffix = f"{suffix}_{filename}" if suffix else filename
                add_suffix_to_files_in_folder(full_path, chosen_suffix, new_suffix)
            
            # If it's a file, rename it
            elif os.path.isfile(full_path):
                # Split the filename into name and extension
                name, ext = os.path.splitext(filename)
                
                # Create the new filename with the chosen suffix and subfolder names
                new_filename = f"{name}_{chosen_suffix}{suffix}{ext}"
                
                # Get the full path for the new filename
                new_full_path = os.path.join(folder_path, new_filename)
                
                # Rename the file
                os.rename(full_path, new_full_path)
                print(f"Renamed: {filename} -> {new_filename}")
                
        print(f"Renaming completed in folder: {folder_path}")
        
    except Exception as e:
        print(f"Error: {e}")

# Example usage
folder_path = os.getcwd()  # Automatically detect the current folder
chosen_suffix = "ciao"  # Choose the suffix to add
add_suffix_to_files_in_folder(folder_path, chosen_suffix)
