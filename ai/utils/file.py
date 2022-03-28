import os 
def read_file_content(file_path):
        """
        Read file content and return it as string array with each line as an element.
        """
        with open(file_path, 'r') as file:
            file_content = file.read()
        return file_content.split('\n')