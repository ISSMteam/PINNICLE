
def is_file_ext(path, ext):
    """
    check if a given path is ended by ext
    """
    if isinstance(path, str):
        if path.endswith(ext):
            return True

    return False
