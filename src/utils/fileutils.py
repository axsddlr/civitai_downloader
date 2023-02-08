import os


def checkIfFileExists(filename, size):
    """
    If the file exists and is the correct size, return True, otherwise return False

    :param filename: The name of the file to check
    :param size: The size of the file in bytes
    :return: True or False
    """
    if os.path.exists(filename):
        if os.path.getsize(filename) == size:
            return True
    return False


def compareSizes(filename, size):
    """
    If the file exists and is the right size, return True, otherwise return False

    :param filename: The name of the file to check
    :param size: The size of the file in bytes
    :return: A boolean value.
    """
    if os.path.exists(filename):
        if os.path.getsize(filename) == size:
            return True
    return False
