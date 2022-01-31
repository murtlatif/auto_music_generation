from random import randrange
from typing import Optional


def get_random_substring(string: str, substring_length: int, step: int = 1) -> str:
    """
    Gets a random substring of the specified size from the string.

    Args:
        string (str): The string to extract a substring from
        substring_length (int): The length of the substring to extract
        step (int, optional): The step of initial indices selected. Defaults to 1.

    Returns:
        str: The extracted substring
    """
    random_idx = randrange(0, len(string) - substring_length + 1, step=step)
    end_idx = random_idx + substring_length

    random_substr = string[random_idx:end_idx]
    return random_substr


def get_random_substrings(string: str, substring_length: int, num_substrings: int, step: int = 1) -> list[str]:
    """
    Returns a list of random substrings of the specified size from the string.
    This may include duplicates and overlap.

    Args:
        string (str): The string to extract substrings from
        substring_length (int): The length of the substring to extract
        num_substrings (int): The number of substrings to extract
        step (int, optional): The step of initial indices selected. Defaults to 1.

    Returns:
        list[str]: The extracted substrings
    """
    random_substrings: list[str] = []
    for i in range(num_substrings):
        random_substring = get_random_substring(
            string, substring_length, step=step)
        random_substrings.append(random_substring)

    return random_substrings


def get_all_substrings(string: str, max_size: Optional[int] = None) -> list[str]:
    """
    Returns all the substrings until max_size. If max_size is None then
    it will be set to the length of the string.

    Args:
        string (str): The string to extract substrings from
        max_size (Optional[int]): The maximum size of the substring

    Returns:
        list[str]: All substrings of size 1 to max_size
    """
    all_substrings: list[str] = []
    if not max_size:
        max_size = len(string)

    for start_idx in range(len(string)):
        # TODO: Do not copy final sizes multiple times
        substrings = [string[start_idx:start_idx+size]
                      for size in range(1, max_size+1)]
        all_substrings.extend(substrings)

    return all_substrings
