from random import randrange


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
        random_substring = get_random_substring(string, substring_length, step=step)
        random_substrings.append(random_substring)

    return random_substrings
