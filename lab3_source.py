class Array:
    def __init__(self, size):
        self.data = [None] * size

    def __getitem__(self, idx):
        raise NotImplementedError("Not running student code")

    def __setitem__(self, idx, value):
        raise NotImplementedError("Not running student code")

    def __len__(self):
        raise NotImplementedError("Not running student code")


class ArrayList:
    def __init__(self, size=1000):
        self.max_size = size  # maximum memory capacity
        self.data = Array(self.max_size)  # create initial array
        self.curr_size = 0  # current actual size
        # TODO: Feel free to add more lines here

    # TODO: Implement this method - Required Time Complexity: O(1)
    def __getitem__(self, idx):
        print("IDX is", idx, type(idx))
        """Implements 'value = self[idx]'
        Raises IndexError if idx is invalid."""
        raise NotImplementedError("Not running student code")

    # TODO: Implement this method - Required Time Complexity: O(1)
    def __setitem__(self, idx, value):
        """Implements 'self[idx] = value'
        Raises IndexError if idx is invalid."""

        raise NotImplementedError("Not running student code")

    def __len__(self):
        """Implements 'len(self)'"""
        return self.curr_size

    # TODO: Implement this method - Required Time Complexity: O(1), except
    # when you need to create a larger array to fit more elements
    def append(self, value):
        """Appends value to the end of this list."""
        raise NotImplementedError("Not running student code")

    # TODO: Implement this method - Required Time Complexity: O(1), except
    # when you need to create a larger array to fit more elements
    def preprend(self, value):
        """Prepends value to the start of this list."""
        raise NotImplementedError("Not running student code")

    # TODO: Implement this method - Required Time Complexity: O(n), except
    # when idx == 0 or idx == len(self). In these cases, call append/prepend
    def insert(self, idx, value):
        """Inserts value at position idx, shifting the original elements down
        the list, as needed. Note that inserting a value at len(self) ---
        equivalent to appending the value --- is permitted.
        Raises IndexError if idx is invalid."""

        raise NotImplementedError("Not running student code")

    # TODO: Implement this method - Required Time Complexity: O(n), except
    # when 'value' is the first element in the list. In that case,
    # the expected time complexity is O(1)
    def remove(self, value):
        """Removes the first (closest to the front) instance of value from the
        list. Raises a ValueError if value is not found in the list."""

        raise NotImplementedError("Not running student code")

    # TODO: Implement this method - Required Time Complexity: O(n), except
    # when idx == 0 or idx == len(self) - 1. In those cases,
    # the expected time complexity is O(1)
    def delete(self, idx):
        """Removes the element at index 'idx' from the
        list. Raises a IndexError if index is invalid"""

        raise NotImplementedError("Not running student code")

    # TODO: Implement this method - Required Time Complexity: O(n)
    def __contains__(self, value):
        """Implements `val in self`. Returns true iff value is in the list."""

        raise NotImplementedError("Not running student code")


def circular_shift_1(nums, k):
    raise NotImplementedError("Not running student code")


def circular_shift_2(nums, k):
    raise NotImplementedError("Not running student code")


def one_edit_away(s1, s2):
    raise NotImplementedError("Not running student code")
