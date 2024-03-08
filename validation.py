class Validation:
    @staticmethod
        
    @staticmethod
    def validate_int(value: int):
        if isinstance(value, int):
            return value
        raise ValueError("Value must be of int type")

    @staticmethod
    def validate_polynomial_degree(value):
        if isinstance(value, tuple) and len(value) == 2:
            start, stop = value
            if isinstance(start, int) and isinstance(stop, int) and start > 0 and stop > start:
                return True
        raise ValueError("Value must be a tuple of two positive integers representing the range for np.arange(start:stop)")

    @staticmethod
    def read_in_bool(message: str):
        while True:
            user_input = input(message)
            if user_input in ["TRUE", "True", "true", "t"]:
                return True
            elif user_input in ["FALSE", "False", "false", "f"]:
                return False
            print("Invalid input. Please enter 'True' or 'False'.")

    @staticmethod
    def read_in_float(validation_function, message: str):
        while True:
            try:
                user_input = float(input(message))
                if validation_function(user_input):
                    return user_input
            except ValueError:
                pass
            print("Invalid input. Please enter a valid float value.")

    @staticmethod
    def read_in_integer(validation_function, message: str):
        while True:
            try:
                user_input = int(input(message))
                if validation_function(user_input):
                    return user_input
            except ValueError:
                pass
            print("Invalid input. Please enter a valid integer value.")

    @staticmethod
    def read_in_integer_tuple(validation_function, message: str):
        while True:
            user_input = input(message)
            try:
                value = tuple(map(int, user_input.split(',')))
                if validation_function(value):
                    return value
            except ValueError:
                pass
            print("Invalid input. Please enter two integers separated by a comma.")

    @staticmethod
    def validate_test_size(value: float):
        if isinstance(value, float) and 0.0 < value < 1.0:
            return value
        raise ValueError("Value must be a float between 0.0 and 1.0")