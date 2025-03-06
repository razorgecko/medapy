def class_in_iterable(iterable, class_obj, iter_name):
    if not all(isinstance(item, class_obj) for item in iterable):
        raise TypeError(f"All items in {iter_name} must be {class_obj.__name__} objects")