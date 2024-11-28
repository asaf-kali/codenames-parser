import os

RESOURCE_DIR = os.path.join(os.path.dirname(__file__))


def get_resource_path(file: str) -> str:
    return os.path.join(RESOURCE_DIR, file)


def get_card_template_path():
    return get_resource_path("card_template.jpg")
