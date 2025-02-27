import requests
from PIL import Image

def get_image_from_url(url):
    request = requests.get(url,stream=True).raw
    image=Image.open(request)
    return image
