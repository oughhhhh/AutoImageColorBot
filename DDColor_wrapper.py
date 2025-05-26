import base64
import requests
import PIL.Image
from io import BytesIO


def DDColor(input_bytes):
    input_bytes = "data:image/png;base64," + base64.b64encode(input_bytes).decode('ascii')
    url = "http://localhost:5000/predictions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "input": {
            "image": input_bytes,
            "model_size": "large"
        }
    }
    response = requests.post(url, headers=headers, json=data)
    data = response.json()["output"][len("data:image/png;base64,"):]
    imgdata = base64.b64decode(data)
    return imgdata


if __name__ == "__main__":
    x = "https://replicate.delivery/pbxt/KDMkjS4SpsGieAxMdkBUNWT5zFI8BvAU4XjiyI2xmLny3skZ/Buffalo%20Bank%20Buffalo%2C%20New%20York%2C%20circa%201908.%20Erie%20County%20Savings%20Bank%2C%20Niagara%20Street.jpg"
    imgdata = DDColor(x)
    image = PIL.Image.open(BytesIO(imgdata)).convert("RGB")
    image.show()
