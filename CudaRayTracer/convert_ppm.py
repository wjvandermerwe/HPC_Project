from PIL import Image

with Image.open("./cmake-build-release/out") as img:
    img.save("./output.jpg", format="JPEG")
