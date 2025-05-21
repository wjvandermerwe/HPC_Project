from PIL import Image

with Image.open("./cmake-build-release/out_test") as img:
    img.save("./output.jpg", format="JPEG")
