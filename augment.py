import numpy as np

import cv2

__all__ = [
    "insert_brightness",
    "insert_shadow",
    "insert_snow",
    "insert_rain",
    "insert_fog",
]


def insert_brightness(image):

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    hls = np.array(hls, dtype=np.float64)

    coeff = np.random.uniform(0.5, 1.5)

    hls[:, :, 1] = hls[:, :, 1] * coeff

    hls[:, :, 1][hls[:, :, 1] > 255] = 255

    hls = np.array(hls, dtype=np.uint8)

    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    return rgb


def _generate_shadow_coordinates(imshape, no_of_shadows=1):

    vertices_list = []

    for index in range(no_of_shadows):
        vertex = []
        for dimensions in range(np.random.randint(3, 15)):
            vertex.append(
                (
                    imshape[1] * np.random.uniform(),
                    imshape[0] // 3 + imshape[0] * np.random.uniform(),
                )
            )
            vertices = np.array([vertex], dtype=np.int32)
            vertices_list.append(vertices)

    return vertices_list


def insert_shadow(image, no_of_shadows=1):

    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list = _generate_shadow_coordinates(imshape, no_of_shadows)

    for vertices in vertices_list:

        cv2.fillPoly(mask, vertices, 255)
        image_HLS[:, :, 1][mask[:, :, 0] == 255] = (
            image_HLS[:, :, 1][mask[:, :, 0] == 255] * 0.5
        )
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


def insert_snow(image):

    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float64)
    brightness_coefficient = 2.5
    snow_point = 140
    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] = (
        image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] * brightness_coefficient
    )
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


def _generate_random_lines(imshape, slant, drop_length):
    drops = []
    for i in range(1500):
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
            y = np.random.randint(0, imshape[0] - drop_length)
            drops.append((x, y))
    return drops


def insert_rain(image):
    imshape = image.shape
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 20
    drop_width = 2
    drop_color = (200, 200, 200)
    rain_drops = _generate_random_lines(imshape, slant, drop_length)
    for rain_drop in rain_drops:
        cv2.line(
            image,
            (rain_drop[0], rain_drop[1]),
            (rain_drop[0] + slant, rain_drop[1] + drop_length),
            drop_color,
            drop_width,
        )
    image = cv2.blur(image, (7, 7))
    brightness_coefficient = 0.7
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * brightness_coefficient
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


def _blur(image, x, y, hw):

    image[y : y + hw, x : x + hw, 1] = image[y : y + hw, x : x + hw, 1] + 1
    image[:, :, 1][image[:, :, 1] > 255] = 255
    image[y : y + hw, x : x + hw, 1] = cv2.blur(
        image[y : y + hw, x : x + hw, 1], (10, 10)
    )
    return image


def _generate_blur_coordinates(imshape, hw):

    blur_points = []
    midx = imshape[1] // 2 - hw - 100
    midy = imshape[0] // 2 - hw - 100
    index = 1
    while midx > -100 or midy > -100:
        for i in range(250 * index):

            x = np.random.randint(midx, imshape[1] - midx - hw)

            y = np.random.randint(midy, imshape[0] - midy - hw)

            blur_points.append((x, y))

            midx -= 250 * imshape[1] // sum(imshape)

            midy -= 250 * imshape[0] // sum(imshape)

            index += 1

    return blur_points


def insert_fog(image):

    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    mask = np.zeros_like(image)

    imshape = image.shape

    hw = 100

    image_HLS[:, :, 1] = image_HLS[:, :, 1] * 0.8

    haze_list = generate_blur_coordinates(imshape, hw)

    for haze_points in haze_list:

        image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255

        image_HLS = _blur(image_HLS, haze_points[0], haze_points[1], hw)

    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)

    return image_RGB
