import cv2


def ImgAdaptiveBinary(image, k=(3, 3), thr=11):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gauss = cv2.GaussianBlur(gray, k, 1)
    gaus = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thr, 1)
    image = cv2.cvtColor(gaus, cv2.COLOR_GRAY2RGB)

    return image