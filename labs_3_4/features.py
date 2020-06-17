import cv2
import mahotas
bins = 8


def fd_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 5, 50)
    return canny.flatten()


def fd_sobelx(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    return sobelx.flatten()


def fd_sobely(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    return sobely.flatten()


def fd_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.flatten()


def fd_kaze(image):
    kaze = cv2.KAZE_create()
    # kp, dsc = kaze.detectAndCompute(image, None)

    kps = kaze.detect(image)
    kps = sorted(kps, key=lambda x: -x.response)[:64]

    kps, dsc = kaze.compute(image, kps)

    return dsc.flatten()


def fd_sift(image):
    sift = cv2.xfeatures2d.SIFT_create(800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kps = sift.detect(gray)
    kps = sorted(kps, key=lambda x: -x.response)[:32]

    kps, dsc = sift.compute(gray, kps)

    return dsc.flatten()


# feature-descriptor-1: Hu Moments - Shape
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture - Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram - Color
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
