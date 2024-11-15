import cv2
import numpy as np



"""
TODO Part 1: Gamma correction
"""
def gamma_correction(img,gamma):
    #normorlize image
    nm_img=img/255.0
    #do the gamma correction
    ret_img=np.power(nm_img,gamma)
    #denormorlize img
    ret_img=(ret_img*255.0).astype(np.uint8)

    return ret_img


"""
TODO Part 2: Histogram equalization
"""
def histogram_equalization(img):
    hist=np.zeros((3,256),dtype=int)
    R,G,B= cv2.spilt(img)
    
    for i, channel in enumerate([R,G,B]):
        values,counts=np.unique(channel,return_counts=True)
        hist[i,values]=counts



"""
Bonus
"""
def other_enhancement_algorithm():
    raise NotImplementedError


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # TODO: modify the hyperparameter
    gamma_list = [1, 1, 1] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img)

        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.waitKey(0)

    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)

    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
