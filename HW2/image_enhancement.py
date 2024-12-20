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
import cv2
import numpy as np

def histogram_equalization(img):
    
    hist = np.zeros((3, 256), dtype=int)

    #spilt the R G B channel 
    R, G, B = cv2.split(img)

  
    R_eq = np.zeros_like(R)
    G_eq = np.zeros_like(G)
    B_eq = np.zeros_like(B)

    
    for i, channel in enumerate([R, G, B]):
        # calculate histogram using calcHist
        temp = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        hist[i, :] = temp  #update the hist

        # calculate cdf
        cdf = np.cumsum(temp)
        cdf_min = cdf[np.nonzero(cdf)].min()
        cdf_normalized = (cdf - cdf_min) / (cdf[-1] - cdf_min) * 255
        cdf_normalized = cdf_normalized.astype('uint8')

        # transform the cdf to channel
        equalized_channel = cdf_normalized[channel]

        # update the equalized channel 
        if i == 0:
            R_eq = equalized_channel
        elif i == 1:
            G_eq = equalized_channel
        else:
            B_eq = equalized_channel

    
    equalized_img = cv2.merge([R_eq, G_eq, B_eq])

    return equalized_img
        

        

    





"""
Bonus
"""
def other_enhancement_algorithm():
    pass


"""
Main function
"""
def main():
    img = cv2.imread("data/image_enhancement/input.bmp")

    # TODO: modify the hyperparameter
    gamma_list = [0.5, 1, 2] # gamma value for gamma correction

    # TODO Part 1: Gamma correction
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img,gamma)

        cv2.imshow("Gamma correction | Gamma = {}".format(gamma), np.vstack([img, gamma_correction_img]))
        cv2.waitKey(0)

    # TODO Part 2: Image enhancement using the better balanced image as input
    histogram_equalization_img = histogram_equalization(img)

    cv2.imshow("Histogram equalization", np.vstack([img, histogram_equalization_img]))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
