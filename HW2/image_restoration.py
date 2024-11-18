import cv2
import numpy as np


"""
TODO Part 1: Motion blur PSF generation
"""
def generate_motion_blur_psf(img, imgnum):
    shape = img.shape[:2]  


    if imgnum == 0:
        angle = 30
        length = 20
    else:
        angle = 15
        length = 10

   
    psf = np.zeros(shape)
    center_row = shape[0] // 2
    psf[center_row, :length] = 1  


    rot = cv2.getRotationMatrix2D((shape[1] // 2, shape[0] // 2), angle, 1)
    psf = cv2.warpAffine(psf, rot, (shape[1], shape[0]))

    psf /= psf.sum()
    return psf


"""
TODO Part 2: Wiener filtering (for color images)
"""
def wiener_filtering(img, psf, K):
    
    channels = cv2.split(img)
    restored_channels = []
    for channel in channels:
      
        G = np.fft.fft2(channel)
        H = np.fft.fft2(psf, s=channel.shape)
        H_conj = np.conj(H)

       
        S = np.abs(H) ** 2
        F_hat = (H_conj / (S + K)) * G

        f_hat = np.fft.ifft2(F_hat)
        f_hat = np.abs(f_hat)  
        restored_channel = np.clip(f_hat, 0, 255).astype(np.uint8)
        restored_channels.append(restored_channel)

    
    return cv2.merge(restored_channels)


"""
TODO Part 3: Constrained least squares filtering (for color images)
"""
def constrained_least_square_filtering(img, psf, alpha):
    channels = cv2.split(img)
    restored_channels = []
    for channel in channels:
        
        G = np.fft.fft2(channel)
        H = np.fft.fft2(psf, s=channel.shape)
        H_conj = np.conj(H)

        
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        P = np.fft.fft2(laplacian, s=channel.shape)

        
        F_hat = (H_conj / (np.abs(H) ** 2 + alpha * np.abs(P) ** 2)) * G

        
        f_hat = np.fft.ifft2(F_hat)
        f_hat = np.abs(f_hat)  
        restored_channel = np.clip(f_hat, 0, 255).astype(np.uint8)
        restored_channels.append(restored_channel)

    
    return cv2.merge(restored_channels)


"""
PSNR Calculation
"""
def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    mse = np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2)
    if mse == 0:  
        return float("inf")
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


"""
Main function
"""
def main():
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))

        if img_original is None or img_blurred is None:
            print(f"Error: Could not load images for testcase {i+1}")
            continue

        psf = generate_motion_blur_psf(img_blurred, i)

        K = 0.01
        wiener_img = wiener_filtering(img_blurred, psf, K)

        alpha = 0.01
        constrained_least_square_img = constrained_least_square_filtering(img_blurred, psf, alpha)

        print("\n---------- Testcase {} ----------".format(i + 1))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, wiener_img)))

        print("Method: Constrained least squares filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, constrained_least_square_img)))

        wiener_img = np.clip(wiener_img, 0, 255).astype(np.uint8)
        constrained_least_square_img = np.clip(constrained_least_square_img, 0, 255).astype(np.uint8)

        cv2.imshow(f"Testcase {i+1} Results", np.hstack([img_blurred, wiener_img, constrained_least_square_img]))
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
