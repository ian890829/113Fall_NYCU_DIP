import numpy as np
import cv2
import argparse
import math
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true',default=False)
    parser.add_argument('--median', action='store_false',default=False)
    parser.add_argument('--laplacian', action='store_false',default=False)
    parser.add_argument('--kernelSize',type=int, default=15)
    parser.add_argument('--sigma',default=3)
    parser.add_argument('--filter',type=bool ,default=False,help="For laplacian filter, choosing filter 1 or 2, True for filter 1, False for filter 2")
    args = parser.parse_args()
    return args

def padding(input_img, kernel_size):
    ############### YOUR CODE STARTS HERE ###############

    # Zero padding

    # the paddingSize is defined by how much 0 we need to pad around corner pixel,for example:
    # kernel size=3 , picture is 3*4
    #           oooooo   
    #           oxxxxo
    #           oxxxxo
    #           oxxxxo
    #           oooooo
    # o is padding zero
    # after padding, the size become (3+2*1) * (4+2*1)
    paddingSize=kernel_size//2
    im_Height,im_Width=input_img.shape[0],input_img.shape[1]
    
    #create the return img
    output_img=np.zeros([im_Height+2*paddingSize,im_Width+2*paddingSize,3])

    for i in range(im_Height+2*paddingSize):
        for j in range (im_Width+2*paddingSize):
            for c in range (3):# RGB
                if i<paddingSize or j<paddingSize or i>=im_Height+paddingSize or j>=im_Width+paddingSize:
                    output_img[i,j,c]=0 # padding zero
                else :
                    output_img[i,j,c]= input_img[i-paddingSize,j-paddingSize,c] #keep input pixel value
    ############### YOUR CODE ENDS HERE #################
    return output_img

def convolution(input_img, kernel):
    ############### YOUR CODE STARTS HERE ###############
    kernel_size=kernel.shape[0]
    padImage=padding(input_img,kernel_size) # first, zero-padding for the convolution

    output_img=np.zeros_like(input_img) # output_img should be the same size with input_img

    im_Height,im_Width=input_img.shape[0],input_img.shape[1]
   
    for i in range(im_Height):
        for j in range (im_Width):
            for c in range (3): # color : RGB
                region=padImage[i:i+kernel_size,j:j+kernel_size,c] # define the convoltion region
                output_img[i,j,c]=abs(np.sum(region*kernel)) # do convolution
                
    ############### YOUR CODE ENDS HERE #################
    return output_img
    
def gaussian_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############
    args=parse_args()
    kernelSize=args.kernelSize

    kernel = np.zeros([kernelSize,kernelSize]) # create kernel
    sum_val=0
    mid=kernelSize//2
    sigma=args.sigma
    for i in range(kernelSize):
        for j in range (kernelSize):
            di=i-mid
            dj=j-mid
            kernel[i,j]=np.exp(-(di**2+dj**2)/2*(sigma**2))/(2*math.pi*(sigma**2)) # assign value into kernel using the gaussian function
            sum_val+=kernel[i,j]

    kernel=kernel/sum_val # do regularation so that the sum of kernel index is 1
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

def median_filter(input_img):
    ############### YOUR CODE STARTS HERE ###############

    # Applying median filter without padding:
    # Edge and corner pixels retain their original values as the kernel
    # cannot fully cover them. Only fully covered interior pixels are filtered,
    # preserving boundary information and avoiding padding artifacts.
    # for example : 5*6 picture, 3*3 kernel size
    # c for corner pixel, m for mid pixel(will be changed by median filter)
    #    cccccc
    #    cmmmmc
    #    cmmmmc
    #    cmmmmc
    #    cccccc

    args=parse_args()
    kernelSize=args.kernelSize
    mid=kernelSize//2
    
    output_img=input_img
    im_Height,im_Width=input_img.shape[0],input_img.shape[1]

    for i in range(mid,im_Height-mid): #ignore from 0 to mid
        for j in range (mid,im_Width-mid): # ignore from 0 to mid
            for c in range (3): # color : RGB
                region=input_img[i-mid:i+mid+1,j-mid:j+mid+1,c]
                output_img[i,j,c]=np.median(region) # assign the median of kernel
    ############### YOUR CODE ENDS HERE #################
    return output_img

def laplacian_sharpening(input_img):
    ############### YOUR CODE STARTS HERE ###############
    args=parse_args()

    filter1=np.array([[0,-1,0],
                     [-1,5,-1],
                     [0,-1,0]])
    filter2=np.array([[-1,-1,-1],
                     [-1,9,-1],
                     [-1,-1,-1]])
    # using argument to decide filter 1 or 2
    kernel=filter1 if args.filter else filter2
    ############### YOUR CODE ENDS HERE #################
    return convolution(input_img, kernel)

if __name__ == "__main__":
    
    args = parse_args()
    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img)
        filename=f"G_KS={args.kernelSize}_sigma={args.sigma}.jpg"
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img)
        filename=f"M_KS={args.kernelSize}.jpg"
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img)
        filename="L_Filter1.jpg" if args.filter else "L_filter2.jpg"
    # input_img = cv2.imread("input_part2.jpg")
    # output_img=cv2.Laplacian(input_img,cv2.CV_64F,ksize=3)
    # output_img=np.uint8(np.absolute(output_img))
    cv2.imwrite(filename, output_img)