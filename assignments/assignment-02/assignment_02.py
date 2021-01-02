
import math
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt



class ImageOperation:
    def __init__(self,image_file):
        # read image to array
        self.image = Image.open(image_file)
        # image.show()
        
        self.modified_image = Image.open(image_file)
        
        # convert image into greyscale
        self.image_grey = self.image.convert("L")
        # img.show()
        
        self.dimension = self.image_grey.size
        # print(self.dimension)
    
    #Question no: 1
    def laplacianEdgeProfile(self):
        
        image_values = np.array(self.image_grey)
#         final_image_values = np.array(self.image_grey)
        final_image_values = np.zeros((self.dimension[1],self.dimension[0]))
        for i in range(len(image_values)):
            for j in range(len(image_values[0])):
                if i == 0: y_previous = 0
                else: y_previous = image_values[i-1][j]
                if i == len(image_values) - 1: y_forward = 0
                else: y_forward = image_values[i+1][j]
                if j == 0: x_previous = 0
                else: x_previous = image_values[i][j-1]
                if j == len(image_values[0]) - 1: x_forward = 0
                else: x_forward = image_values[i][j+1]
                    
                center = 4*(image_values[i][j])
#                 print(x_forward, x_previous, y_forward, y_previous, center)
                final_value = x_previous + x_forward + y_previous + y_forward - center 
#                 if final_value < 0:
#                     print(final_value, i)
                final_image_values[i][j] = final_value

        img = Image.fromarray(final_image_values)
        img.show()
        
        
    def laplacianEdgeProfile(self, image):
        
        image_values = np.array(image)
#         final_image_values = np.array(self.image_grey)
        final_image_values = np.zeros((self.dimension[1],self.dimension[0]))
        for i in range(len(image_values)):
            for j in range(len(image_values[0])):
                if i == 0: y_previous = 0
                else: y_previous = image_values[i-1][j]
                if i == len(image_values) - 1: y_forward = 0
                else: y_forward = image_values[i+1][j]
                if j == 0: x_previous = 0
                else: x_previous = image_values[i][j-1]
                if j == len(image_values[0]) - 1: x_forward = 0
                else: x_forward = image_values[i][j+1]
                    
                center = 4*(image_values[i][j])
#                 print(x_forward, x_previous, y_forward, y_previous, center)
                final_value = x_previous + x_forward + y_previous + y_forward - center 
#                 if final_value < 0:
#                     print(final_value, i)
                final_image_values[i][j] = final_value

        img = Image.fromarray(final_image_values)
        return img 
        
    
    
        
    #Question no: 2
    def laplacianOfGaussian(self, sigma):
        """First we will calculate the Gaussian lowpass filter to smooth the image and then apply the Laplacian to get the required filter"""
        image_values = np.array(self.image_grey)
        final_image_values = np.zeros((self.dimension[1],self.dimension[0]))
#         sigma = np.var(image_values)
        
        # Gaussian Low pass filter
        for i in range(len(image_values)):
            for j in range(len(image_values[0])):
                
                final_value = round(math.exp((-(image_values[i][j]**2) / (sigma**2))))
#                 print(final_value)
                final_image_values[i][j] = final_value
                
                
        img = Image.fromarray(final_image_values,"L")
#         instance = ImageOperation(img)
        img = self.laplacianEdgeProfile(img)
        img.show()
        
        
    #Question no: 3
    def signalToNoiseRatio(self, image_file):
        original_image_values = np.array(self.image_grey)
        # read image to array
        noisey_image = Image.open(image_file)
        # image.show()
        
        # convert image into greyscale
        image_grey = noisey_image.convert("L")
        noisey_image_values = np.array(image_grey)
        
        sum_of_noise_values = 0
        for i in range(len(noisey_image_values)):
            for j in range(len(noisey_image_values[0])):
                sum_of_noise_values += (noisey_image_values[i][j]**2)
                
        difference_of_noisey_original = 0
        for i in range(len(noisey_image_values)):
            for j in range(len(noisey_image_values[0])):
                difference_of_noisey_original += ((original_image_values[i][j] - noisey_image_values[i][j]) ** 2)
                
        snr = sum_of_noise_values // difference_of_noisey_original 
        return snr
                
        
    #Question no: 4
    def frequencyDomain(self):
        original_image_values = np.array(self.image_grey)
        magnitude_image_values = np.zeros((self.dimension[1],self.dimension[0]))
        phase_image_values = np.zeros((self.dimension[1],self.dimension[0]))
        
        M = len(magnitude_image_values)
        N = len(magnitude_image_values[0])  
        for m in range(len(magnitude_image_values)):
            for n in range(len(magnitude_image_values[0])):
                imaginary_value = 0
                real_value = 0
                for x in range(len(original_image_values)):
                    for y in range(len(original_image_values[0])):
                        power_value = ((x * m) / M) + ((y*n) / N)
                        result = f"{(math.e**(math.pi*2j*power_value)) * original_image_values[x][y]:.2f}"
                        r_value = 1
                        if result[0] == '-':
                            result = result[1:]
                            r_value *= -1
                        result = result.split("-")
                        if len(result)>1:
                            real_value += r_value*float(result[0])
                            imaginary_value += -1*float(result[1][:-1])

                        else:
                            result = result[0].split("+")
                            real_value += r_value*float(result[0])
                            imaginary_value += float(result[1][:-1])
                            
                magnitude_image_values[m][n] = ((real_value ** 2) + (imaginary_value ** 2)) ** 0.5
                try:
                    phase_image_values[m][n] = math.atan(abs(real_value/imaginary_value))
                except ZeroDivisionError as error:
#                     print(error)
                    phase_image_values[m][n] = math.radians(90)
                    
        img = Image.fromarray(magnitude_image_values)
        img.show()
        
        
    
    
    #Question no: 5 
    def lowPassFilter(self, cut_off_distance=5):
        image_values = np.array(self.image_grey)
        M = len(image_values)
        N = len(image_values[0])
        filter_image_values = np.zeros((self.dimension[1],self.dimension[0]))
        for u in range(len(filter_image_values)):
            for v in range(len(filter_image_values[0])):
                distance = ((((u - M) / 2) ** 2) + (((v - N) / 2) ** 2)) ** 0.5
                if distance <= cut_off_distance:
                    filter_image_values[u][v] = 1
                else:
                    filter_image_values[u][v] = 0
                    
    
    
    #Question no: 6 
    def highPassFilter(self, cut_off_distance=5):
        image_values = np.array(self.image_grey)
        M = len(image_values)
        N = len(image_values[0])
        filter_image_values = np.zeros((self.dimension[1],self.dimension[0]))
        for u in range(len(filter_image_values)):
            for v in range(len(filter_image_values[0])):
                distance = ((((u - M) / 2) ** 2) + (((v - N) / 2) ** 2)) ** 0.5
                if distance <= cut_off_distance:
                    filter_image_values[u][v] = 0
                else:
                    filter_image_values[u][v] = 1
                


if __name__ == "__main__":
    # image = "./pictures/Fig0338(a)(blurry_moon).tif"
    image = "./pictures/Fig0338(a)(blurry_moon).tif"
    instance = ImageOperation(image)
    # instance.laplacianEdgeProfile()
    # instance.laplacianOfGaussian(10)
    instance.signalToNoiseRatio(image)
    