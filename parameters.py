import cv2 as cv

# In aceasta clasa vom stoca detalii legate de algoritm si de imaginea pe care este aplicat.
class Parameters:

    def is_grayscale(self):
        print(self.image.shape)
        for i in range(len(self.image)):
            for j in range(len(self.image[i])):
                # print(self.image[i])
                if self.image[i][j][0] != self.image[i][j][1] or self.image[i][j][1] != self.image[i][j][2]:
                    return False
        return True

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv.imread(image_path)
        if self.image is None:
            print('%s is not valid' % image_path)
            exit(-1)
        if self.is_grayscale():
            self.grayscale = True
            self.image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2GRAY)
        else:
            self.grayscale = False
        self.image_resized = None
        self.small_images_dir = './../data/colectie/'
        self.image_type = 'png'
        self.num_pieces_horizontal = 100
        self.num_pieces_vertical = None
        self.show_small_images = False
        self.layout = 'caroiaj'
        self.criterion = 'aleator'
        self.hexagon = False
        self.small_images = None
        self.different_neighbours = False
        self.cifar = False
        self.cifar_name = ''