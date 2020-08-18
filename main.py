from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk
import cv2
import random
import numpy as np
from PIL import Image
import pygame

class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Project 309 - Abnormal Vision Simulator")
        self.minsize(1000, 720)
        self.wm_iconbitmap('icon.ico')
        self['bg'] = '#5d666a'
        #8d9593
        #33464b
        #5d666a
        #a6a297
        self.labelFrame = ttk.LabelFrame(self, text = "Abnormal Vision Simulator", labelanchor = N)
        self.labelFrame.grid(column = 0, row = 1, padx = 140, pady = 10)
        self.button()
        self.cataract_but()
        self.color1_but()
        self.color2_but()
        self.color3_but()
        self.asti_but()
        self.glaucoma()
        self.Macular_Deg()
        self.drunk_but()
        self.bli_but()
        self.sketch_but()
        self.car_but()
        #self.LSD_but()
        self.pet_but()
        self.torn_retina()
        self.face()
        #self.red_but()
        #self.green_but()
        #self.blue_but()
        #self.flip_but()
        self.reset()

    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse An Image", command = self.fileDialog)
        self.button.grid(column = 3, row = 1)

    def cataract_but(self):
        self.cataract_but = ttk.Button(self.labelFrame, text = "Cataract" , command = self.cataract)
        self.cataract_but.grid(column = 0, row = 2)

    def color1_but(self):
        self.color1_but = ttk.Button(self.labelFrame, text = "Tritanopia", command = self.color2 )
        self.color1_but.grid(column = 1, row = 2)

    def color2_but(self):
        self.color2_but = ttk.Button(self.labelFrame, text = "Protanopia", command = self.color1)
        self.color2_but.grid(column = 2, row = 2)

    def color3_but(self):
        self.color3_but = ttk.Button(self.labelFrame, text = "Achromatopsia", command = self.color3)
        self.color3_but.grid(column = 3, row = 2)

    def bli_but(self):
        self.bli_but = ttk.Button(self.labelFrame, text = "Pr. Blind", command = self.pr_blind)
        self.bli_but.grid(column = 6, row = 2)

    def glaucoma(self): ### veya torn_retina
        self.color3_but = ttk.Button(self.labelFrame, text = "Glaucoma",  command = self.gla)
        self.color3_but.grid(column = 5, row = 2)

    def asti_but(self):
        self.asti_but = ttk.Button(self.labelFrame, text = "Astigmatism", command = self.asti)
        self.asti_but.grid(column = 4, row = 2)

    """
    def flip_but(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "Flip", command = self.flip)
        self.pet_but.grid(column = 0, row = 3)
    """
    def drunk_but(self):
        self.drunk_but = ttk.Button(self.labelFrame, text = "Drunk",  command = self.alc)
        self.drunk_but.grid(column = 1, row = 3)

    def pet_but(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "Pets", command = self.pet)
        self.pet_but.grid(column = 2, row = 3)

    def Macular_Deg(self):
        self.color3_but = ttk.Button(self.labelFrame, text = "Macular Deg.", command = self.AMD)
        self.color3_but.grid(column = 4, row = 3)
    """

    def LSD_but(self):
        self.hero_but = ttk.Button(self.labelFrame, text = "LSD", command = self.LSD)
        self.hero_but.grid(column =4, row = 3)
    """
    def face(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "Detect Faces ", command = self.faces)
        self.pet_but.grid(column = 3, row =3)

    def car_but(self):
        self.coc_but = ttk.Button(self.labelFrame, text = "Cartoon", command = self.cart)
        self.coc_but.grid(column = 5, row = 3)

    def sketch_but(self):
        self.weed_but = ttk.Button(self.labelFrame, text = "Sketch", command = self.sek)
        self.weed_but.grid(column = 6, row = 3)

    def torn_retina(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "Torn Retina", command = self.torn)
        self.pet_but.grid(column = 0, row =3)
        
    """
    def red_but(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "Red", command = self.red)
        self.pet_but.grid(column = 2, row =4)
    def green_but(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "Green", command = self.green)
        self.pet_but.grid(column = 3, row = 4)
    def blue_but(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "Blue", command = self.blue)
        self.pet_but.grid(column = 4, row = 4)

    """
    def reset(self):
        self.pet_but = ttk.Button(self.labelFrame, text = "RESET", command = self.reset_but)
        self.pet_but.grid(column = 3, row = 5)


    def fileDialog(self):
        global img
        self.filename = filedialog.askopenfilename(filetypes = [("Image File", "*.bmp"),("Image File", "*.jpg"),("Image File", "*.png"),("Image File", "*.jpeg")])
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)

        self.img = Image.open(self.filename)
        self.resized_img = self.img.resize((500, 500))
        render = ImageTk.PhotoImage(self.resized_img)
        img = Label(self, image=render)
        img.image = render
        img.place(x=500, y=450, anchor = CENTER)
        self.button['state'] = DISABLED


        """
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=500, y=450, anchor = CENTER)
        self.button['state'] = DISABLED
        """

    def cataract(self):
        global text
        img = cv2.imread(self.filename)
        blurImg = cv2.blur(img,(9,9))
        cv2.imshow('CATARCT',blurImg)
        """
        root = Tk()
        text = Text(root)
        text.insert(INSERT, "CATARCT" "\n")
        text.insert(INSERT, "A cataract is a clouding of the normally clear lens of your eye. For people who have cataracts, seeing through cloudy lenses is a bit like looking through a frosty or fogged-up window.")
        text.pack()
        """

    def color1(self):
        def colour_blind_adjustments(base_image, image_one):
            #screen = pygame.display.set_mode((400, 300))
            protanopia_image = image_one
            protanomaly_colour = pygame.PixelArray(protanopia_image)
            image_width, image_height = base_image.get_size()
            for x in range(0, image_width):
                for y in range(0, image_height):
                    red = base_image.get_at((x, y)).r
                    green = base_image.get_at((x, y)).g
                    blue = base_image.get_at((x, y)).b
                    
                    red_channel_red = red * 0.56667
                    red_channel_green = green * 0.43337
                    red_channel_blue = blue * 0.0
                    green_channel_red = red * 0.55883
                    green_channel_green = green * 0.44167
                    green_channel_blue = blue * 0.0
                    blue_channel_red = red * 0.0
                    blue_channel_green = green * 0.24167
                    blue_channel_blue = blue * 0.75833
                    red = red_channel_red + red_channel_green + red_channel_blue
                    green = green_channel_red + green_channel_green + green_channel_blue
                    blue = blue_channel_red + blue_channel_green + blue_channel_blue
                    
                    protanomaly_colour[x, y] = (blue, green, red)
            del protanomaly_colour
            pygame.image.save(protanopia_image, "protoanomaly_screen.PNG")
            #cv2.imshow('CATARCT',protanopia_image)
            #pygame.image.save(protanopia_image, "protoanomaly")
            #pygame.display.set_caption('protanopia')
            #img=pygame.image.load(protanopia_image)
            #pygame.display.flip()

        pygame.init()
        image = pygame.image.load(self.filename)
        base_image = image.convert(image)
        image = pygame.image.load(self.filename)
        image_one = image.convert(image)
        colour_blind_adjustments(base_image, image_one)
        im = cv2.imread("protoanomaly_screen.PNG")
        cv2.imshow('Protanopia Image', im)

    def color2(self):
            #pygame.display.flip()
        def colour_blind_adjustments(base_image, image_two):
            tritanomaly = image_two
            tritanomaly_colour = pygame.PixelArray(tritanomaly)
            image_width, image_height = base_image.get_size()
            for x in range(0, image_width):
                for y in range(0, image_height):
                    red = base_image.get_at((x, y)).r
                    green = base_image.get_at((x, y)).g
                    blue = base_image.get_at((x, y)).b
                    red_channel_red = red * 0.95
                    red_channel_green = green * 0.05
                    red_channel_blue = blue * 0.0
                    green_channel_red = red * 0.0
                    green_channel_green = green * 0.43333
                    green_channel_blue = blue * 0.56667
                    blue_channel_red = red * 0.0
                    blue_channel_green = green * 0.475
                    blue_channel_blue = blue * 0.525
                    red = red_channel_red + red_channel_green + red_channel_blue
                    green = green_channel_red + green_channel_green + green_channel_blue
                    blue = blue_channel_red + blue_channel_green + blue_channel_blue
                    
                    tritanomaly_colour[x, y] = (blue, green, red)
            del tritanomaly_colour
            pygame.image.save(tritanomaly, "tritanomaly.screen.PNG")
        pygame.init()
        #pygame.display.set_caption('tritanopia')
        #screen = pygame.display.set_mode((400, 300))
        surface = pygame.Surface((100, 100), pygame.SRCALPHA)
        image = pygame.image.load(self.filename)
        base_image = image.convert(image)
        image = pygame.image.load(self.filename)
        image_two = image.convert(image)
        colour_blind_adjustments(base_image, image_two)
        image = pygame.image.load(self.filename)
        im = cv2.imread("tritanomaly.screen.PNG")
        cv2.imshow('Tritanopia Image', im)

    def color3(self):

        def colour_blind_adjustments(base_image, image_one):
            #screen = pygame.display.set_mode((400, 300))
            Achromatopsia_Image = image_one
            Achromatopsia_colour = pygame.PixelArray(Achromatopsia_Image)
            image_width, image_height = base_image.get_size()
            for x in range(0, image_width):
                for y in range(0, image_height):
                    red = base_image.get_at((x, y)).r
                    green = base_image.get_at((x, y)).g
                    blue = base_image.get_at((x, y)).b
                    red_channel_red = red * 0.299
                    red_channel_green = green * 0.587
                    red_channel_blue = blue * 0.114
                    green_channel_red = red * 0.299
                    green_channel_green = green * 0.587
                    green_channel_blue = blue * 0.114
                    blue_channel_red = red * 0.299
                    blue_channel_green = green * 0.587
                    blue_channel_blue = blue * 0.114
                    red = red_channel_red + red_channel_green + red_channel_blue
                    green = green_channel_red + green_channel_green + green_channel_blue
                    blue = blue_channel_red + blue_channel_green + blue_channel_blue
                    
                    Achromatopsia_colour[x, y] = (blue, green, red)
            del Achromatopsia_colour
            pygame.image.save(Achromatopsia_Image, "Achromatopsia_screen.PNG")

        pygame.init()
        image = pygame.image.load(self.filename)
        base_image = image.convert(image)
        image = pygame.image.load(self.filename)
        image_one = image.convert(image)
        colour_blind_adjustments(base_image, image_one)
        im = cv2.imread("Achromatopsia_screen.PNG")
        cv2.imshow('Achromatopsia Image', im)


    def asti(self):
        image = cv2.imread(self.filename)
        shift = 5

        for i in range(image.shape[0] -1, image.shape[0] - shift, -1):
            image = np.roll(image, -1, axis=0)
            image[-1, :] = 0
            im = image

        for i in range(image.shape[0] -1, image.shape[0] - shift, -1):
            image = np.roll(image, 1, axis=0)
            image[-1, :] = 0
            ima = image

        #add = im + image
        #add = cv2.add(im, image)

        weighted = cv2.addWeighted(im, 0.5, image, 0.5, 0)
        lol = cv2.addWeighted(weighted, 0.5, ima, 0.5, 0)
        lola = cv2.addWeighted(weighted, 0.5, lol, 0.5, 0)
        blurImg = cv2.blur(lola,(10,10))

        cv2.imshow('ASTIGMATISM', blurImg)
        #cv2.imshow('image', im)


    def pr_blind(self):
        img = cv2.imread(self.filename)
        blurImg = cv2.blur(img,(16,16))
        cv2.imshow('PRACTICAL BLIND',blurImg)


    def reset_but(self):
        img.destroy()
        self.button['state'] = NORMAL

    """
    def red(self):
        im = cv2.imread(self.filename)
        #im= io.imread('1.jpg')
        red_channel = deepcopy(im)
        blue_channel = deepcopy(im)
        green_channel = deepcopy(im)

        red_channel[:,:,1]= 0
        red_channel[:,:,2]= 0

        green_channel[:,:,0]= 0
        green_channel[:,:,2]=0

        blue_channel[:,:,0]= 0
        blue_channel[:,:,1]= 0

        cv2.imshow('RED',blue_channel)

    def blue(self):
        im = cv2.imread(self.filename)
        red_channel = deepcopy(im)
        blue_channel = deepcopy(im)
        green_channel = deepcopy(im)

        red_channel[:,:,1]= 0
        red_channel[:,:,2]= 0

        green_channel[:,:,0]= 0
        green_channel[:,:,2]=0

        blue_channel[:,:,0]= 0
        blue_channel[:,:,1]= 0

        cv2.imshow('BLUE', red_channel)
    def green(self):
        im = cv2.imread(self.filename)
        red_channel = deepcopy(im)
        blue_channel = deepcopy(im)
        green_channel = deepcopy(im)

        red_channel[:,:,1]= 0
        red_channel[:,:,2]= 0

        green_channel[:,:,0]= 0
        green_channel[:,:,2]=0

        blue_channel[:,:,0]= 0
        blue_channel[:,:,1]= 0
        cv2.imshow('GREEN', green_channel)
    """
    def gla(self):
        img = cv2.imread(self.filename)
        re = cv2.resize(img,(500,500))
        mask = cv2.imread('mask.png',0)
        res = cv2.bitwise_and(re,re,mask = mask)
        gla = cv2.blur(res,(3,3))
        cv2.imshow('Glaucoma', gla)

    def torn(self):
        img = cv2.imread(self.filename)
        re = cv2.resize(img,(500,500))
        mask = cv2.imread('mask.png',0)
        res = cv2.bitwise_and(re,re,mask = mask)
        for i in range (0,20):
            cX = random.randint(0, 500)
            cY = random.randint(0, 500)
            j = random.randint(4,20)
            for k in range(0,5):
                cv2.circle(res, (cX, cY), j, (0, 0, 0), -1)
                cv2.circle(res, (cY, cX), j, (0, 0, 0), -1)

        cv2.imshow('TORN RETINA', res)

    """
    !!!I will code this topic with deep dream using tensorflow!!!
    def LSD(self):
        img = cv2.imread(self.filename)
        re = cv2.resize(img,(225,225))
        mask = cv2.imread('lsd.jpeg')
        #res = cv2.bitwise_and(re,re,mask = mask)
        dst = cv2.addWeighted(mask,0.3, re,0.7,0)
        cv2.imshow('LSD',dst)
    """
    """

    def flip(self):
        im= io.imread(self.filename)
        flipVertical = cv2.flip(im, 0)
        flipHorizontal = cv2.flip(im, 1)
        flipBoth = cv2.flip(im, -1)

        fig, ax = plt.subplots(ncols = 2, nrows = 2)

        ax[0,0].imshow(im)
        ax[0,0].text(0.5,1.04, "Original Image", size=12, ha="center", transform = ax[0,0].transAxes)

        ax[0,1].imshow(flipVertical)
        ax[0,1].text(0.5,1.04, "Vertical Flip", size=12, ha="center", transform = ax[0,1].transAxes)

        ax[1,0].imshow(flipHorizontal)
        ax[1,0].text(0.5,-0.25, "Horizontal Flip", size=12, ha="center", transform = ax[1,0].transAxes)

        ax[1,1].imshow(flipBoth)
        ax[1,1].text(0.5,-0.25, "Flip Both", size=12, ha="center", transform = ax[1,1].transAxes)
        plt.show()
        #cv2.imshow("lol",flipHorizontal)
    """

    def alc(self):
        image= cv2.imread(self.filename)
        shift = 7

        for i in range(image.shape[1] -1, image.shape[1] - shift, -1):
            image = np.roll(image, -1, axis=1)
            image[:, -1] = 255
            im_1 = image

        for i in range(image.shape[0] -1, image.shape[0] - shift, -1):
            image = np.roll(image, -1, axis=0)
            image[-1, :] = 255
            im_2 = image

        for i in range(image.shape[1] -1, image.shape[1] - shift, -1):
            image = np.roll(image, 2, axis=1)
            image[:, -1] = 255
            im_3 = image

        for i in range(image.shape[0] -1, image.shape[0] - shift, -1):
            image = np.roll(image, 2, axis=0)
            image[-1, :] = 255
            im_4 = image

        for i in range(image.shape[1] -1, image.shape[1] - shift, -1):
            image = np.roll(image, -1, axis=1)
            image[:, -1] = 255
            im_5 = image

        for i in range(image.shape[0] -1, image.shape[0] - shift, -1):
            image = np.roll(image, -1, axis=0)
            image[-1, :] = 255
            im_6 = image

        q = cv2.addWeighted(im_1, 0.5, im_2, 0.5, 10)
        y = cv2.addWeighted(im_4, 0.5, im_3, 0.5, 10)
        w = cv2.addWeighted(im_5, 0.5, im_6, 0.5, 10)
        all = cv2.addWeighted(q, 0.5, y, 0.5, 10)
        weighted = cv2.addWeighted(all, 0.5, w, 0.5, 10)

        cv2.imshow('ALCOHOL', weighted)

    def AMD(self):
        img = cv2.imread(self.filename)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(gray_image,127,255,0)

        # calculate moments of binary image
        M = cv2.moments(thresh)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # put text and highlight the center
        cv2.circle(img, (cX, cY), 25, (40, 40, 40), -1)
        cv2.circle(img, (cX+20, cY+10), 25, (40, 40, 40), -1)
        cv2.circle(img, (cX-10, cY-20), 30, (40, 40, 40), -1)
        cv2.circle(img, (cX+23, cY-15), 10, (40, 40, 40), -1)
        cv2.circle(img, (cX-15, cY+15), 25, (40, 40, 40), -1)
        # display the image
        cv2.imshow("Aged-Related Macular Degeneration", img)

    def pet(self):
        def colour_blind_adjustments(base_image, image_one):
            pet_image = image_one
            pet_colour = pygame.PixelArray(pet_image)
            image_width, image_height = base_image.get_size()
            for x in range(0, image_width):
                for y in range(0, image_height):
                    red = base_image.get_at((x, y)).r
                    green = base_image.get_at((x, y)).g
                    blue = base_image.get_at((x, y)).b
                    
                    red_channel_red = red * 0.625
                    red_channel_green = green * 0.375
                    red_channel_blue = blue * 0.0
                    green_channel_red = red * 0.5
                    green_channel_green = green * 0.5
                    green_channel_blue = blue * 0.0
                    blue_channel_red = red * 0.0
                    blue_channel_green = green * 0.3
                    blue_channel_blue = blue * 0.7
                    
                    red = red_channel_red + red_channel_green + red_channel_blue
                    green = green_channel_red + green_channel_green + green_channel_blue
                    blue = blue_channel_red + blue_channel_green + blue_channel_blue
                    pet_colour[x, y] = (blue, green, red)
            del pet_colour
            pygame.image.save(pet_image, "Pet_screen.PNG")

        pygame.init()
        image = pygame.image.load(self.filename)
        base_image = image.convert(image)
        image = pygame.image.load(self.filename)
        image_one = image.convert(image)
        colour_blind_adjustments(base_image, image_one)
        im = cv2.imread("Pet_screen.PNG")
        cv2.imshow('Pet Image', im)

    def sek(self):
        image= cv2.imread(self.filename)
        Sketch, sketch2  = cv2.pencilSketch(image, sigma_s=40, sigma_r=0.4, shade_factor=0.02)
        cv2.imshow("Black&White Sketch", Sketch)
        cv2.imshow("Colored Sketch", sketch2)


    def faces(self):
        image = cv2.imread(self.filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]


        status = cv2.imwrite('faces_detected.jpg', image)
        cv2.imshow('faces_detected.jpg', image)

    
    def cart(self):
        image= cv2.imread(self.filename)
        cartoon_image = cv2.stylization(image, sigma_s=100, sigma_r=0.25)
        cv2.imshow("Cartoon", cartoon_image)
        

    """
    def faces(self):
        image = cv2.imread(self.filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

        #status = cv2.imwrite('faces_detected.jpg', image)
        cv2.imshow('faces_detected.jpg', image)
        print("[INFO] Image faces_detected.jpg written to filesystem: ", status)
    """

if __name__ == '__main__':
    root = Root()
    root.mainloop()


