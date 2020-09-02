from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import pandas as pd
import PIL.Image
import PIL.ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pytorch_trainer_v2 as ptt
import nibabel as nib
import os
import sys
sys.setrecursionlimit(5000)
import PIL._tkinter_finder
import six
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cpu")

class MyModel(nn.Module):
    
    def make_conv_block(self, in_channels, out_channels, padding, kernel_size=3):
        layers = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False,
                     ),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
        ]

        return nn.Sequential(*layers)
    
    def make_last_conv_block(self, in_channels, out_channels, padding, kernel_size=3):
        layers = [
            nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                        ),
        ]

        return nn.Sequential(*layers)
      
    def make_upsample_block(self, size, scale_factor, mode='nearest', align_corners=None):
        layers = [
            nn.Upsample(size=size,
                      scale_factor=scale_factor,
                      mode=mode,
                      align_corners=align_corners,
                     ),
            nn.LeakyReLU(),
        ]

        return nn.Sequential(*layers)
      
    

    def __init__(self):
        super(MyModel, self).__init__()
        
        self.max_pool = nn.MaxPool2d(2)
        self.act_func = nn.ReLU()
        self.Softmax = nn.Softmax2d()
        self.conv1 = self.make_conv_block(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = self.make_conv_block(in_channels=64, out_channels=64, kernel_size=3, padding=1)    
        
        self.conv3 = self.make_conv_block(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = self.make_conv_block(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        self.conv5 = self.make_conv_block(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = self.make_conv_block(in_channels=256, out_channels=256, kernel_size=3, padding=1)
                
        self.upsample1 = self.make_upsample_block(size=None, scale_factor=2, mode='bilinear', align_corners=None)
        self.conv7_1x1 = self.make_conv_block(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.conv7 = self.make_conv_block(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv8 = self.make_conv_block(in_channels=128, out_channels=128, kernel_size=3, padding=1)
                                          
        self.upsample2 = self.make_upsample_block(size=None, scale_factor=2, mode='bilinear', align_corners=None)
        self.conv9_1x1 = self.make_conv_block(in_channels=128, out_channels=64, kernel_size=1, padding=0)
        self.conv9 = self.make_conv_block(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv10 = self.make_conv_block(in_channels=64, out_channels=64, kernel_size=3, padding=1)
                                          
        self.conv11 = self.make_last_conv_block(in_channels=64, out_channels=2, kernel_size=1, padding=0)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) 
        xMxP = self.max_pool(x) 
        
        x2 = self.conv3(xMxP) 
        x2 = self.conv4(x2)    
        x2MxP = self.max_pool(x2)

        x3 = self.conv5(x2MxP) 
        x3 = self.conv6(x3) 
                    
        x3 = self.conv7_1x1(x3)
        x4 = self.upsample1(x3)
        x4 = torch.cat((x2, x4), 1) 
        x4 = self.conv7(x4) 
        x4 = self.conv8(x4) 
        
        x4 = self.conv9_1x1(x4)
        x4 = self.upsample2(x4)
        x4 = torch.cat((x, x4), 1) 
        x4 = self.conv9(x4) 
        x4 = self.conv10(x4) 
        x4 = self.conv11(x4) 
        

        return self.Softmax(x4)

class Root(Tk, MyModel):    

    def __init__(self):
        super(Root, self).__init__()
        super(MyModel, self).__init__()
        self.title('HypAST - Hypothalamus Automatic Segmentation Tool')
        self.minsize(690,500)
        
        self.frameEnd(690,98, "white",10,0, 402)        
        self.frame(470,200, "white",20,110,200)
        self.labelFrame('Volume information', 30, 20)
        self.labelFrame('Texture information', 30, 70)
        self.labelFrame('Visualization Tool', 30, 120)
        
        self.clicButton(self, self.fileDialog, "Open File", 50, 50)
        self.clicButton(self, self.prediction, "Run Code ", 50, 90)
        #self.labelVol()
        self.creatCanvasImage()
        self.state = NORMAL
        self.clicButton(self, self.saveFile, "    Save Segmentation    ",  50, 150)       
        self.outlier = False
        self.run = False


    def frame(self, width, heigh,bg, hl, x, y):
        self.f1 = Frame(self, width = width, heigh = heigh, bg = bg, highlightthickness=hl)
        self.f1.place(x = x, y = y)
        
    def frameEnd(self, width, heigh,bg, hl, x, y):
        f = Frame(self, width = width, heigh = heigh, bg = bg, highlightthickness=hl)
        f.place(x = x, y = y)
        
    def labelFrame(self, text, x, y):
        lb_vol = Label(self.f1, text = text, 
                               font = ('courier', 15),
                             fg = 'black',
                            bg = 'white')
        
        lb_vol.place(x = x, y  = y)
        
    def clicButton(self, master, command, text, x, y):
        bt = Button(master = master,text = text, command = command)
        bt.place(x = x, y = y)
        
            
    def creatCanvasImage(self):       
        img = Image.open('logoMIClab.jpg')
        basewidth = 150
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        canvas = Canvas(self, bg = "black", height = hsize, width = 150,bd=0, highlightthickness=0)
        canvas.place(x = 260, y = 412)
        canvas.image = ImageTk.PhotoImage(img)
        canvas.create_image(0,0, image = canvas.image, anchor = 'nw')
        
    def createBatch(self, filename):
        
        self.img_nii = nib.load(filename)
        self.imgArray_raw = np.array(self.img_nii.dataobj).transpose(1,0,2)
        (self.W,self.L,self.C) = self.imgArray_raw.shape
        self.W1 = (self.W-120)/2
        self.L1 = (self.L-100)/2
        self.C1 = (self.C-80)/2
        
        self.imgArray = self.imgArray_raw.reshape(1,self.W,self.L,self.C)[0,int(self.W1):-int(self.W1+0.5),int(self.L1):-int(self.L1+0.5),int(self.C1):-int(self.C1+0.5)]
        
        N = 120

        self.new_image = np.zeros((120,3,100,80))

        for idx in range(120):
            
            if idx%N == 0:
                self.im0 = self.imgArray[idx]

            else:
                self.im0 = self.imgArray[idx-1]

            if idx%N == (N-1):
                self.im1 = self.imgArray[idx]

            else:
                self.im1 = self.imgArray[idx+1]

            self.im = self.imgArray[idx]

            self.new_image[idx,0,...] = self.im0
            self.new_image[idx,1,...] = self.im
            self.new_image[idx,2,...] = self.im1

        self.batch1 = torch.from_numpy(self.new_image[:int(self.W1)]).float()
        self.batch2 = torch.from_numpy(self.new_image[int(self.W1):]).float()
        
        return self.batch1, self.batch2
        
    def fileDialog(self):
        
        self.filez = filedialog.askopenfilenames(initialdir = "/home",title = "Select file",filetypes = (("NIfTI Files","*.nii*"),("all files","*.*")))
        self.lst = list(self.filez)
        
        self.lb = Label(self, text = str(len(self.lst))+" file(s) openned", 
                                fg = "black",)

        self.lb.place(x = 150, y =55)       
                
        if self.run:
            self.lbdone.destroy()
            self.lbsave.destroy()
        
    def soft_dice_loss(y_pred, y_true, epsilon=1e-6):
        pred = y_pred[:,1,...]
        true = y_true.float()
        final_loss = []

        numerator = 2. * torch.sum(pred * true)
        denominator = torch.sum(pred + true)

        return 1 - torch.mean(numerator / (denominator + epsilon)) 
    
    def array2nii(self, filename, out_final):
        
        img_nii2 = nib.load(filename)
        header = img_nii2.header
        affine = header.get_sform()  
        img_arr = np.zeros((self.W,self.L,self.C))
        img_arr[int(self.W1):-int(self.W1+0.5),int(self.L1):-int(self.L1+0.5),int(self.C1):-int(self.C1+0.5)] = out_final
        img_arr = img_arr.transpose(1,0,2)
        seg_nii = nib.Nifti1Image(img_arr, affine)
        
        return seg_nii, img_arr
        
    def histstat(self,filename, mask=[]):
        
        img = nib.load(filename)
        f = np.array(img.dataobj).astype(np.int64)
        
        h = np.bincount(f.ravel())

        hn = 1.0*h/h.sum()
        stats = np.zeros(7)
        n = len(h) # number of gray values
        stats[0] = np.sum(np.arange(n)*hn)                                                   # media
        stats[1] = np.sum(np.power((np.arange(n)-stats[0]),2)*hn)                            # variancia  
        stats[2] = np.sum(np.power((np.arange(n)-stats[0]),3)*hn)/(np.power(stats[1],1.5))   # obliquidade
        stats[3] = np.sum(np.power((np.arange(n)-stats[0]),4)*hn)/(np.power(stats[1],2))-3   # curtose
        stats[4] = -1*(hn*np.log10(hn+np.spacing(1))).sum() 
        stats[5] = np.argmax(hn)      
        stats[6] = np.where(np.cumsum(hn) >= 0.5)[0][0] 

        if stats[1] == 0:
            stats[2] = 0
            stats[3] = 0
    
        return (np.ndarray.tolist(stats))
    
    def posProcessing(self, maskvol, aux):
    
        'Retira falsos positivos analisando o tamanho dos componentes conexos no volume'
        'Mantém apenas o maior CC do volume'

        vol_sum = maskvol.sum(axis = 2).sum(axis = 1)
        t = vol_sum.argmax()
        for i in range(30):
            if vol_sum[t+i] == 0:
                tafter = t+i
                break

        for j in range(30):
            if vol_sum[t-j] == 0:
                tbef = t-j
                break
        
        maskvol[:tbef,...] = 0
        maskvol[tafter:,...] = 0
        if aux == 0:
            return maskvol
        if aux == 1:
            return tbef, tafter
    
    def prediction(self):
        
        model2 = MyModel()
        self.criteria = self.soft_dice_loss
        self.optimizer = optim.Adam(model2.parameters(), lr = 0.001)
        self.exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0.00001)
        
        trainer = ptt.DeepNetTrainer(
            model       = model2, 
            criterion   = self.criteria, 
            optimizer   = self.optimizer,
            lr_scheduler= self.exp_lr_scheduler,
            callbacks   = [
                           ptt.PrintCallback(),
                           ptt.ModelCheckpoint('weights', reset=False, verbose=1)
                          ],
            devname     = device
            )

        trainer.load_state('weights')
        
        self.hypothalamus = {}
        text1 = '0 images done'

        
        for i in range(len(self.lst)):
            
            filename = self.lst[i]
            self.batch1, self.batch2 = self.createBatch(filename)
            
            self.output1 = model2(self.batch1.to(device)).data.cpu().numpy()
            self.output2 =model2(self.batch2.to(device)).data.cpu().numpy()
            self.out = np.concatenate((self.output1, self.output2), axis = 0)
            out_bin = (self.out[:,1,...]>0.98).astype(int) 
            out_final = self.posProcessing(out_bin,0)
            self.seg_nii, self.seg_arr = self.array2nii(filename, out_final)
            
            self.hypothalamus["segmentation{0}".format(i)] = out_final
            self.hypothalamus["volume{0}".format(i)] = out_final.sum()
            self.hypothalamus["nii_segmentation{0}".format(i)] = self.seg_nii
            self.hypothalamus["texture{0}".format(i)] = self.histstat(filename, self.seg_arr)
            self.hypothalamus["seg_arr{0}".format(i)] = self.seg_nii

        self.lbdone = Label(self, text = "Done!!", 
                                bg = "green",
                                fg = "white")
        self.lbdone.place(x = 150, y =95)
        
        self.lbsave = Label(self, text = "Please, save your segmentation files bellow", 
                                fg = "black",
                       )
        
        self.lbsave.place(x = 50, y =130)
        
        self.run = True
    
    
    def volumeInformation(self):     
        
        
        self.window2 = Tk()
        self.window2.minsize(1000,500)
        self.window2.title('Volume Information')
        self.framevol = Frame(self.window2, width = 770, heigh = 500, bg = 'white')
        self.framevol.place(x = 420, y = 10)
        
        
        self.vol = []
        
        self.tvv = ttk.Treeview(self.window2, columns = ('Volume'))
        self.tvv.heading('#0', text='FileName')
        self.tvv.heading('#1', text='Volume (mm3)')
        self.tvv.place(x = 10,y=10)
        
        for i in range(len(self.lst)):
            aux_v = self.hypothalamus["volume"+str(i)]
            self.vol.append(aux_v)
            self.tvv.insert('', 'end', text=self.name_file+' '+str(i+1),
                         values=(aux_v))            

        self.x_vol = np.arange(len(self.lst))
                
        
            
#             outliersVizButton = Button(self.window2,text ='Display Outliers ID', command = outliersID)
#             outliersVizButton.place(x = 10, y = 350)
            
        boxplotButton = Button(self.window2, text ='Boxplot Graph', command = self.boxplot)
        boxplotButton.place(x = 10, y = 320)
        
        dispersionButton = Button(self.window2,text ='Dispersion Graph', command = self.dispersionGraph)
        dispersionButton.place(x = 10, y = 270)
        
        
    def dispersionGraph(self):
            
        median = np.ones(len(self.lst))*np.median(self.vol)
        avg = np.ones(len(self.lst))*np.average(self.vol)
        x = np.arange(len(self.lst))
        
        fig_vol_disp,ax = plt.subplots(1,1, figsize = (6,4))
        subjects = ax.scatter(self.x_vol,np.asarray(self.vol))
        mean = ax.plot(x,avg, '--',color = 'r')
        median = ax.plot(x,median,color ='y')
        ax.set_title('Dispersion Graph')      
        ax.set_ylabel('Volume (in pixels)')
        ax.set_xlabel('Subject ID')
        ax.legend(['mean', 'median', 'subjects'])  
        
        self.canvas_volume = FigureCanvasTkAgg(fig_vol_disp, master=self.framevol)
        self.canvas_volume.draw()
        self.canvas_volume.get_tk_widget().place(x = 0, y=0)
        toolbar = NavigationToolbar2Tk(self.canvas_volume, self.framevol).place(x=0,y=450)
#             toolbar.update()

    def boxplot(self):
                        
        figplot,ax1 = plt.subplots(1, 1, figsize=(6, 4))
        bplot = ax1.boxplot(self.vol, vert=True)
        ax1.plot(1,np.average(self.vol), color='yellow', marker='*', markeredgecolor='k')
        ax1.set_title('Boxplot of segmented cases')
        med = bplot['medians'][0].get_data()[1][0]
        ax1.text(1.1, np.average(self.vol), 'MEAN = '+str(np.round(np.average(self.vol))), color='red', fontsize=9)
        ax1.text(1.1, med, 'MEDIAN = '+str(med), color='red', fontsize=9)
        outliers_area =bplot['fliers'][0].get_data()[1]
        self.outID = []
        if outliers_area.sum() != 0:
            for i in range (0,outliers_area.size):
#                     print('area', outliers_area)
                (outliers_id,) = np.where(self.vol == outliers_area[i])
                ax1.text(1.01, outliers_area[i], 'ID = '+str(outliers_id[0]+1), color='red', fontsize=9)
                self.outID.append(outliers_id[0])

        self.canvas_volume = FigureCanvasTkAgg(figplot, master=self.framevol)
        self.canvas_volume.draw()
        self.canvas_volume.get_tk_widget().pack()
        self.canvas_volume.get_tk_widget().place(x = 0, y = 0)
        toolbar = NavigationToolbar2Tk(self.canvas_volume, self.framevol).place(x=0,y=450)
#             toolbar.update()
#             self.canvas_volume._tkcanvas.place(x = 0, y= 0)#.pack(side = TOP, fill = BOTH, expand = True)

        outliersButton = Button(self.window2,text ='Display Outliers ID', command = self.outliersID)
        outliersButton.place(x = 10, y = 370)  
        
        
    def outliersID(self):

        if self.outID == []:
            labelOut = Label(self.window2, text = 'No outliers', fg = 'white', bg = 'green')
            labelOut.place(x = 200, y = 375)
        else: 
            self.outIDlist = Listbox(self.window2, width=12, height=7, selectmode = BROWSE)
            self.outIDlist.insert(END, " Outliers ") 
            for line in range(0,len(self.outID)):
                self.outIDlist.insert(END, str(self.outID[line]+1))

            self.outIDlist.place(x = 250, y = 270)
            self.outIDlist.configure(justify=CENTER)
            
            outliersVizButton = Button(self.window2,text ='Visualize', command = self.visualizeOutlier)
            outliersVizButton.place(x = 247, y = 370)
    
    def visualizeOutlier(self):
        
        self.outlier = True
        all_items = self.outIDlist.get(0, END)
        self.clic = self.outIDlist.curselection()
        (self.index_out,) = self.clic
        self.index = int(all_items[self.index_out])
        self.visualizeSegmentation()

    def textureInformation(self):
        
        self.window3 = Tk()
        self.window3.minsize(100,250)
        self.window3.title('Texture Information')
        
        self.tvt = ttk.Treeview(self.window3, columns = ('Mean', 'Variance', 'Skewness', 
                                                        'Kurtosis', 'Entropy', 'Mode', 'Median'))
        self.tvt.heading('#0', text='FileName')
        self.tvt.column("#0",minwidth=0,width=100, stretch=NO)
        
        self.tvt.heading('#1', text='Mean')
        self.tvt.column("#1",minwidth=0,width=100, stretch=NO)
        
        self.tvt.heading('#2', text='Variance')
        self.tvt.column("#2",minwidth=0,width=100, stretch=NO)
        
        self.tvt.heading('#3', text='Skewness')
        self.tvt.column("#3",minwidth=0,width=100, stretch=NO)
        
        self.tvt.heading('#4', text='Kurtosis')
        self.tvt.column("#4",minwidth=0,width=100, stretch=NO)
        
        self.tvt.heading('#5', text='Entropy')
        self.tvt.column("#5",minwidth=0,width=100, stretch=NO)
        
        self.tvt.heading('#6', text='Mode')
        self.tvt.column("#6",minwidth=0,width=100, stretch=NO)
        
        self.tvt.heading('#7', text='Median')
        self.tvt.column("#7",minwidth=0,width=100, stretch=NO)
        
        self.tvt.pack()
        for i in range(len(self.lst)):
            self.tvt.insert('', 'end', text=self.name_file+' '+str(i+1),
                         values=(np.round(self.hypothalamus["texture"+str(i)][1:]))) 
    
    def plus(self):
        self.aux+=3
        self.visImage()

    def minus(self):
        self.aux-=3
        self.visImage()
        
    def loadImage(self):    
        
            
        if not self.outlier:
            self.clic = self.listbox.curselection()
            (self.index,) = self.clic

        self.index = self.index-1
        filename = self.lst[self.index]
        self.segmentation_raw = self.hypothalamus["segmentation"+str(self.index)]
        self.seg_arr = np.array(self.hypothalamus["seg_arr"+str(self.index)].dataobj) 
        self.img = nib.load(filename)
#         self.f = np.array(self.img.dataobj)
        self.lbload = Label(self.window4, text = "Loaded ID "+str(self.index+1), 
                                fg = "black")
        self.lbload.place(x = 30, y = 230)
        self.ax()
        self.outlier = False
        load = True

    def sag(self):

        self.seg_arr_transp = self.seg_arr.transpose(2,1,0)[:,::-1,:]
        self.segmentation_transp = self.segmentation_raw.transpose(1,0,2).transpose(2,1,0)[:,::-1,:]
        self.tbef, self.tafter = self.posProcessing(self.segmentation_transp, 1) 
        self.f_transp = np.array(self.img.dataobj).transpose(2,1,0)[:,::-1,:]
        self.tbefnii = self.tbef+int(self.C1)
        self.visImage()

    def cor(self):

        self.seg_arr_transp = self.seg_arr[:,::-1,:]
        self.segmentation_transp = self.segmentation_raw.transpose(1,0,2)[:,::-1,:]
        self.tbef, self.tafter = self.posProcessing(self.segmentation_transp, 1) 
        self.f_transp = np.array(self.img.dataobj)[:,::-1,:]
        self.tbefnii = self.tbef+int(self.L1)
        self.visImage()

    def ax (self):

        self.seg_arr_transp = self.seg_arr.transpose(1,0,2)
        self.segmentation_transp = self.segmentation_raw
        self.tbef, self.tafter = self.posProcessing(self.segmentation_transp, 1) 
        self.f_transp = np.array(self.img.dataobj).transpose(1,0,2)
        self.tbefnii = self.tbef+int(self.W1)
        self.visImage()

    def visImage(self):        

        self.alpha = self.slider.get()/100
        fig, ax = plt.subplots(1,3, figsize = (9,6))
        self.mask = np.ma.masked_where(self.seg_arr_transp == 0, self.seg_arr_transp)
        for i in range(3):
            ax[i].imshow(self.f_transp[(self.tbefnii+i+self.aux),...], cmap = 'gray')
            ax[i].imshow(self.mask[(self.tbefnii+i+self.aux),...].astype(np.uint8), 'Wistia', interpolation='none', alpha = self.alpha)
            ax[i].set_title('slice '+str(self.tbefnii+i+self.aux))

        canvas = FigureCanvasTkAgg(fig, master=self.frameimg)
        canvas.draw()
        canvas.get_tk_widget().pack()
        canvas.get_tk_widget().place(x=0, y=0)
#             canvas._tkcanvas.place(x=0, y=0)
        toolbar = NavigationToolbar2Tk(canvas, self.frameimg)
        toolbar.place(x = 0, y = 550)
        toolbar.update() 
            
    def visualizeSegmentation(self):
        
        self.aux = 2
        self.alpha = 0
        self.window4 = Tk()
        self.window4.minsize(1100,600)
        self.window4.title('Visualization')
        self.frameimg = Frame(self.window4, width = 900, heigh = 600, bg = 'white')     
        self.frameimg.place(x = 200, y = 0)
        
        self.lbVis = Label(self.window4, text = "   Segmented Files  ", 
                                    bg = "white",
                                    fg = "black")
        
        self.lbVis.place(x = 30, y = 30)
        
        self.listbox = Listbox(self.window4, selectmode = BROWSE)
        self.listbox.insert(END, " File Names ") 
        
        for i in range(len(self.lst)):
            self.listbox.insert(END, ' '+self.name_file+' '+str(i+1))            
        self.listbox.place(x = 30, y = 50)
                              
        self.buttonLoad = Button(self.window4, text = "Load Image", command = self.loadImage) 
        self.buttonLoad.place(x = 30, y = 200)
        
        self.buttonVizAx = Button(self.window4, text = "Ax", command = self.ax) 
        self.buttonVizAx.place(x = 30, y = 260)

        self.buttonVizCor = Button(self.window4, text = "Cr", command = self.cor) 
        self.buttonVizCor.place(x = 83, y = 260)

        self.buttonVizSg = Button(self.window4, text = "Sg", command = self.sag) 
        self.buttonVizSg.place(x = 133, y = 260)
        
        self.buttonPassPlus = Button(self.window4, text = "+", command = self.plus) 
        self.buttonPassPlus.place(x = 111, y = 480)
        
        self.buttonPassMinus = Button(self.window4, text = "-", command = self.minus) 
        self.buttonPassMinus.place(x = 61, y = 480)
        
        self.slider = Scale(self.window4, orient = HORIZONTAL, length = 150, width = 20, sliderlength = 10)#,command = self.visImage)
        self.slider.place(x = 25, y = 350)#.place(x = 300, y = 100)
                
        self.buttonOverl = Button(self.window4, text = "Ok", command = self.visImage) 
        self.buttonOverl.place(x = 83, y = 400)
        
        self.lbnavig = Label(self.window4, text = "View slices",
                                    bg = "white",
                                    fg = "Black")
        self.lbnavig.place(x = 65, y = 460)
        
        self.lbseg = Label(self.window4, text = "Segmentation transparency",
                                   bg = "white", 
                                   fg = "Black")
                                    
        self.lbseg.place(x = 10, y = 330)
        
        
        if self.outlier:
            self.loadImage()
            self.ax()
        
        
    def saveFile(self):
        
        self.path_save = filedialog.asksaveasfilename(initialdir = "/home", title = "Select path",
        filetypes = (("NIfTI Files","*.nii"),("all files","*.*")))
        head_tail = os.path.split(str(self.path_save))
        
        self.path_file = head_tail[0]
        self.name_file = head_tail[1].split('.')[0]
        
        for i in range(len(self.lst)):
            self.save = self.hypothalamus["segmentation"+str(i)]
            self.save_nii = self.hypothalamus["nii_segmentation"+str(i)]
            np.save(self.path_file+'/'+self.name_file+str(i+1), self.save)
            nib.save(self.save_nii, self.path_file+'/'+self.name_file+str(i+1)+'.nii')
            
        vol_info = []
        text_info = []
        for i in range(len(self.lst)):
            self.unit_vol = ([self.name_file+str(i+1), self.hypothalamus["volume"+str(i)]])
            self.txt_list = self.hypothalamus["texture"+str(i)]
            item = self.name_file+str(i+1)
            self.txt_list.insert(0,item)
            vol_info.append(self.unit_vol)
            text_info.append(self.txt_list)
            
        self.dfObjv = pd.DataFrame(vol_info)     
        self.dfObjv = pd.DataFrame(vol_info, columns = ['FileName' , 'Total Volume']) 
        self.dfObjv.to_csv(self.path_file+'/'+self.name_file+'_volumeInfo.csv', index = None, header=True)    
        
        self.dfObjt = pd.DataFrame(text_info)     
        self.dfObjt = pd.DataFrame(text_info, columns = ['FileName' , 'Mean', 'Variance', 'Skewness', 
                                                        'Kurtosis', 'Entropy', 'Mode', 'Median']) 
        
        
        self.dfObjt.to_csv(self.path_file+'/a'+self.name_file+'_TextureInfo.csv', index = None, header=True)    
        
        self.lb3 = Label(self, text = self.path_file+'/'+self.name_file, 
                                fg = "black") 
        self.lb3.place(x = 260, y =155)
        
        
        self.clicButton(self.f1, self.volumeInformation, " Volume  ", 310, 20)
        self.clicButton(self.f1, self.textureInformation, " Texture ", 310, 70)
        self.clicButton(self.f1,self.visualizeSegmentation, "Visualize", 310, 120)


root = Root()
root.mainloop()
