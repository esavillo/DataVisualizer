'''
# Evan Savillo
# 04/11/16
'''

import Tkinter as tk
import tkFont as tkf
import tkFileDialog
import tkMessageBox
import math
import scipy.stats as ss
import numpy
import view
import data
import analysis
import time

# create a class to build and manage the display
class DisplayApp:

    def __init__(self, width, height, parent=None, screen=None):
        self.screen = screen

        self.saved_anly = {} # Holds saved analyses, such as linear regressions
        self.anly_counter = [0,0]
        self.export_counter = 0
        self.saveQ = [] # holds anything that needs to be saved for just a second
        self.saveQ2 = [] # extra safe
        self.objects = [] # list of data objects that will be drawn in the canvas
        self.obj2info = {} # Dict linking canvas objects to data info
        #                                       {point:x, y, z, info_size, info_color}

        self.dx = 2 # initial size of created points

        # create a tk object, which is the root window
        self.root = tk.Tk()

        self.K = 1 # intial K number of clusters
        self.Kvar = tk.StringVar(self.root)
        self.Kvar.set('K: '+str(self.K)) # will eventually condense
        self.krgbs = [] # colors for each cluster
        self.metric = tk.StringVar(self.root)
        self.metric.set('sum2ed')

        # current position of mouse var
        self.mouseoverinfo = tk.StringVar(self.root)
        self.mouseoverinfo.set('...')

        # width and height of the window
        self.initDx = width
        self.initDy = height

        # set up the geometry for the window
        self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

        self.root.title("An Extra-Dimensional Domain of Data Wizard Gumby's")

        # set the maximum size of the window for resizing
        self.root.maxsize( 1600, 900 )

        self.data = data.Data('data.csv')

        # Sets which columns to plot on which dimensions
        self.axes_map = [tk.StringVar(self.root),tk.StringVar(self.root),
                         tk.StringVar(self.root),tk.StringVar(self.root),
                         tk.StringVar(self.root)] # x, y, z, size, color
        for i, header in enumerate(self.data.get_headers()):
            self.axes_map[i].set(header)

        # setup the menus
        self.buildMenus()

        # Make dict for strVars and build info display, their home, at bottom
        self.orinfo = {'Scale:':tk.StringVar(self.root),
                       'Pitch:':tk.StringVar(self.root),
                       'Yaw:':tk.StringVar(self.root)}
        self.buildInfoDisplay()

        # build the controls
        self.buildControls()

        self.buildLegend()

        # build the Canvas
        self.buildCanvas(parent)

        # bring the window to the front
        self.root.lift()

        # - do idle events here to get actual canvas size
        self.root.update_idletasks()

        # now we can ask the size of the canvas
        print self.canvas.winfo_geometry()

        # set up the key bindings
        self.setBindings()

        # set up the application state
        self.baseClick = None # used to keep track of mouse movement

        self.distr = ['Uniform', 'Uniform']
        self.user_c = [1.0, 1.0, 1.0] # 0:Pan, 1:Scale, 2:Rotate

        # Setting up view, axes, et cetera
        self.view = view.View(screen=self.screen)
        self.axes_ep = numpy.matrix( [[0,0,0,1],
                                      [1,0,0,1],
                                      [0,1,0,1],
                                      [0,0,1,1]] )

        self.axes_gfx = [] # Holds the actual canvas objects
        self.axes_labels = []
        self.buildAxes()


        # code for Data shall follow
        self.d_points = None # shall be a numpy matrix
        self.d_sizedim = None
        self.d_colordim = None

        # Regression etc below
        self.anal_frame = None
        self.lr_ep = None # numpy matrix
        self.lr_gfx = []

        self.last_pca = None
        self.olddata = None

    def buildAxes(self):
        vtm = self.view.build()
        pts = (vtm * self.axes_ep.T).T
        xaxis = (pts[0].tolist()[0][:-2], pts[1].tolist()[0][:-2])
        yaxis = (pts[0].tolist()[0][:-2], pts[2].tolist()[0][:-2])
        zaxis = (pts[0].tolist()[0][:-2], pts[3].tolist()[0][:-2])
        self.axes_gfx = [self.canvas.create_line(xaxis,
                                                 fill="red", arrow=tk.LAST, width=2),
                         self.canvas.create_line(yaxis,
                                                 fill="green", arrow=tk.LAST, width=2),
                         self.canvas.create_line(zaxis,
                                                 fill="blue", arrow=tk.LAST, width=2)]

        self.axes_labels = [self.canvas.create_text(xaxis[1], text='X-AXIS',
                                                    anchor=tk.SE),
                            self.canvas.create_text(yaxis[1], text='Y-AXIS',
                                                    anchor=tk.SE),
                            self.canvas.create_text(zaxis[1], text='Z-AXIS',
                                                    anchor=tk.SE)]
        #note to self: switchboard ^

    def updateAxes(self):
        vtm = self.view.build()
        pts = (vtm * self.axes_ep.T).T
        xaxis = (pts[0].tolist()[0][:-2], pts[1].tolist()[0][:-2])
        yaxis = (pts[0].tolist()[0][:-2], pts[2].tolist()[0][:-2])
        zaxis = (pts[0].tolist()[0][:-2], pts[3].tolist()[0][:-2])
        self.canvas.coords(self.axes_gfx[0], xaxis[0][0], xaxis[0][1],
                           xaxis[1][0], xaxis[1][1])
        self.canvas.coords(self.axes_gfx[1], yaxis[0][0], yaxis[0][1],
                           yaxis[1][0], yaxis[1][1])
        self.canvas.coords(self.axes_gfx[2], zaxis[0][0], zaxis[0][1],
                           zaxis[1][0], zaxis[1][1])

        self.canvas.coords(self.axes_labels[0], xaxis[1][0], xaxis[1][1])
        self.canvas.coords(self.axes_labels[1], yaxis[1][0], yaxis[1][1])
        self.canvas.coords(self.axes_labels[2], zaxis[1][0], zaxis[1][1])

        if len(self.objects) > 0:
            self.updatePoints()

        if len(self.lr_gfx) > 0:
            lr_pts = (vtm * self.lr_ep.T).T
            self.canvas.coords(self.lr_gfx[0],
                               lr_pts[0,0], lr_pts[0,1],
                               lr_pts[1,0],lr_pts[1,1])

    # Takes in a list of headers, deletes any existing data on canvas,
    # and then creates a new set for the current plot.
    def buildPoints(self, headers):
        if len(self.lr_gfx) > 0:
            self.handleClear()
        else:
            self.handleClear(lr_clear=True)

        pre_rawinfo = [] # Stores raw info for mouseover

        # Creates dpoints (from scratch)
        self.d_points = numpy.matrix([[]])
        self.d_points = numpy.resize(self.d_points, (0, self.data.get_raw_num_rows()))
        for i, header in enumerate(headers):
            if len(self.d_points) == 3:
                self.d_points = numpy.concatenate((self.d_points,
                                        data.generateColumn(self.data.get_raw_num_rows(),
                                                            1)))
                pre_rawinfo += [None]
            if i < 3:
                if header == '':
                    self.d_points = numpy.concatenate((self.d_points,
                                        data.generateColumn(self.data.get_raw_num_rows(),
                                                            0)))
                    pre_rawinfo += [None]
                else:
                    # Normalizes data here as well
                    self.d_points = numpy.concatenate((self.d_points,
                            analysis.normalize_columns_separately(self.data, [header])))
                    pre_rawinfo += [self.data.get_data([header])]

            elif header != '':
                if i == 3:
                    self.d_sizedim = numpy.matrix(
                            analysis.normalize_columns_separately(self.data, [header]))
                elif i == 4:
                    self.d_colordim = numpy.matrix(
                            analysis.normalize_columns_separately(self.data, [header]))
                    break

        # some more junk to build raw list
        if headers[3] != '':
            pre_rawinfo += [self.data.get_data([headers[3]])]
        else:
            pre_rawinfo += [None]

        if headers[4] != '':
            pre_rawinfo += [self.data.get_data([headers[4]])]
        else:
            pre_rawinfo += [None]


        # Transforms according to the VTM
        vtm = self.view.build()
        pts = (vtm * self.d_points).T

        size = 0.5
        fillcolor = 'default'
        for i,row in enumerate(pts):
            # First deal with raw stuff
            rawinfo = []
            for j in range( len(headers) ):
                if pre_rawinfo[j] is None:
                    rawinfo += [None]
                else:
                    rawinfo += [ pre_rawinfo[j][0,i] ]

            if headers[3] != '':
                size = self.d_sizedim[0,i]
            if headers[4] != '':
                fillcolor = self.d_colordim[0,i]

            self.createPoint(x=row[0,0], y=row[0,1],
                             size=size, fillcolor=fillcolor, rawinfo=rawinfo)

    # Updates points
    def updatePoints(self):
        vtm = self.view.build()
        pts = (vtm * self.d_points).T

        i = 0
        for object in self.objects:
            self.movePoint(object, pts[i,0], pts[i,1])
            i+=1

        self.updateLegend()

    def buildMenus(self):
        
        # create a new menu
        menu = tk.Menu(self.root)

        # set the root menu to our new menu
        self.root.config(menu = menu)

        # create a variable to hold the individual menus
        menulist = []

        # create a file menu
        filemenu = tk.Menu( menu )
        menu.add_cascade( label = "File", menu = filemenu )
        menulist.append(filemenu)

        # create another menu for kicks
        cmdmenu = tk.Menu( menu )
        menu.add_cascade( label = "Command", menu = cmdmenu )
        menulist.append(cmdmenu)

        # create analysis menu
        analmenu = tk.Menu( menu )
        menu.add_cascade(label='Analysis', menu=analmenu)
        menulist.append(analmenu)

        # menu text for the elements
        # the first sublist is the set of items for the file menu
        # the second sublist is the set of items for the option menu
        menutext = [
            ['Open   \xE2\x8C\x98O', 'Clear   \xE2\x8C\x98N',
             '-', 'Quit   \xE2\x8C\x98Q'],
            ['User Preferences', 'Data Sheet   \xE2\x8C\x98D'],
            ['Linearly Regress', 'PCA Analysis', 'Cluster',
             '-', 'Analysis Viewer   \xE2\x8C\x98V']]

        # menu callback functions (note that some are left blank,
        # so that you can add functions there if you want).
        # the first sublist is the set of callback functions for the file menu
        # the second sublist is the set of callback functions for the option menu
        menucmd = [
            [self.handleOpen, self.handleClear,
             None, self.handleQuit],
            [self.handleDlog2, self.handleDlog4],
            [self.handleLinearRegression, self.handlePCAAnalysis,
                                                            self.handleNewClusterAnalysis,
             None, self.handleAnalysisViewer]]
        
        # build the menu elements and callbacks
        for i in range( len( menulist ) ):
            for j in range( len( menutext[i]) ):
                if menutext[i][j] != '-':
                    menulist[i].add_command( label = menutext[i][j],
                                             command=menucmd[i][j] )
                else:
                    menulist[i].add_separator()

    # create the canvas object
    def buildCanvas(self, parent=None):
        if parent is not None:
            theroot = parent
        else:
            theroot = self.root

        self.canvas = tk.Canvas( theroot, width=self.initDx, height=self.initDy,
                                 cursor='gumby')
        self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
        return

    # build a frame and put controls in it
    def buildControls(self):

        ### Control ###
        # make a control frame on the right
        rightcntlframe = tk.Frame(self.root)
        rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

        # make a separator frame
        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

        # use a label to set the size of the right panel
        label = tk.Label( rightcntlframe, text="Control Panel", width=20 )
        label.pack( side=tk.TOP, pady=10 )

        # make a menubutton
        self.colorOption = tk.StringVar( self.root )
        self.colorOption.set("black")
        colorMenu = tk.OptionMenu( rightcntlframe, self.colorOption, 
                                        "black", "blue", "red", "green" ) # can add a
        #                                                             command to the menu
        colorMenu.pack(side=tk.TOP)

        # make a button in the frame
        # and tell it to call the handleButton method when it is pressed.
        button = tk.Button( rightcntlframe, text="Update Color",
                               command=self.handleButton1 )
        button.pack(side=tk.TOP)  # default side is top

        # makes plot data button
        sep2 = tk.Frame( rightcntlframe, width=150,
                         height=2, bd=1, relief=tk.SUNKEN )
        sep2.pack(side=tk.TOP, padx=2, pady=10, fill=tk.Y)

        button2_frame = tk.Frame( rightcntlframe )
        button2_frame.pack(side=tk.TOP, pady=2, fill=tk.X)

        button2_options = tk.Button(button2_frame, command=self.handleDlog3,
                                    bitmap='questhead')
        button2_options.pack(side=tk.RIGHT)
        button2 = tk.Button( button2_frame, text="Plot Data",
                                command=self.handlePlotData )
        button2.pack(side=tk.RIGHT, padx=5)

        # makes shape selector
        sep3 = tk.Frame( rightcntlframe, width=150,
                         height=2, bd=1, relief=tk.SUNKEN )
        sep3.pack(side=tk.TOP, padx=2, pady=10, fill=tk.Y)
        self.shapebox = tk.Listbox(rightcntlframe, selectmode=tk.SINGLE,
                                   exportselection=0, height=3)
        self.shapebox.pack()
        for item in ["Circle", "Square", "Triangle"]:
            self.shapebox.insert(tk.END, item)
        self.shapebox.select_set(0)
        self.shapebox.event_generate("<<ListboxSelect>>")

        # reset button
        sep4 = tk.Frame( rightcntlframe, width=150, height=2, bd=1, relief=tk.SUNKEN )
        sep4.pack(side=tk.TOP, padx=2, pady=10, fill=tk.Y)
        button3 = tk.Button( rightcntlframe, text="Reset View",
                             command=self.handleButton3 )
        button3.pack(side=tk.BOTTOM, pady=2)

        # point size
        self.spin1 = tk.Spinbox( rightcntlframe, command=self.handleSpin1, wrap=True,
                                 from_=3, to=10, increment=1, state='readonly' )
        self.spin1['from_'] = 1
        self.spin1.pack(side=tk.TOP, pady=2)
        self.spin1.bind('<Return>', func=lambda Q:self.root.focus_set())


    # Build Info display at bottom
    def buildInfoDisplay(self):
        self.orinfo['Scale:'].set('100.0')
        self.orinfo['Pitch:'].set('0.0')
        self.orinfo['Yaw:'].set('180.0')

        self.btmcntlframe = tk.Frame(self.root)
        self.btmcntlframe.pack(side=tk.BOTTOM, padx=2, pady=2, fill=tk.X)

        sep1 = tk.Frame(self.root, height=2, width=self.initDx, bd=1, relief=tk.SUNKEN)
        sep1.pack(side=tk.BOTTOM, fill=tk.X)

        info = ['Pitch:', 'Yaw:', 'Scale:']
        for item in info:
            l1 = tk.Label(self.btmcntlframe, text=item)
            l1.pack(side=tk.LEFT, padx=5)
            l2 = tk.Entry(self.btmcntlframe, textvariable=self.orinfo[item],
                          relief=tk.RIDGE, width=7, justify=tk.CENTER)
            l2.pack(side=tk.LEFT)

            l2.bind('<Return>', self.handleOrientInput)

        m1 = tk.Label(self.btmcntlframe, textvariable=self.mouseoverinfo)
        m1.pack(side=tk.RIGHT, padx=3)

    # Builds legend
    def buildLegend(self):
        self.panel = tk.Frame(self.root)
        self.panel.pack(side=tk.LEFT, padx=2, pady=2, fill=tk.Y)
        sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
        sep.pack(side=tk.LEFT, padx = 2, pady = 2, fill=tk.Y)

        legend = tk.LabelFrame(self.panel, text='Legend', labelanchor=tk.N)
        legend.pack(side=tk.TOP, padx = 2, pady = 2, fill=tk.X)

        self.legendcanvi = []
        for i, dim in enumerate(['X-Axis:', 'Y-Axis:', 'Z-Axis:', 'Size:', 'Color:']):
            label = tk.Label(legend, textvariable=self.axes_map[i], width=8)
            label.grid(row=i, column=1)

            canvas = tk.Canvas(legend, width=120, height=40,
                               bg='#E4E4E4', relief=tk.GROOVE)
            canvas.grid(row=i, column=0)
            self.legendcanvi+=[canvas]

        self.legendcanvi[0].create_line(20,23,105,23, fill="red", arrow=tk.LAST)

        self.legendcanvi[1].create_line(20,23,105,23, fill="green", arrow=tk.LAST)

        self.legendcanvi[2].create_line(20,23,105,23, fill="blue", arrow=tk.LAST)

        self.updateLegend()

    # updates legend by just building part of it over
    def updateLegend(self):
        for i in self.legendcanvi[3].find_all():
            self.legendcanvi[3].delete(i)
        for i in self.legendcanvi[4].find_all():
            self.legendcanvi[4].delete(i)

        for i in range(11):
            xoffset = (105-20)*(i/10.0) # x-offset
            soffset = float('0.'+str(i)) # secondary-offset
            if i == 10:
                soffset = 1.0

            self.createPoint(20+xoffset, 23, size=soffset, legend=3)
            self.createPoint(20+xoffset, 23, fillcolor=soffset, legend=4)


    # Creates line object showing the lin reg fit
    def buildLinearRegression(self, lr_headers):
        self.saveQ += [self.getAxesMap()]
        self.saveQ += [self.d_sizedim]
        self.saveQ += [self.d_colordim]
        self.saveQ += [self.d_points]

        for i in range( 5-len(lr_headers) ):
            lr_headers += ['']
        self.setAxesMap(lr_headers)
        self.buildPoints(lr_headers)

        lr = self.calcLR(lr_headers) # 0->4; slope, intercept, r_value, p_value, std_err
        lr_ranges = analysis.data_range(self.data, lr_headers[0:2])

        y1 = ((lr_ranges[0][0] * lr[0] + lr[1]) - lr_ranges[1][0])/(lr_ranges[1][1] -
                                                                    lr_ranges[1][0])
        y2 = ((lr_ranges[0][1] * lr[0] + lr[1]) - lr_ranges[1][0])/(lr_ranges[1][1] -
                                                                    lr_ranges[1][0])
        self.lr_ep = numpy.matrix( [[0.0,y1,0,1],
                                    [1.0,y2,0,1],
                                    [0,0,0,1],
                                    [0,0,0,1]] )

        vtm = self.view.build()
        lr_pts = (vtm * self.lr_ep.T).T

        self.lr_gfx = [self.canvas.create_line(lr_pts[0,0], lr_pts[0,1],
                                               lr_pts[1,0], lr_pts[1,1],
                                               fill='purple')]

        # Builds Linear Regression Info Square
        self.buildAnalysis_info('LR', lr)

        self.saveQ2 += ['LR', lr_headers, self.data, lr]

    # Re-sets the primary points and whatnot after analysis closes
    def regenPrimaryJunk(self):
        # for Linear Regression
        if len(self.saveQ) > 0:
            self.setAxesMap(self.saveQ.pop(0))
            self.d_sizedim = self.saveQ.pop(0)
            self.d_colordim = self.saveQ.pop(0)
            self.d_points = self.saveQ.pop(0)

        self.saveQ = []

        # for K Clustering
        if self.olddata is not None:
            self.K = 1
            self.krgbs = []
            self.data = self.olddata
            self.olddata = None
            self.OpenFile(self.data.filename)


    # Builds Lin regression info pane
    def buildAnalysis_info(self, type, info=None):
        if self.anal_frame is not None:
            self.anal_frame.destroy()
            self.saveQ2 = []

        if type == 'LR':
            self.anal_frame = tk.LabelFrame(self.panel,
                                            text='Linear Regression', labelanchor=tk.N)
            self.anal_frame.pack(side=tk.TOP, pady=10, fill=tk.X)
            labels = ['Slope: ', 'Intercept: ', 'R^2_Value: ', 'P_Value: ', 'Std_Err: ']

            info[2] = info[2] ** 2 # squares r-value
            for i,info in enumerate(info):
                text = labels[i] + str(info)
                label = tk.Label(self.anal_frame, text=text)
                label.pack(side=tk.TOP, padx=2)

        elif type == 'PCA':
            self.anal_frame = tk.LabelFrame(self.panel,
                                            text='Principal Components Analysis',
                                            labelanchor=tk.N)
            self.anal_frame.pack(side=tk.TOP, pady=10, fill=tk.X)

        elif type == 'Cluster':
            self.anal_frame = tk.LabelFrame(self.panel,
                                            text='Cluster',
                                            labelanchor=tk.N)
            self.anal_frame.pack(side=tk.TOP, pady=10, fill=tk.X)

        # bit of space
        tk.Frame(self.anal_frame).pack(side=tk.TOP, pady=5)

        if type == 'PCA':
            PCAbutton = tk.Button(self.anal_frame, text='View PCA Table',
                                  command=self.buildPCAAnalysis)
            PCAbutton.pack(side=tk.TOP, padx=2)

        elif type == 'Cluster':
            self.Kvar.set('K: '+str(self.K))
            tk.Message(self.anal_frame, textvar=self.Kvar).pack(side=tk.TOP, padx=2)

            metricMenu = tk.OptionMenu(self.anal_frame, self.metric,
                                   "sum2ed", "corre")
            metricMenu.pack(side=tk.TOP, pady=2)

            cb1 = tk.Button(self.anal_frame, text='Adjust Cluster',
                            command=self.handleClusterAnalysis)
            cb1.pack(side=tk.TOP, padx=2)


        button1 = tk.Button(self.anal_frame, text='Save Analysis',
                            command=self.handleSaveAnalysis)
        button1.pack(side=tk.TOP, padx=2)
        button2 = tk.Button(self.anal_frame, text='Close Analysis',
                            command=self.handleCloseAnalysis)
        button2.pack(side=tk.TOP, padx=2)

    # actually calculates the linear regression
    def calcLR(self, lr_headers):
        x = self.data.get_data([lr_headers[0]])
        y = self.data.get_data([lr_headers[1]])
        a,b,c,d,e = ss.linregress(x, y) #
        return [a,b,c,d,e]

    def setBindings(self):
        # bind mouse motions to the canvas
        self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
        self.canvas.bind( '<Button-2>', self.handleMouseButton2 )
        self.canvas.bind( '<Button-3>', self.handleMouseButton3 )
        self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
        self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
        self.canvas.bind( '<B2-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )
        self.canvas.bind( '<B3-Motion>', self.handleMouseButton3Motion)

        # bind command sequences to the root window
        self.root.bind( '<Command-q>', self.handleQuit )
        self.root.bind( '<Command-n>', self.handleClear_HK )
        self.root.bind( '<Command-o>', self.handleOpen )
        self.root.bind( '<Command-d>', self.handleDlog4 )
        self.root.bind( '<Command-v>', self.handleAnalysisViewer)

    # Handles file selection
    def handleOpen(self, event=None):
        if self.anal_frame is not None:
            self.handleCloseAnalysis()
        filename = tkFileDialog.askopenfilename(parent=self.root,
                                                title='Choose a Weapon',
                                                initialdir='.')
        filename = filename.split('/')[-1]
        self.OpenFile(filename)

    def OpenFile(self, filename):
        if filename != '':
            if len(self.saveQ) > 0:
                self.regenPrimaryJunk()
            self.setAxesMap(['','','','',''])
            self.handleClear(lr_clear=True)

            self.data = data.Data(filename)
            for i, header in enumerate(self.data.get_headers()):
                if i > 4:
                    break
                self.axes_map[i].set(header)

    # Handles some input in the orientation info bar
    def handleOrientInput(self, event=None):
        children = self.btmcntlframe.winfo_children()

        # Now, after the finagling above, go though the whole info bar and update
        for i in range(1, 6, 2):
            label = self.root.nametowidget(children[i-1]).cget('text')
            entry = self.root.nametowidget(children[i])

            entry_value = entry.get()
            self.orinfo[label].set(entry_value)

        self.updateOrinfo()
        self.root.focus_set()

    # Updates orientation according to orinfo
    def updateOrinfo(self):
        self.view.reset()

        # Corrects the Extent
        scale1 = float(self.orinfo['Scale:'].get())
        scale2 = 100/self.view.extent[0]
        self.view.extent = [(scale2/scale1)*self.view.extent[0],
                            (scale2/scale1)*self.view.extent[1],
                            (scale2/scale1)*self.view.extent[2]]


        # Corrects the Pitch/ Yaw
        newp = math.radians( float(self.orinfo['Pitch:'].get()) )
        newy = math.radians( float(self.orinfo['Yaw:'].get()) )

        self.view.rotateVRC(newy+math.pi, -newp)

        self.updateAxes()

    def handleQuit(self, event=None):
        self.root.destroy()

    def handleClear(self, event=None, lr_clear=False):
        if lr_clear:
            self.handleClear_LR()

        for point in self.objects:
            self.canvas.delete(point)
        self.objects = []
        self.obj2info = {}

    # Handles hotkey clearing
    def handleClear_HK(self, event=None):
        self.handleClear(lr_clear=True)

    # Handles save analysis button press
    def handleSaveAnalysis(self, event=None):
        #save pertinent info
        if len(self.saveQ2) > 0:
            type = self.saveQ2.pop(0)
        else:
            type = None
            self.root.bell()

        if type == 'LR':
            pertinent_package = [self.saveQ2.pop(0),
                                 self.saveQ2.pop(0),
                                 self.saveQ2.pop(0)]
            self.saved_anly['LR_'+str(self.anly_counter[0])] = pertinent_package
            self.anly_counter[0] += 1

        elif type == 'PCA':
            pertinent_package = [self.saveQ2.pop(0),
                                 self.saveQ2.pop(0)]
            self.saved_anly['PCA_'+str(self.anly_counter[1])] = pertinent_package
            self.anly_counter[1] += 1


    # Handles close analysis button press
    def handleCloseAnalysis(self, event=None):
        self.anal_frame.destroy()
        self.anal_frame=None
        self.regenPrimaryJunk()
        self.handlePlotData()

    def handleAnalysisViewer(self, event=None):
        d5 = Dialog5(parent=self.root, saved_anly=self.saved_anly,
                     counter=self.export_counter)
        self.saved_anly = d5.saved_anly
        self.export_counter = d5.counta

    # Opens user preferences dialog
    def handleDlog2(self):
        d2 = Dialog2(parent=self.root, user_c=self.user_c, title='User Preferences')
        self.user_c = d2.user_c[:]

    # Opens Axes Select Dialog
    def handleDlog3(self, event=None):
        d3 = Dialog3(parent=self.root, map=self.axes_map,
                     headers=self.data.get_headers(), title="Choose Axes")

        self.axes_map = d3.map

    # Opens Data Sheet dialog
    def handleDlog4(self, event=None):
        d4 = Dialog4(parent=self.root, print_string=self.data.print_string(),
                     title='Data Sheet')

    # Opens Linear Regression dialog
    def handleLinearRegression(self):
        headers = self.data.get_headers()
        d_SA = Dialog_SelectAxes(parent=self.root, headers=headers,
                                 title='Linearly Regress')

        if d_SA.selected_headers is not None and len(d_SA.selected_headers) > 1:
            self.handleClear()
            self.handleClear_LR()
            self.handleButton3() # reset and update

            self.buildLinearRegression(d_SA.selected_headers)
        elif not d_SA.cancelled:
            tkMessageBox.showerror(title='Input Error',
                                   message="Select Two or More Features")
            self.handleLinearRegression()

    # Opens PCA Analysis dialog
    def handlePCAAnalysis(self):
        headers = self.data.get_headers()
        d_SA = Dialog_SelectAxes(parent=self.root, headers=headers,
                                  title='Principal Components Analysis')

        selected_headers = d_SA.selected_headers
        if selected_headers is not None and len(selected_headers) > 1:
            self.last_pca = analysis.pca(self.data, selected_headers)
            self.buildPCAAnalysis()
            self.buildAnalysis_info('PCA')
            self.saveQ2 += ['PCA', self.last_pca, self.data]
        elif not d_SA.cancelled:
            tkMessageBox.showerror(title='Input Error',
                                   message="Select Two or More Features")
            self.handlePCAAnalysis()

    # Opens PCA Analysis Table
    def buildPCAAnalysis(self):
        d_PCA = Dialog_PCA(parent=self.root, pca=self.last_pca,
                           title='Principal Components Analysis')

    # Clears any LR objects from canvas (and info pane)
    def handleClear_LR(self, event=None):
        if self.anal_frame is not None:
            self.anal_frame.destroy()

        self.lr_ep = None
        for thing in self.lr_gfx:
            self.canvas.delete(thing)
        self.lr_gfx = []

    # handles cluster analysis
    def handleClusterAnalysis(self):
        dc = Dialog_Cluster(parent=self.root, K=self.K, krgbs=self.krgbs,
                            title='Cluster Analysis')

        if not dc.cancelled:
            if self.olddata is not None:
                old_headers = self.getAxesMap()

                self.data = self.olddata
                self.olddata = None
                self.OpenFile(self.data.filename)

                self.setAxesMap(old_headers)

            # update
            self.K = dc.K
            self.krgbs = dc.krgbs


            self.update4ClusterAnalysis()

            self.buildAnalysis_info('Cluster', self.K)

    # handles CA when clicked from menu rather than the analysis box
    def handleNewClusterAnalysis(self):
        self.regenPrimaryJunk()
        self.handleClusterAnalysis()

    # updates data for cluster analysis
    def update4ClusterAnalysis(self):
        headers = []
        for axis in self.getAxesMap():
            if axis != '':
                headers += [axis]


        metric = self.metric.get()
        if metric == 'sum2ed':
            codebook, codes, error = analysis.kmeans_numpy(self.data, headers,
                                                           self.K, False)
        elif metric == 'corre':
            codebook, codes, error = analysis.kmeans(self.data, headers, self.K, False,
                                       metric=self.metric.get())
            codebook = codebook.A
            codes = codes.T.tolist()[0]

        newdata = data.Data(filename=self.data.filename)
        newdata.addColumn('Clusters', 'numeric', codes)
        newdata.write('kced_'+str(newdata.filename), headers + ['Clusters'])


        # adds centroid points to file
        file = open('kced_'+str(newdata.filename), 'a')
        c = 0
        w_str = ''
        for point in codebook:
            for f in point:
                w_str += str(f) + ', '

            w_str += str(c) + '\n'
            c += 1
        w_str.rstrip('\n')
        file.write(w_str)
        file.close()


        self.olddata = self.data
        self.OpenFile('kced_'+str(newdata.filename))
        self.setAxesMap(headers)

        self.updatePoints4ClusterAnalysis()

        pass

    # recolors points's outlines according to clustering
    def updatePoints4ClusterAnalysis(self, updating=False):
        if not updating:
            self.handlePlotData()

        pt = 0
        for point in self.data.get_data().T:
            id = point[0,-1]
            color = '#{0}{1}{2}'.format(heximilate( self.krgbs[int(id)][0].get() ),
                                        heximilate( self.krgbs[int(id)][1].get() ),
                                        heximilate( self.krgbs[int(id)][2].get() ))
            self.canvas.itemconfigure(self.objects[pt], outline=color, width=2)
            pt+=1


        for i in range(-1, -self.K-1, -1):
            color = self.canvas.itemcget(self.objects[i], 'outline')
            self.canvas.itemconfigure(self.objects[i], fill='white', outline='black',
                                      dash=(2,3,2,3))


    # Update Color Button
    def handleButton1(self):
        for pt in self.objects:
            self.canvas.itemconfigure(pt, fill=self.colorOption.get())

    # builds the data.
    def handlePlotData(self, buttoned=True):
        if buttoned:
            if len(self.saveQ) > 0:
                self.regenPrimaryJunk()
            self.handleClear_LR()

        columns = self.chooseAxes()
        self.buildPoints(columns)
        self.updateLegend()

        # for K cluster
        if self.olddata is not None:
            self.updatePoints4ClusterAnalysis(True)

    # Feeds latest axis info into buildPoints
    def chooseAxes(self):
        headers = self.getAxesMap()
        return headers

    # Returns values of stringvars in axes map
    def getAxesMap(self):
        return [self.axes_map[0].get(), self.axes_map[1].get(), self.axes_map[2].get(),
                self.axes_map[3].get(), self.axes_map[4].get()]

    # Modifies all of axes map according to inputted list
    def setAxesMap(self, newheaders):
        for i,axis in enumerate(self.axes_map):
            if len(newheaders) > i:
                axis.set(newheaders[i])
            else:
                axis.set('')

    # Spinbutton
    def handleSpin1(self, event=None):
        prexist = None # keeps analysis info pane from not being rebuilt
        if self.anal_frame is not None:
            prexist = self.anal_frame['text']

        self.dx = float(self.spin1.get())
        if len(self.objects) > 0:
            self.handlePlotData(buttoned=False)
            self.updateLegend()

        if prexist is not None:
            self.buildAnalysis_info(prexist)

    # Reset View
    def handleButton3(self):
        self.view.reset(screen=self.screen)
        self.orinfo['Pitch:'].set('0.0')
        self.orinfo['Yaw:'].set('180.0')
        self.orinfo['Scale:'].set('100.0')
        self.updateAxes()

    # Handles Mouse over of point on canvas
    def handleMouseOver(self, event):
        point = self.canvas.find_closest(event.x, event.y)[0]
        format_list = ['X-Axis', 'Y-Axis', 'Z-Axis', 'Size', 'Color']
        for i in range( len(format_list) ):
            if self.axes_map[i].get() != '':
                format_list[i] = self.axes_map[i].get()

        for i in range(5):
            format_list += [ self.obj2info[point][i] ]

        string = '{0}: {5}, {1}: {6}, {2}: {7}, {3}: {8}, {4}: {9}'.format(
                format_list[0], format_list[1], format_list[2], format_list[3],
                format_list[4], format_list[5], format_list[6], format_list[7],
                format_list[8], format_list[9]) # bleh

        self.mouseoverinfo.set(string)

    def handleMouseUnder(self, event):
        self.mouseoverinfo.set('...')

    # Panning
    def handleMouseButton1(self, event):
        self.baseClick = (event.x, event.y)
        self.og_view = self.view.clone()

    # Rotation
    def handleMouseButton2(self, event):
        self.baseClick2 = (event.x, event.y)
        self.og_view = self.view.clone()

    # Scaling
    def handleMouseButton3(self, event):
        self.baseClick = (event.x, event.y)
        self.og_view = self.view.clone()
        self.og_extent = self.og_view.extent

    # Translation Motion
    def handleMouseButton1Motion(self, event):
        delta0, delta1 = (float(event.x)-self.baseClick[0],
                          float(event.y)-self.baseClick[1] )
        self.baseClick = (event.x, event.y)
        delta0, delta1 = ( delta0 * self.view.extent[0], delta1 * self.view.extent[1] )
        delta0, delta1 = ( delta0 / self.view.screen[0], delta1 / self.view.screen[1] )

        delta0*=self.user_c[0]
        delta1*=self.user_c[0]

        self.view.vrp += delta0 * self.view.u + delta1 * self.view.vup

        self.updateAxes()

    # Rotation Motion
    def handleMouseButton2Motion(self, event):
        c = math.pi/200
        delta0 = c*(float(event.x)-self.baseClick2[0])
        delta1 = c*(float(event.y)-self.baseClick2[1])

        delta0*=self.user_c[2]
        delta1*=self.user_c[2]

        self.view = self.og_view.clone()
        self.view.rotateVRC(-delta0, delta1)
        self.updateAxes()

        pitch = math.asin(self.view.vpn[0,1])
        yaw = math.atan2(self.view.vpn[0,0], self.view.vpn[0,2])
        self.orinfo['Pitch:'].set( str(math.degrees(pitch))[:6] )
        self.orinfo['Yaw:'].set( str(math.degrees(yaw))[:6] )

    # Scaling Motion
    def handleMouseButton3Motion(self, event):
        c = 0.03
        c *= self.user_c[1]
        scale = c*(float(event.y)-self.baseClick[1])

        total = scale+self.og_extent[0]

        if self.og_extent[0] >= 2.979 and total > 3.0:
            scale = 3.0-self.og_extent[0]
            total = scale+self.og_extent[0]

        if 0.1 <= total <= 3.0:

            self.view.extent = [self.og_extent[0]+scale,
                                self.og_extent[1]+scale,
                                self.og_extent[2]+scale]

        self.orinfo['Scale:'].set( 100/self.view.extent[0] )
        self.updateAxes()

    # Creates a point and returns its tag
    def createPoint(self, x, y, size=None, fillcolor="default", rawinfo=None,legend=None):
        # Color
        if fillcolor == "default":
            fillcolor = self.colorOption.get()
        else:
            if 0 > fillcolor or 1 < fillcolor:
                fillcolor = int(fillcolor)
                print 'erred?'
            if 0 <= fillcolor < 0.5:
                c = fillcolor*2
                rgb = (200,0,90)
                fillcolor = [rgb[0]-rgb[0]*c,
                             rgb[1],
                             rgb[2]-rgb[2]*c]
            elif 0.5 <= fillcolor <= 1.0:
                c = (fillcolor-0.5)*2
                rgb = (0,204,204)
                fillcolor = [rgb[0]*c,
                             rgb[1]*c,
                             rgb[2]*c]

            fillcolor = '#{0}{1}{2}'.format(heximilate( int(fillcolor[0]) ),
                                            heximilate( int(fillcolor[1]) ),
                                            heximilate( int(fillcolor[2]) ))

        # (200,0,90) - purp
        # (0,0,0) - bluck
        # (50,255,100) - tealy

        # Size
        if size is None:
            size = 1

        dx = (self.dx * size) + 1

        if legend is None:
            if self.shapebox.get(tk.ACTIVE) == "Triangle":
                point = self.canvas.create_polygon(x, y-dx,
                                                   x+dx, y+dx,
                                                   x-dx, y+dx,
                                                   fill=fillcolor, outline=fillcolor)

            elif self.shapebox.get(tk.ACTIVE) == "Square":
                point = self.canvas.create_rectangle(x-dx, y-dx,
                                                     x+dx, y+dx,
                                                     fill=fillcolor, outline=fillcolor)
            else:
                point = self.canvas.create_oval(x-dx, y-dx, x+dx, y+dx,
                                                fill=fillcolor, outline=fillcolor)

            self.canvas.tag_bind(point, '<Enter>', self.handleMouseOver)
            self.canvas.tag_bind(point, '<Leave>', self.handleMouseUnder)

            self.objects.append(point)
            self.obj2info[point] = rawinfo

        else:
            if legend == 3:
                point = [self.legendcanvi[3].create_oval(
                        x-dx, y-dx, x+dx, y+dx, fill=fillcolor, outline=fillcolor)]
            elif legend == 4:
                point = [self.legendcanvi[4].create_oval(
                        x-dx, y-dx, x+dx, y+dx, fill=fillcolor, outline=fillcolor)]

        return point

    # Moves point
    def movePoint(self, point, newx, newy):
        if self.shapebox.get(tk.ACTIVE) == "Triangle":
            dx = self.canvas.coords(point)[2] - self.canvas.coords(point)[0]

            self.canvas.coords(point,
                               newx, newy-dx,
                               newx+dx, newy+dx,
                               newx-dx, newy+dx)
        else:
            dx = (self.canvas.coords(point)[2] - self.canvas.coords(point)[0]) * 0.5

            self.canvas.coords(point,
                               newx-dx, newy-dx,
                               newx+dx, newy+dx)

    def main(self):
        print 'Entering main loop'
        self.root.mainloop()

# Some functions which make life easier

# Converts to an appropriate hexadecimal string to specify color
def heximilate(number):
    string = hex(number)[2:]
    if len(string) < 2:
        string = '0' + string
    return string

# converts a list to a string; spacing between list item and list item length parameters
def l2s(list, length=7):
    string = ''
    for item in list:
        string += str(item)[0:length] + ('{0}')
    return string.rstrip('{0}')


# Support class for creating dialogue windows
class Dialog(tk.Toplevel):

    def __init__(self, parent, title = None):

        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent

        self.result = None

        body = tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.initial_focus.focus_set()

        self.cancelled = False

        self.wait_window(self)

    #
    # construction hooks

    def body(self, master):
        # create dialog body.  return widget that should have
        # initial focus.  this method should be overridden

        pass

    def buttonbox(self):
        # add standard button box. override if you don't want the
        # standard buttons

        box = tk.Frame(self)

        w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)
        w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    #
    # standard button semantics

    def ok(self, event=None):

        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        self.apply()

        self.cancel()
        self.cancelled = False

    def cancel(self, event=None):

        # put focus back to the parent window
        self.parent.focus_set()
        self.destroy()
        self.cancelled = True

    #
    # command hooks

    def validate(self):

        return 1 # override

    def apply(self):

        pass # override


# Dialogue for Axes Selection for Analysis
class Dialog_SelectAxes(Dialog):
    def __init__(self, parent, headers, title=None):

        self.type = title
        self.headers = headers
        self.selected_headers = None

        Dialog.__init__(self, parent, title)



    def body(self, master):
        iterations = 0
        if self.type == 'Linearly Regress':
            tk.Label(master, text="X Dimension:").grid(row=0, column=0)
            tk.Label(master, text="Y Dimension:").grid(row=0, column=1)
            iterations = 2
        elif self.type == 'Principal Components Analysis':
            tk.Label(master, text="Pick Thy Columns:").grid(row=0, column=0)
            iterations = 1

        self.lbs = []
        for i in range(iterations):
            if self.type == 'Linearly Regress':
                self.lbs += [tk.Listbox(master, selectmode=tk.SINGLE,
                                        exportselection=0)]
                self.lbs[i].grid(row=1, column=i)

            elif self.type == 'Principal Components Analysis':
                self.lbs += [tk.Listbox(master, selectmode=tk.MULTIPLE,
                                        exportselection=0)]
                self.lbs[i].grid(row=1, column=i, sticky=tk.N+tk.S+tk.E+tk.W)

                scroll = tk.Scrollbar(master, orient=tk.VERTICAL,
                                      command=self.lbs[i].yview)
                scroll.grid(row=1, column=i+1, sticky=tk.N + tk.S)
                self.lbs[i]['yscrollcommand'] = scroll.set

        for item in self.headers:
            if item is not None:
                for lb in self.lbs:
                    lb.insert(tk.END, item)

        return

    def apply(self):
        self.selected_headers = []
        for selected in self.lbs:
            select_few = selected.curselection()
            for chosen in select_few:
                self.selected_headers += [selected.get(chosen)]

# Dialog window for PCA Analysis
class Dialog_PCA(Dialog):
    def __init__(self, parent, pca, title = None):
        self.pca = pca

        Dialog.__init__(self, parent, title)

    def body(self, master):

        labels = ['Eigenvalue:', 'Eigenvector'] + self.pca.get_data_headers()

        for i, labeling in enumerate(labels):
            label = tk.Label(master, text=labeling, width=11)
            label.grid(row=i, column=0)
            if i % 2 == 0:
                label['bg'] = '#D9BCE3'


        text0 = l2s(self.pca.get_eigenvalues().tolist()).format(' ' * 3)
        row0 = tk.Label(master, text=text0, bg='#D9BCE3')
        row0.grid(row=0, column=1, padx=3)

        sep = tk.Frame(master, height=2, bd=1, relief=tk.SUNKEN)
        sep.grid(row=1, column=1, sticky=tk.W+tk.E, padx=3)

        for i,row in enumerate(self.pca.get_eigenvectors()):
            text = l2s(row.tolist()).format(' ' * 3)
            row_label = tk.Label(master, text=text)
            row_label.grid(row=2+i, column=1, padx=3)
            if i % 2 == 0:
                row_label['bg'] = '#D9BCE3'

        return


    def apply(self):
        pass

class Dialog_Cluster(Dialog):
    def __init__(self, parent, K, krgbs, title = None):
        self.parent = parent

        self.K = K
        self.krgbs = krgbs


        self.labels = []
        self.colorboxes = [] # box of color
        self.sliders = [] # 3 sliders per

        Dialog.__init__(self, parent, title)

    def body(self, master):
        # K specification part
        label1 = tk.Label(master, text='K number of Clusters:')
        label1.grid(row=0, column=38)
        self.spin1 = tk.Spinbox(master, from_=self.K, to=15, width=3,justify=tk.CENTER,
                                increment=1, state='readonly', command=self.updateKboxes)
        self.spin1['from_'] = 1
        self.spin1.grid(row=0, column=39, pady=5)

        label2 = tk.Label(master, text='Distance Metric:')
        label2.grid(row=1, column=38)

        # Color Selection part
        self.Kboxes = tk.LabelFrame(master, text='Select Cluster Colors', labelanchor=tk.N)
        self.Kboxes.grid(row=2, column=0, columnspan=40, sticky=tk.E+tk.W, pady=5)

        for i in range(self.K):
            self.makeKbox(self.Kboxes, i)

        return

    def apply(self):
        pass

    def makeKbox(self, parent, row):
        self.labels += [tk.Label(parent, text='Cluster '+str(row+1)+':', bg='#E4E4E4')]
        self.labels[-1].grid(row=row, column=0, padx=2)
        self.colorboxes += [tk.Frame(parent, width=20, height=20, relief=tk.RAISED, bd=5)]
        self.colorboxes[-1].grid(row=row, column=1, padx=2)
        if row >= len(self.krgbs):
            self.krgbs += [[tk.IntVar(self),
                            tk.IntVar(self),
                            tk.IntVar(self)]]

        temp = []
        for i,bar in enumerate(['R', 'G', 'B']):
            temp += [tk.Scale(parent, orient=tk.HORIZONTAL, from_=0, to=255,
                              label=bar, resolution=1, variable=self.krgbs[row][i],
                              command=self.updateColor,
                              font=tkf.Font(family='Helvetica', weight='bold'))]
            temp[-1].grid(row=row, column=i+2, pady=3, padx=2)

        self.sliders += [temp]


    # handles the number of color selector boxes to display
    def updateKboxes(self, event=None):
        newK = int(self.spin1.get())
        if newK > self.K:
            self.makeKbox(self.Kboxes, self.K)
        else:
            self.krgbs.pop(-1)
            naughtylist = [self.labels.pop(-1), self.colorboxes.pop(-1), self.sliders.pop(-1)]
            i = 0
            for child in naughtylist:
                if i == 2:
                    for bar in child:
                        bar.destroy()
                else:
                    child.destroy()

                i+=1

        self.K = newK

    def updateColor(self, event=None):
        for i in range(len(self.colorboxes)):
            self.colorboxes[i]['bg'] = '#{0}{1}{2}'.format(heximilate(self.krgbs[i][0].get()),
                                                           heximilate(self.krgbs[i][1].get()),
                                                           heximilate(self.krgbs[i][2].get()))


# Dialog window for interaction constraints
class Dialog2(Dialog):
    def __init__(self, parent, user_c, title = None):

        self.user_c = user_c

        Dialog.__init__(self, parent, title)

    def body(self, master):

        tk.Label(master, text="Panning Speed:").grid(row=0)
        tk.Label(master, text="Scaling Speed:").grid(row=1)
        tk.Label(master, text="Rotation Speed:").grid(row=2)

        self.s1 = tk.Scale( master, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                            resolution=0.1)
        self.s1.pack(side=tk.TOP)
        self.s1.set(self.user_c[0])

        self.s2 = tk.Scale( master, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                            resolution=0.1)
        self.s2.pack(side=tk.TOP)
        self.s2.set(self.user_c[1])

        self.s3 = tk.Scale( master, from_=0.1, to=2.0, orient=tk.HORIZONTAL,
                            resolution=0.1)
        self.s3.pack(side=tk.TOP)
        self.s3.set(self.user_c[2])


        self.s1.grid(row=0, column=1)
        self.s2.grid(row=1, column=1)
        self.s3.grid(row=2, column=1)

        return self.s1 # initial focus

    def apply(self):
        self.user_c[0] = self.s1.get()
        self.user_c[1] = self.s2.get()
        self.user_c[2] = self.s3.get()


# Plotting Dialogue
class Dialog3(Dialog):
    def __init__(self, parent, map, headers, title = None):
        self.map = map
        self.headers = headers
        self.buttons = []

        Dialog.__init__(self, parent, title)

    def body(self, master):
        canvas = tk.Canvas(master, width = 230, height = 120)
        scrollbar = tk.Scrollbar(master, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.grid(row=0, column=0, sticky=tk.E+tk.W)
        scrollbar.grid(row=1, column=0, sticky=tk.E+tk.W)
        canvas['xscrollcommand'] = scrollbar.set

        frame = tk.Frame(canvas)
        frame_id = canvas.create_window(0, 0, window=frame, anchor=tk.NW)


        # Builds labels
        dimensions = ['X-Axis:', 'Y-Axis:', 'Z-Axis:', 'Size:', 'Color:']
        for i,label in enumerate(dimensions):
            tk.Label(frame, text=label).grid(row=i, column=0)

        # Builds rows of Buttons
        for i in range(5):
            row = []
            for header in self.headers:
                row += [tk.Radiobutton(frame, value=header , variable=self.map[i],
                                       text=header, padx=3)]
            row += [tk.Radiobutton(frame, value='', variable=self.map[i],
                                   text='None', padx=3)]
            self.buttons += [row]

        for i, row in enumerate(self.buttons):
            for j, button in enumerate(row):
                button.grid(row=i, column=j+1)

        # Restores Selection
        for i,row in enumerate(self.buttons):
            selected = self.map[i].get()
            for button in row:
                if button['value'] == selected:
                    button.select()

        canvas['scrollregion']=(0,0,2050,0)

        return

    def apply(self):
        # Prioritizes x>y>z>color>size>
        already_selected= []
        for selected in self.map:
            if selected.get() in already_selected:
                selected.set('')
            else:
                already_selected += [selected.get()]

# Data Sheet Dialog
class Dialog4(Dialog):
    def __init__(self, parent, print_string, title = None):
        self.print_string = print_string

        Dialog.__init__(self, parent, title)

    def body(self, master):
        junk = []
        junk += [tk.Label(master, text='Data Sheet')]
        junk += [tk.Text(master, wrap=tk.NONE)]
        junk += [tk.Scrollbar(master, command=junk[1].yview)]
        junk += [tk.Scrollbar(master, command=junk[1].xview, orient=tk.HORIZONTAL)]

        junk[0].grid(row=0,column=0)
        junk[1].grid(row=2,column=0)
        junk[2].grid(row=2,column=1, sticky=tk.N+tk.S)
        junk[3].grid(row=1,column=0, sticky=tk.E+tk.W)

        junk[1]['yscrollcommand'] = junk[2].set
        junk[1]['xscrollcommand'] = junk[3].set

        junk[1].insert('1.0', self.print_string)
        #junk[1].tag_configure('center', justify='center')
        #junk[1].tag_add('center', 1.0, 'end')

        junk[1]['state'] = tk.DISABLED

        return

    def apply(self):
        return

# Analysis Viewer
class Dialog5(Dialog):
    def __init__(self, parent, saved_anly, counter, title = None):
        self.saved_anly = saved_anly
        self.counta = counter
        self.trunk = []
        self.current = None

        title = 'Analysis Viewer'

        Dialog.__init__(self, parent, title)

    def body(self, master):
        self.trunk += [tk.LabelFrame(master)]

        # Strips a display App to its canvas
        self.display = DisplayApp(400, 400, self.trunk[0], screen=[325,325])
        self.display.root.geometry('1x1') # sweep it under the rug
        self.display.root.withdraw()

        self.trunk += [tk.Frame(master, relief=tk.SUNKEN)]
        self.trunk += [tk.Frame(master)]
        self.trunk += [tk.Listbox(self.trunk[2], selectmode=tk.SINGLE)]
        self.trunk += [tk.Button(self.trunk[2], text='Plot', command=self.handlePlot)]
        self.trunk += [tk.Button(self.trunk[2], text='Delete', command=self.handleDelete)]
        self.trunk += [tk.Button(self.trunk[2], text='Export', command=self.handleExport)]
        self.trunk += [tk.Button(self.trunk[2], text='Reset View',
                                 command=self.display.handleButton3)]
        self.trunk += [tk.Button(self.trunk[2], text='Plot Info',
                                 command=self.handleInfo)]

        for i in range(len(self.trunk)):
            if i < 3:
                if i == 1:
                    self.trunk[i].pack(side=tk.LEFT, fill=tk.Y, padx=3)
                else:
                    self.trunk[i].pack(side=tk.LEFT)
            else:
                self.trunk[i].pack(side=tk.TOP)

        # Populates the lb_labels with Saved analyses
        for saved in self.saved_anly.keys():
            self.trunk[3].insert(tk.END, saved)

        self.select()

        return

    # Auto-selects the first item in LB
    def select(self, auto=None):
        if auto is None:
            i = 0
        else:
            i = auto

        self.trunk[3].select_set(i)
        self.trunk[3].activate(i)
        self.trunk[3].event_generate("<<ListboxSelect>>")


    def handlePlot(self, event=None):
        if len(self.trunk[3].get(0,tk.END)) > 0:
            self.current = self.saved_anly[self.trunk[3].get(tk.ACTIVE)]
            self.plot()

    def handleInfo(self, event=None):
        if self.trunk[3].get(tk.ACTIVE)[0:2] == 'LR':
            self.bell()

        elif self.trunk[3].get(tk.ACTIVE)[0:3] == 'PCA':
            Dialog_PCA(parent=self, pca=self.display.data,
                       title='Principal Components Analysis')

        else:
            self.bell()

    def handleExport(self, event=None):
        name = 'SavedAnalysis'+str(self.counta)
        # Saves Image
        self.display.canvas.postscript(colormode='color',
                                       file=name)
        self.counta+=1

        # Writes data to file
        name += ': '
        file = open('SavedAnalyses.txt', 'a')
        write_string = ''

        if self.trunk[3].get(tk.ACTIVE)[0:2] == 'LR':
            write_string = name + 'Slope: {0} Intercept: {1} R^2_Value: {2}' \
                                  ' P_Value: {3} Std_Err: {4}'.format(self.current[2][0],
                                                                      self.current[2][1],
                                                                      self.current[2][2],
                                                                      self.current[2][3],
                                                                      self.current[2][4])
            write_string += '\nDatafile: {0} X: {1} Y: {2}'.format(
                self.current[1].filename, self.current[0][1], self.current[0][0])

        elif self.trunk[3].get(tk.ACTIVE)[0:3] == 'PCA':
            h = self.current.get_headers()
            write_string = name + '\nDatafile: {0} X: {1} Y:{2} Z:{3} ' \
                                  'Size:{4} Color:{5}'.format(
                self.current[1].filename, h[0], h[1], h[2], h[3], h[4])


        write_string += '\n\n'
        file.write(write_string)
        file.close()

    def handleDelete(self, event=None):
        self.saved_anly.pop( self.trunk[3].get(tk.ACTIVE) )
        i = len( self.trunk[3].get(0, tk.ACTIVE) )-1
        self.trunk[3].delete(tk.ACTIVE)
        if i < 1:
            i+=1
        self.select(i-1)

    def plot(self):
        if self.trunk[3].get(tk.ACTIVE)[0:2] == 'LR':
            self.display.data = self.current[1]
            self.display.handleClear_LR()
            self.display.buildLinearRegression(self.current[0])

            for i,label in enumerate(self.display.axes_labels):
                if i == 0:
                    self.display.canvas.itemconfigure(label, text=str(self.current[0][0]))
                if i == 1:
                    self.display.canvas.itemconfigure(label, text=str(self.current[0][1]))
                if i == 2:
                    self.display.canvas.itemconfigure(label, text=str(self.current[0][2]))

        elif self.trunk[3].get(tk.ACTIVE)[0:3] == 'PCA':
            self.display.data = self.current[0]
            self.display.setAxesMap(self.display.data.get_headers())
            self.grab_release()
            self.display.handleDlog3()
            self.grab_set()
            self.display.handlePlotData()

    def apply(self):
        self.display.root.destroy()


if __name__ == "__main__":
    dapp = DisplayApp(1200, 675)
    dapp.main()


