import cv2
import numpy as np
from numpy import linalg as LA
import math
import wx
import os
import vtk
import vtk.util.numpy_support
import sys


class MainFrame(wx.Frame):
    def __init__(self, data):
        wx.Frame.__init__(self, None, wx.NewId(), 'U-TRACK')
        data_pass = data
        panel = MainPanel(self)
        self.Fit()
        self.Centre()
        self.Show(True)
        self.SetPosition((250, 30))
        self.SetMaxSize((200, 525))
        self.SetSize((200, 525))
        self.SetBackgroundColour((50, 50, 50))
        self.SetForegroundColour((255, 255, 255))


class MainPanel(wx.Panel, MainFrame):
    def __init__(self, frame, *args, **kwargs):
        wx.Panel.__init__(self, frame, *args, **kwargs)

        # Button
        cmd_open = wx.Button(self, wx.ID_ANY, 'Open...')
        select_pb_points = wx.Button(self, wx.ID_ANY, 'Place PB Pts')
        load_prev_seg = wx.Button(self, wx.ID_ANY, 'Load Previous Segmentations')
        select_ant_points = wx.Button(self, wx.ID_ANY, 'Outline Ant Urethra')
        select_post_points = wx.Button(self, wx.ID_ANY, 'Outline Post Urethra')
        select_sling_points = wx.Button(self, wx.ID_ANY, 'Outline Sling')
        self.ruler = wx.Button(self, wx.ID_ANY, 'Ruler')
        self.protractor = wx.Button(self, wx.ID_ANY, 'Protractor')
        self.exporter = wx.Button(self, wx.ID_ANY, 'Export Current Frame')
        self.clear = wx.Button(self, wx.ID_ANY, 'Clear')

        # button sizer
        button_sizer = wx.BoxSizer(wx.VERTICAL)
        button_sizer.Add((200, 20))
        button_sizer.Add(cmd_open, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(load_prev_seg, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(select_pb_points, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(select_ant_points, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(select_post_points, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(select_sling_points, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(self.clear, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add((200, 20))
        button_sizer.Add((200, 20))
        button_sizer.Add((200, 20))
        button_sizer.Add(self.ruler, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(self.protractor, 1, wx.EXPAND)
        button_sizer.Add((200, 20))
        button_sizer.Add(self.exporter, 1, wx.EXPAND)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(button_sizer)
        main_sizer.Add((200, 50))
        self.SetSizer(main_sizer)
        self.Fit()

        self.dirname = ''
        self.Bind(wx.EVT_BUTTON, self.OnOpen, cmd_open)
        self.Bind(wx.EVT_BUTTON, self.load_method, load_prev_seg)
        self.Bind(wx.EVT_BUTTON, self.select_PB_points, select_pb_points)
        self.Bind(wx.EVT_BUTTON, self.select_ANT_points, select_ant_points)
        self.Bind(wx.EVT_BUTTON, self.select_POST_points, select_post_points)
        self.Bind(wx.EVT_BUTTON, self.select_SLING_points, select_sling_points)
        self.Bind(wx.EVT_BUTTON, self.ruler_method, self.ruler)
        self.Bind(wx.EVT_BUTTON, self.protractor_method, self.protractor)
        self.Bind(wx.EVT_BUTTON, self.export_method, self.exporter)
        self.Bind(wx.EVT_BUTTON, self.clear_method, self.clear)

        self.ant_pts = []
        self.sling_pts = []
        self.pb_pts = []
        self.post_pts = []
        self.first_window_up = True
        self.ruler_pts = []
        self.ruler_tool = False
        self.ruler_print = False
        self.num_ruler_pts = 0
        self.angle_pts = []
        self.angle_tool = False
        self.angle_print = False
        self.num_angle_pts = 0
        self.point_selected = False
        self.POST_check = False
        self.ANT_check = False
        self.SLING_check = False
        self.pb_check = False
        self.point_type = ""
        self.track_pos = 0
        self.current_img = -1000
        self.clear = False
        self.loadedData = False

    # font for text in windows
    font = cv2.FONT_HERSHEY_SIMPLEX

    def load_method(self, e):
        prev = []
        self.pb_pts = []
        self.ant_pts = []
        self.post_pts = []
        self.sling_pts = []
        self.shut_down_windows()
        self.first_window_up = False
        self.loadedData = True
        cv2.destroyAllWindows()
        dlg = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.txt", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            self.filename_total = os.path.join(self.dirname, self.filename)
        dlg.Destroy()

        oldSeg = open(self.filename_total, 'r')
        oldSegData = oldSeg.readlines()
        self.filename_total = oldSegData[0][:-1]
        self.trackpos = int(oldSegData[1][:-1])

        self.pb_pts = eval(oldSegData[2][:-1])
        self.ant_pts = eval(oldSegData[3][:-1])
        self.post_pts = eval(oldSegData[4][:-1])
        if len(oldSegData) == 6:
            self.sling_pts = eval(oldSegData[5][:-1])

        def homebrewRiffle(data):
            formattedData = []
            for item in range(len(data)):
                if item % 2 == 0:
                    temp = [data[item], data[item + 1]]
                    formattedData.append(tuple(temp))
            return tuple(formattedData)

        self.pb_pts = homebrewRiffle(self.pb_pts)
        self.ant_pts = homebrewRiffle(self.ant_pts)
        self.post_pts = homebrewRiffle(self.post_pts)
        if self.sling_pts != []:
            self.sling_pts = homebrewRiffle(self.sling_pts)

        self.cap = cv2.VideoCapture(self.filename_total)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.track_pos)
        _, prev = self.cap.read()
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.w_main = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.w_main != 800:
            prev = prev[:, :]
            # prev = prev[43:680, 64:821]
        elif self.w_main == 800 and sys.platform == "win32":
            prev = cv2.resize(prev, (0, 0), fx=0.75, fy=0.75)
        if self.pb_pts != []:
            for pnt in self.pb_pts:
                cv2.circle(prev, pnt, 5, (0, 0, 255), 2)
        if self.ant_pts != []:
            for pnt in self.ant_pts:
                cv2.polylines(prev, [np.array(self.ant_pts)], True, (0, 255, 255), 1)
        if self.post_pts != []:
            for pnt in self.post_pts:
                cv2.polylines(prev, [np.array(self.post_pts)], True, (0, 100, 255), 1)
        if self.sling_pts != []:
            for pnt in self.sling_pts:
                cv2.polylines(prev, [np.array(self.sling_pts)], True, (255, 0, 0), 1)

        cv2.imshow('Loaded Segmentations', prev)
        cv2.waitKey(1)

    def export_method(self, e):
        if self.POST_check and self.ANT_check and self.pb_check:
            # Pack data into VTK format
            outpoints = vtk.vtkPoints()
            outpoints.SetDataTypeToFloat()
            if self.sling_pts != []:
                total_data = self.pb_pts + self.ant_pts + self.post_pts + self.sling_pts
            else:
                total_data = self.pb_pts + self.ant_pts + self.post_pts
            data = np.zeros([len(total_data), 3], dtype=np.float32)
            data[:, 0:2] = np.array(total_data, dtype=np.float32) / self.scale
            outpoints.SetData(vtk.util.numpy_support.numpy_to_vtk(data, deep=True, array_type=vtk.VTK_FLOAT))
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(len(self.pb_pts))

            for i in range(len(self.pb_pts)):
                lines.InsertCellPoint(i)
            lines.InsertNextCell(len(self.ant_pts) + 1)

            for i in range(len(self.ant_pts)):
                lines.InsertCellPoint(i + len(self.pb_pts))
            lines.InsertCellPoint(0 + len(self.pb_pts))
            lines.InsertNextCell(len(self.post_pts) + 1)

            for i in range(len(self.post_pts)):
                lines.InsertCellPoint(i + len(self.pb_pts) + len(self.ant_pts))
            lines.InsertCellPoint(0 + len(self.pb_pts) + len(self.ant_pts))
            lines.InsertNextCell(len(self.sling_pts) + 1)

            if self.sling_pts != []:
                for i in range(len(self.sling_pts)):
                    lines.InsertCellPoint(i + len(self.pb_pts) + len(self.ant_pts) + len(self.post_pts))
                lines.InsertCellPoint(0 + len(self.pb_pts) + len(self.ant_pts) + len(self.post_pts))

            # write transformed data to files
            if vtk.VTK_MAJOR_VERSION <= 5:
                outpoints.Update()

            writer = vtk.vtkPolyDataWriter()
            writer.SetFileVersion(42)
            polygon = vtk.vtkPolyData()
            polygon.SetPoints(outpoints)
            polygon.SetLines(lines)

            """ Save a file"""
            with wx.FileDialog(self, "Save PB Points file", wildcard="VTK files (*.vtk)|*.vtk",
                               style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:

                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return  # the user changed their mind

                # save the current contents in the file
                pathname = fileDialog.GetPath()
                writer.SetFileName(pathname)
                # writer.SetDataModeToAscii()
                if vtk.VTK_MAJOR_VERSION <= 5:
                    writer.SetInput(polygon)
                else:
                    writer.SetInputData(polygon)

                try:
                    writer.Write()
                except IOError:
                    wx.LogError("Cannot save current data in file '%s'." % pathname)

            def convertTuple(tup):
                converted = str(tup[0][0]) + ', ' + str(tup[0][1])
                for i in tup[1:]:
                    converted = converted + ', ' + str(i[0]) + ', ' + str(i[1])
                converted = converted + '\n'
                return converted

            textdata = open(pathname[:-4] + '.txt', 'w')
            framenumber = cv2.getTrackbarPos("scroll", 'Selection & Meshing')
            textdata.writelines(str(self.dirname) + "/" + str(self.filename) + '\n')
            textdata.writelines(str(framenumber) + '\n')
            textdata.writelines(convertTuple(self.pb_pts))
            textdata.writelines(convertTuple(self.ant_pts))
            textdata.writelines(convertTuple(self.post_pts))
            if self.sling_pts != []:
                textdata.writelines(convertTuple(self.sling_pts))

            textdata.close()

    def clear_method(self, e):
        self.ant_pts = []
        self.pb_pts = []
        self.post_pts = []
        self.sling_pts = []
        self.old_points = []
        self.cleared = True
        self.POST_check = False
        self.SLING_check = False
        self.ANT_check = False
        self.pb_check = False

    def protractor_method(self, e):
        if self.dirname != '':
            self.first_window_up = False
            self.angle_pts = []
            self.angle_tool = True
            self.ruler_tool = False
            self.angle_selected = False
            self.angle_window()

    def ruler_method(self, e):
        if self.dirname != '':
            self.first_window_up = False
            self.ruler_pts = []
            self.ruler_tool = True
            self.angle_tool = False
            self.ruler_selected = False
            self.ruler_window()

    def LOAD_PREV_SEG(self, e):
        self.first_window_up = False

    def select_PB_points(self, e):
        self.first_window_up = False
        self.meshed = False
        self.pb_check = False
        self.ruler_tool = False
        self.point_selected = False
        self.angle_tool = False
        self.pb_pts = []
        self.old_points = []
        self.point_type = "PB"
        self.kill_loop = True
        self.selection_window()

    def select_ANT_points(self, e):
        self.first_window_up = False
        self.meshed = False
        self.ANT_check = False
        self.ruler_tool = False
        self.point_selected = False
        self.angle_tool = False
        self.mesh_check = True
        self.ant_pts = []
        self.old_points = []
        self.point_type = "ANT"
        self.kill_loop = True
        self.selection_window()

    def select_POST_points(self, e):
        self.first_window_up = False
        self.meshed = False
        self.ruler_tool = False
        self.POST_check = False
        self.angle_tool = False
        self.point_selected = False
        self.mesh_check = True
        self.post_pts = []
        self.old_points = []
        self.point_type = "POST"
        self.kill_loop = True
        self.selection_window()

    def select_SLING_points(self, e):
        self.first_window_up = False
        self.meshed = False
        self.ruler_tool = False
        self.SLING_check = False
        self.angle_tool = False
        self.point_selected = False
        self.mesh_check = True
        self.sling_pts = []
        self.old_points = []
        self.point_type = "SLING"
        self.kill_loop = True
        self.selection_window()

    def OnOpen(self, e):
        self.first_window_up = True
        self.ant_pts = []
        self.pb_pts = []
        self.post_pts = []
        self.sling_pts = []
        self.track_pos = 0
        self.current_img = -1000
        """ Open a file"""
        dlg = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            self.filename_total = os.path.join(self.dirname, self.filename)
            self.first_window()
        dlg.Destroy()
        return

    # create the window and window parameters
    def establish_frame(self, frame_name, file, trackbar_y_n, mouse_select_y_n):
        frame_position = self.track_pos
        self.cap = cv2.VideoCapture(file)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.w_main = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.w_main != 800:
            if int(self.w_main) == 992:
                self.w = 992
                self.h = 802
            if int(self.w_main) == 1016:
                self.w = 1016
                self.h = 936
            else:
                self.w = 733
                self.h = 617
        if self.w_main == 800:
            self.scale = 122  # 122 pixels = 1cm
        else:
            self.scale = 115  # 115 pixels = 1cm
        if self.ruler_tool == True:
            select_point_method = self.select_ruler
        elif self.angle_tool == True:
            select_point_method = self.select_angle
        else:
            select_point_method = self.select_point
        cv2.namedWindow(frame_name)
        if mouse_select_y_n == 1:
            cv2.setMouseCallback(frame_name, select_point_method)
        if trackbar_y_n == 1:
            cv2.createTrackbar("scroll", frame_name, 1, self.n_frames - 1, self.nothing)
            cv2.setTrackbarPos("scroll", frame_name, self.track_pos)
        cv2.moveWindow(frame_name, 450, 30)
        cv2.resizeWindow(frame_name, 1500, 1000)
        self.current_img = -1000

    # necessary for trackbar
    def nothing(self, x):
        pass

    # selection Mouse function
    def select_angle(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.num_angle_pts < 3:
                self.angle_pts.append((x, y))
                self.num_angle_pts = len(self.angle_pts)
                if len(self.angle_pts) == 3:
                    self.angle_print = True
            else:
                self.angle_pts = []
                self.angle_print = False
                self.angle_pts.append((x, y))
                self.num_angle_pts = len(self.angle_pts)

    # selection Mouse function
    def select_ruler(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.num_ruler_pts < 2:
                self.ruler_pts.append((x, y))
                self.num_ruler_pts = len(self.ruler_pts)
                if len(self.ruler_pts) == 2:
                    self.ruler_print = True
            else:
                self.ruler_pts = []
                self.ruler_print = False
                self.ruler_pts.append((x, y))
                self.num_ruler_pts = len(self.ruler_pts)

    # selection Mouse function
    def select_point(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.old_points.append((x, y))
            self.point_selected = True

    # shutdown window
    def shut_down_windows(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def first_window(self):
        self.shut_down_windows()
        self.establish_frame('Scroll Through Using Trackbar', self.filename_total, 1, 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.track_pos)
        while self.first_window_up:
            self.track_pos = cv2.getTrackbarPos("scroll", 'Scroll Through Using Trackbar')
            if self.track_pos != self.current_img:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.track_pos)
                __, img = self.cap.read()
                if self.w_main != 800:
                    img = img[:, :]
                    # img = img[43:680, 64:821]
                elif self.w_main == 800 and sys.platform == "win32":
                    img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
                cv2.imshow('Scroll Through Using Trackbar', img)
                self.current_img = self.track_pos
            cv2.waitKey(1)

    def angle_window(self):
        self.shut_down_windows()
        self.establish_frame('Select Angle Points to Measure', self.filename_total, 1, 1)
        printed = False
        self.angle_print = False
        while self.angle_tool:
            self.track_pos = cv2.getTrackbarPos("scroll", 'Select Angle Points to Measure')
            if self.track_pos != self.current_img or printed == False or self.cleared == True:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.track_pos)
                __, img = self.cap.read()
                if self.w_main != 800:
                    img = img[:, :]
                    # img = img[43:680, 64:821]
                elif self.w_main == 800 and sys.platform == "win32":
                    img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
                if self.angle_pts:
                    for pnt in self.angle_pts:
                        cv2.circle(img, pnt, 3, (0, 255, 0), 1)
                if self.num_angle_pts == 3:
                    cv2.polylines(img, [np.array(self.angle_pts)], False, (0, 255, 0), 1)
                if self.angle_print:
                    np_angle_pts = np.array(self.angle_pts)
                    placement = np_angle_pts[1] + [10, 25]
                    placement_tuple = tuple(placement)
                    v1 = np_angle_pts[0] - np_angle_pts[1]
                    v2 = np_angle_pts[2] - np_angle_pts[1]
                    unit_v1 = v1 / LA.norm(v1)
                    unit_v2 = v2 / LA.norm(v2)
                    angle = round(math.acos(np.dot(unit_v1, unit_v2)) * 180 / math.pi, 1)
                    for_print = str(angle) + " deg"
                    cv2.putText(img, for_print, placement_tuple, self.font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    self.angle_print = False
                if self.pb_pts:
                    for pnt in self.pb_pts:
                        cv2.circle(img, pnt, 5, (0, 0, 255), 2)
                if self.ant_pts:
                    for pnt in self.ant_pts:
                        cv2.polylines(img, [np.array(self.ant_pts)], True, (0, 255, 255), 1)
                if self.post_pts:
                    for pnt in self.post_pts:
                        cv2.polylines(img, [np.array(self.post_pts)], True, (0, 100, 255), 1)
                if self.sling_pts:
                    for pnt in self.sling_pts:
                        cv2.polylines(img, [np.array(self.sling_pts)], True, (255, 0, 0), 1)

                cv2.imshow('Select Angle Points to Measure', img)
                self.current_img = self.track_pos
            self.cleared = False
            printed = self.angle_print
            cv2.waitKey(1)
        return

    def ruler_window(self):
        self.shut_down_windows()
        self.establish_frame('Select Points to Measure', self.filename_total, 1, 1)
        printed = False
        self.ruler_print = False
        while self.ruler_tool:
            self.track_pos = cv2.getTrackbarPos("scroll", 'Select Points to Measure')
            if self.track_pos != self.current_img or printed == False or self.cleared == True:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.track_pos)
                __, img = self.cap.read()
                if self.w_main != 800:
                    img = img[:, :]
                    # img = img[43:680, 64:821]
                elif self.w_main == 800 and sys.platform == "win32":
                    img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
                if self.ruler_pts:
                    for pnt in self.ruler_pts:
                        cv2.circle(img, pnt, 3, (0, 255, 0), 1)
                if self.num_ruler_pts == 2:
                    cv2.polylines(img, [np.array(self.ruler_pts)], True, (0, 255, 0), 1)
                if self.ruler_print:
                    np_ruler_pts = np.array(self.ruler_pts)
                    placement = np_ruler_pts[1] + [10, -5]
                    placement_tuple = tuple(placement)
                    distance = round(math.sqrt((np_ruler_pts[0, 0] - np_ruler_pts[1, 0]) ** 2 + (
                                np_ruler_pts[0, 1] - np_ruler_pts[1, 1]) ** 2) / self.scale, 2)
                    for_print = str(distance) + " cm"
                    cv2.putText(img, for_print, placement_tuple, self.font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    self.ruler_print = False
                if self.pb_pts:
                    for pnt in self.pb_pts:
                        cv2.circle(img, pnt, 5, (0, 0, 255), 2)
                if self.ant_pts:
                    for pnt in self.ant_pts:
                        cv2.polylines(img, [np.array(self.ant_pts)], True, (0, 255, 255), 1)
                if self.post_pts:
                    for pnt in self.post_pts:
                        cv2.polylines(img, [np.array(self.post_pts)], True, (0, 100, 255), 1)
                cv2.imshow('Select Points to Measure', img)
                self.current_img = self.track_pos
            self.cleared = False
            printed = self.ruler_print
            cv2.waitKey(1)
        return

    def selection_window(self):
        self.shut_down_windows()
        self.establish_frame('Selection & Meshing', self.filename_total, 1, 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.track_pos)
        _, prev = self.cap.read()
        if self.w_main != 800:
            prev = prev[:, :]
            # prev = prev[43:680, 64:821]
        elif self.w_main == 800 and sys.platform == "win32":
            prev = cv2.resize(prev, (0, 0), fx=0.75, fy=0.75)
        cv2.imshow('Selection & Meshing', prev)
        cv2.waitKey(1)
        while self.kill_loop:
            self.track_pos = cv2.getTrackbarPos("scroll", 'Selection & Meshing')
            if self.point_selected or self.track_pos != self.current_img or self.cleared == True:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.track_pos)
                _, prev = self.cap.read()
                if self.w_main != 800:
                    prev = prev[:, :]
                    # prev = prev[43:680, 64:821]
                elif self.w_main == 800 and sys.platform == "win32":
                    prev = cv2.resize(prev, (0, 0), fx=0.75, fy=0.75)
                if self.point_type == "PB":
                    self.pb_pts = self.old_points
                if self.point_type == "ANT":
                    self.ant_pts = self.old_points
                if self.point_type == "POST":
                    self.post_pts = self.old_points
                if self.point_type == "SLING":
                    self.sling_pts = self.old_points
                if self.pb_pts:
                    for pnt in self.pb_pts:
                        cv2.circle(prev, pnt, 5, (0, 0, 255), 2)
                if self.ant_pts:
                    for pnt in self.ant_pts:
                        cv2.polylines(prev, [np.array(self.ant_pts)], True, (0, 255, 255), 1)
                if self.post_pts:
                    for pnt in self.post_pts:
                        cv2.polylines(prev, [np.array(self.post_pts)], True, (0, 100, 255), 1)
                if self.sling_pts:
                    for pnt in self.sling_pts:
                        cv2.polylines(prev, [np.array(self.sling_pts)], True, (255, 0, 0), 1)
                if len(self.pb_pts) > 0:
                    self.pb_check = True
                if len(self.ant_pts) > 2:
                    self.ANT_check = True
                if len(self.post_pts) > 2:
                    self.POST_check = True
                if len(self.sling_pts) > 2:
                    self.SLING_check = True
                cv2.imshow('Selection & Meshing', prev)
                self.point_selected = False
            self.current_img = self.track_pos
            self.cleared = False
            cv2.waitKey(1)


if __name__ == '__main__':
    screen_app = wx.App()
    main_frame = MainFrame(None)
    screen_app.MainLoop()