import sys
import tkinter as tk
from tkinter import ttk
import configparser
from hokuyolx import HokuyoLX
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist import GridHelperCurveLinear

import matplotlib
import time
import threading
import numpy

sensor_ip = "192.168.0.10"
sensor_port = 1090
osc_host_ip = "127.0.0.1"
osc_host_port = 8000
area_left = -8000
area_right = 8000
area_near = 0
area_far = 2000
map_left = 1
map_right = -1
map_near = 0
map_far = 1
marker_amount = 3
marker_size_threshold = 100
continuePlotting = False
range_point = [(area_left, area_far), (area_right, area_far), (area_right, area_near), (area_left, area_near)]
start_flag = False


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def range_plot(range_list):
    range_polar = [(),  (), (), ()]
    for i in range(len(range_list)):
        range_polar[i] = cart2pol(range_list[i][0], range_list[i][1])
    pointTheta, pointR = zip(*range_polar)
    return list(pointTheta), list(pointR)


def change_state():
    global start_flag 
    if start_flag is True:
        start_flag = False
    else:
        start_flag = True


def update():
    global rad_selected
    global plot_on_off_rad
    global osc_on_off_rad
    global range_point
    global laser
    global plot
    global text

    try:
        timestamp, scan = laser.get_filtered_dist(dmax=10000)
    except Exception as e:
        msg_text.insert(1.0,"[ERR] "+str(e)+time.strftime("%X")+"\n")
        msg_text.tag_add("RED", "1.0") 
    if plot_on_off_rad.get() == 1:  # and plt.get_fignums(): #PlotOnRadio selected and has at least 1 figure
        point_R, point_theta = range_plot(range_point)  
        point_R.append(point_R[0])
        point_theta.append(point_theta[0])
        #plot2 = ax.plot([x-np.pi/2 for x in point_theta], point_R, '--', label="Range") #  transform plot cw 90 degree
        if rad_selected.get() == 2:  # line mode
            plot.set_linestyle('--')
            plot.set_marker("_")
        elif rad_selected.get() == 1:  #dot mode
            plot.set_marker(",")
            plot.set_linestyle(" ")
        plot.set_data(*scan.T) # line mode
        text.set_text('t: %d' % timestamp) 
        plt.draw()
        plt.pause(0.001)
    win.after(10,update)

        #while osc_on_off_rad.get() == 1 :



def game_start():
    change_state()
    #threading.Thread(target=update).start()



def read_conf():
    global sensor_ip
    global sensor_port
    global osc_host_ip
    global osc_host_port
    global area_left
    global area_right
    global area_near
    global area_far
    global map_left
    global map_right
    global map_near
    global map_far
    global marker_size_threshold
    global marker_amount
    global range_point

    cf = configparser.ConfigParser()
    cf.read("config.conf")
    sensor_ip = cf.get("SENSOR", "sensor_ip")
    sensor_port = cf.getint("SENSOR", "sensor_port")
    osc_host_ip = cf.get("OSC", "OSC_host_ip")
    osc_host_port = cf.getint("OSC", "OSC_host_port")
    area_left = int(cf.get("AREA", "area_left"))
    area_right = int(cf.get("AREA", "area_right"))
    area_near = int(cf.get("AREA", "area_near"))
    area_far = int(cf.get("AREA", "area_far"))
    map_left = cf.get("AREA", "map_left")
    map_right = cf.get("AREA", "map_right")
    map_near = cf.get("AREA", "map_near")
    map_far = cf.get("AREA", "map_far")
    marker_amount = cf.get("MARKER", "marker_amount")
    marker_size_threshold = cf.get("MARKER", "marker_size_threshold")
    sVar_sensor_ip.set(sensor_ip)
    sVar_sensor_port.set(sensor_port)
    sVar_area_left.set(area_left)
    sVar_area_right.set(area_right)
    sVar_area_near.set(area_near)
    sVar_area_far.set(area_far)
    sVar_map_left.set(map_left)
    sVar_map_right.set(map_right)
    sVar_map_near.set(map_near)
    sVar_map_far.set(map_far)
    sVar_osc_host_ip.set(osc_host_ip)
    sVar_osc_host_port.set(osc_host_port)
    sVar_marker_size_threshold.set(marker_size_threshold)
    sVar_marker_amount.set(marker_amount)
    range_point = [(area_left, area_far), (area_right, area_far), (area_right, area_near), (area_left, area_near)]


def write_conf():
    global sensor_ip
    global sensor_port
    global osc_host_ip
    global osc_host_port
    global area_left
    global area_right
    global map_left
    global map_right
    cf = configparser.ConfigParser()
    sensor_ip = sVar_sensor_ip.get()
    sensor_port = sVar_sensor_port.get()
    osc_host_ip = sVar_osc_host_ip.get()
    osc_host_port = sVar_osc_host_port.get()
    area_left = sVar_area_left.get()
    area_right = sVar_area_right.get()
    map_left = sVar_map_left.get()
    map_right = sVar_map_right.get()
    cf.read("config.conf")
    cf.set("SENSOR", "sensor_ip", sensor_ip)
    cf.set("SENSOR", "sensor_port", str(sensor_port))
    cf.set("OSC", "OSC_host_ip", osc_host_ip)
    cf.set("OSC", "OSC_host_port", str(osc_host_port))
    cf.set("AREA", "area_left", str(area_left))
    cf.set("AREA", "area_right", str(area_right))
    cf.set("AREA", "map_left", str(map_left))
    cf.set("AREA", "map_right", str(map_right))
    with open('config.conf', 'w') as configfile:
        cf.write(configfile)

sys.setrecursionlimit(10000) # python会报一个递归错误，这里设置最大递归数量 update  是一个递归函数
win = tk.Tk()  # 顶级容器
rad_selected = tk.IntVar()
osc_on_off_rad = tk.IntVar()
plot_on_off_rad = tk.IntVar()
plot_on_off_rad.set(3)
osc_on_off_rad.set(3)

win.title("UST-10 to OSC by Quill")
win.configure(background='azure')

sVar_sensor_ip = tk.StringVar(win, value=sensor_ip)
sVar_sensor_port = tk.StringVar(win, value=sensor_port)
sVar_osc_host_ip = tk.StringVar(win, value=osc_host_ip)
sVar_osc_host_port = tk.StringVar(win, value=osc_host_port)
sVar_area_left = tk.StringVar(win, value=area_left)
sVar_area_right = tk.StringVar(win, value=area_right)
sVar_map_left = tk.StringVar(win, value=map_left)
sVar_map_right = tk.StringVar(win, value=map_right)
sVar_area_near = tk.StringVar(win, value=area_near)
sVar_map_near = tk.StringVar(win, value=map_near)
sVar_map_far = tk.StringVar(win, value=map_far)
sVar_area_far = tk.StringVar(win, value=area_far)
sVar_marker_amount = tk.StringVar(win, value=marker_amount)
sVar_marker_size_threshold = tk.StringVar(win, value=marker_size_threshold)

sensor_fr = tk.LabelFrame(win,text=" SENSOR NETWORK SETTING",       bg='snow')
sensor_fr.grid(column=0, row=0, padx=2, pady=4) 
osc_fr =    tk.LabelFrame(win, text=" OSC NETWORK SETTING",         bg='mint cream')
osc_fr.grid(column=0, row=1, padx=2, pady=4) 
area_fr =   tk.LabelFrame(win, text=" DETECTION RANGE",             bg='papaya whip')
area_fr.grid(column=0, row=2, padx=2, pady=4) 
map_fr =    tk.LabelFrame(win,text=" MAP RANGE",                    bg='misty rose')
map_fr.grid(column=0, row=3, padx=2, pady=4) 
marker_fr = tk.LabelFrame(win, text=" MARKER SETTING",              bg='thistle1')
marker_fr.grid(column=0, row=4, padx=2, pady=4) 
control_fr = tk.LabelFrame(win, text="YES!Quill",                        bg='light cyan')
control_fr.grid(column=0, row=5, padx=2, pady=4) 
msg_fr = tk.LabelFrame(win, text="MSG",                        bg='light cyan')
msg_fr.grid(column=0, row=6, padx=2, pady=4) 


read_conf()
matplotlib.use('tkagg')
#plt.style.use("ggplot")  
#plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-white')
plt.style.use('seaborn-white')


plt.ion()
fig = plt.figure(figsize=(8, 8),dpi=80) 
fig.canvas.set_window_title("By INT++ ")

ax = plt.subplot(111, projection='polar')

ax.set_title('Scanning Figure', fontstyle='italic')

ax.set_theta_zero_location('N')
plot = ax.plot([], [], ',')[0]
point_R, point_theta = range_plot(range_point)  
point_R.append(point_R[0])
point_theta.append(point_theta[0])
plot_range = ax.plot([x-np.pi/2 for x in point_theta], point_R, '--', label="Range")  #  transform plot cw 90 degree
text = plt.text(0, 1, '', transform=ax.transAxes)
ax.set_rmax(10000)
ax.grid(True)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()


sensor_ip_L = tk.Label(sensor_fr, text="Sensor IP: ", width=20, bg='snow').grid(column=0, row=0)
sensor_port_L = tk.Label(sensor_fr, text="Sensor Port:", width=20, bg='snow').grid(column=0, row=1)
osc_ip_L = tk.Label(osc_fr, text="OSC Host IP: ", width=20, bg='mint cream').grid(column=0, row=2)
osc_port_L = tk.Label(osc_fr, text="OSC Host Port:", width=20, bg='mint cream').grid(column=0, row=3)
area_left_L = tk.Label(area_fr, text="Area Left:", width=20, bg='papaya whip').grid(column=0, row=0)
area_right_L = tk.Label(area_fr, text="Area Right:", width=20, bg='papaya whip').grid(column=0, row=1)
area_near_L = tk.Label(area_fr, text="Area Near:", width=20, bg='papaya whip').grid(column=0, row=2)
area_far_L = tk.Label(area_fr, text="Area Far:", width=20, bg='papaya whip').grid(column=0, row=3)
map_left_L = tk.Label(map_fr, text="Map Left:", width=20, bg='misty rose').grid(column=0, row=0)
map_right_L = tk.Label(map_fr, text="Map Right:", width=20, bg='misty rose').grid(column=0, row=1)
map_near_L = tk.Label(map_fr, text="Map Near:", width=20, bg='misty rose').grid(column=0, row=2)
map_far_L = tk.Label(map_fr, text="Map Far:", width=20, bg='misty rose').grid(column=0, row=3)
marker_amount_L = tk.Label(marker_fr, text="Marker Amount:", width=20, bg='thistle1').grid(column=0, row=0)
marker_size_threshold_L = tk.Label(marker_fr, text="Marker Size Threshold:", width=20, bg='thistle1').grid(column=0, row=1)


sensor_ip_E = tk.Entry(sensor_fr,  textvariable=sVar_sensor_ip, width=20, bg='snow').grid(column=1, row=0, sticky='w')
sensor_port_E = tk.Entry(sensor_fr, textvariable=sVar_sensor_port, width=20, bg='snow').grid(column=1, row=1, sticky='w')
osc_ip_E = tk.Entry(osc_fr, textvariable=sVar_osc_host_ip, width=20, bg='mint cream').grid(column=1, row=2, sticky='w')
osc_port_E = tk.Entry(osc_fr, textvariable=sVar_osc_host_port, width=20, bg='mint cream').grid(column=1, row=3, sticky='w')
area_left_E = tk.Entry(area_fr, textvariable=sVar_area_left, width=20, bg='papaya whip').grid(column=1, row=0, sticky='w')
area_right_E = tk.Entry(area_fr, textvariable=sVar_area_right, width=20, bg='papaya whip').grid(column=1, row=1, sticky='w')
area_near_E = tk.Entry(area_fr, textvariable=sVar_area_near, width=20, bg='papaya whip').grid(column=1, row=2, sticky='w')
area_far_E = tk.Entry(area_fr, textvariable=sVar_area_far, width=20, bg='papaya whip').grid(column=1, row=3, sticky='w')
map_left_E = tk.Entry(map_fr, textvariable=sVar_map_left, width=20, bg='misty rose').grid(column=1, row=0, sticky='w')
map_right_E = tk.Entry(map_fr, textvariable=sVar_map_right, width=20, bg='misty rose').grid(column=1, row=1, sticky='w')
map_near_E = tk.Entry(map_fr, textvariable=sVar_map_left, width=20, bg='misty rose').grid(column=1, row=2, sticky='w')
map_far_E = tk.Entry(map_fr, textvariable=sVar_map_right, width=20, bg='misty rose').grid(column=1, row=3, sticky='w')
marker_amount_E = tk.Entry(marker_fr, textvariable=sVar_marker_amount, width=20, bg='thistle1').grid(column=1, row=0, sticky='w')
marker_size_threshold_E = tk.Entry(marker_fr, textvariable=sVar_marker_size_threshold, width=20, bg='thistle1').grid(column=1, row=1, sticky='w')

load_conf_btn = tk.Button(control_fr, text="Read Conf", command=read_conf, bg='light cyan', width=15)
write_conf_btn = tk.Button(control_fr, text="Set Conf", command=write_conf, bg='light cyan', width=15)
start_btn = tk.Button(win, text="Grap a single OSC", command=game_start, bg='gold',padx=2, pady=4)
#udp_btn = tk.Button(control_fr, text="OSC Start/Stop", command=lambda: update(laser, plot,text), bg='light cyan', width=15)

#line_btn = tk.Button(win, text="Line/Dots", command=change_isLine)
pixel_rad = tk.Radiobutton(control_fr,   text= '  Dot   ', value=1, variable=rad_selected, bg='light cyan', width=15)
line_rad = tk.Radiobutton(control_fr,    text= '  Line  ', value=2, variable=rad_selected, bg='light cyan', width=15)
plot_on_rad = tk.Radiobutton(control_fr, text= 'Plot  ON', value=1, variable=plot_on_off_rad, bg='light cyan', width=15)
plot_off_rad = tk.Radiobutton(control_fr, text='Plot OFF', value=0, variable=plot_on_off_rad, bg='light cyan', width=15)
osc_on_rad = tk.Radiobutton(control_fr,   text='OSC   ON', value=1, variable=osc_on_off_rad, bg='light cyan', width=15)
osc_off_rad = tk.Radiobutton(control_fr, text= 'OSC  OFF', value=0, variable=osc_on_off_rad, bg='light cyan', width=15)
msg_text = tk.Text(msg_fr,width=40,height=8)
msg_text.grid(column=0, row=0)


load_conf_btn.grid(column=0, row=3)
write_conf_btn.grid(column=1, row=3)
start_btn.grid(column=0, row=7)
#udp_btn.grid(column=1, row=2)

#line_btn.grid(column=4, row=8)
pixel_rad.grid(column=0, row=0)
line_rad.grid(column=1, row=0)
plot_on_rad.grid(column=0,row=1)
plot_off_rad.grid(column=1,row=1)
osc_on_rad.grid(column=0,row=2)
osc_off_rad.grid(column=1,row=2)
msg_text.tag_config('RED', background='red')

try:
    laser = HokuyoLX(addr=('192.168.0.10', 10940),info=False)
    msg_text.insert(1.0,"[MSG]Sensor connected  "+time.strftime("%X")+"\n")
    msg_text.tag_remove("RED", "1.0") 


except  Exception as e:
    msg_text.insert(1.0,"[ERR]Sensor connect failed "+time.strftime("%X")+"\n")
    msg_text.tag_add("RED", "1.0") 

win.after(10,update)
win.mainloop()
