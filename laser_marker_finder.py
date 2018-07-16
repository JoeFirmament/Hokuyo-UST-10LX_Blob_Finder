import sys
import tkinter as tk
import configparser
from hokuyolx import HokuyoLX
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
import time
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

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
marker_angual_interval = 0.01
marker_distance_interval = 100
blob_size_threshold = 200
range_point = [(area_left, area_far), (area_right, area_far),
               (area_right, area_near), (area_left, area_near)]
start_flag = False
errFlag = False
plot_limit = 5000
OSC_msg_raw = []


def get_plot_limit(x_left, x_right, y_far):
    if abs(x_left) > abs(x_right):
        return np.sqrt(x_left**2 + y_far**2)
    else:
        return np.sqrt(x_right**2 + y_far**2)


def cart2pol(x, y):  # 笛卡尔坐标转换为极坐标
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):  # 极坐标转换为笛卡尔坐标系
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def pol2cart_tuple(polar_point_list):
    # polar_point_list[0] as phi;[1] as rho
    x = polar_point_list[1] * np.cos(polar_point_list[0])
    y = polar_point_list[1] * np.sin(polar_point_list[0])
    # 这里把笛卡尔坐标系的y轴与极坐标系的极轴做对齐，方便观察理解。 笛卡尔坐标系逆时针旋转90°
    # x' = xcos(theta)+ysin(theta);y' = ycos(theta)-xsin(theta) ,theta = -90°
    return [-y, x]


def cart2pol_tuple(cart_point_list):
    rho = np.sqrt(cart_point_list[0]**2 + cart_point_list[1]**2)
    phi = np.arctan2(-cart_point_list[0],
                     cart_point_list[1])  # 这里x和y互换，直接旋转90°，变回原极坐标
    return (phi, rho)


def consecutive_pol(array, stepsize=100, angual_interval=0.02):
    '''
    将polar坐标系下的点list，根据Rho和theta进行分组
    array: 经过area 参数过滤的点的极坐标系下的坐标np.array
    stepsize: 前后两个元素的Rho差
    angual_interval:前后两个元素的角度（弧度）差'''

    scan_filter_array = list(np.hsplit(array, 2))  # 这里也可以用list 的zip(*list)方法
    scan_filter_theta = np.ravel(scan_filter_array[0])
    scan_filter_rho = np.ravel(scan_filter_array[1])
    rho_split = np.split(scan_filter_rho,
                         np.where(np.diff(scan_filter_rho) >= stepsize)[0] + 1)
    theta_split = np.split(
        scan_filter_theta,
        np.where(np.diff(scan_filter_rho) >= stepsize)[0] + 1)

    theta_split = np.split(
        scan_filter_theta,
        np.where(np.diff(scan_filter_theta) >= angual_interval)[0] + 1)
    rho_split = np.split(
        scan_filter_rho,
        np.where(np.diff(scan_filter_theta) >= angual_interval)[0] + 1)
    points = [n for n in range(len(rho_split))]
    for i in range(len(rho_split)):
        points[i] = list(zip(theta_split[i], rho_split[i]))
        points[i] = list(map(pol2cart_tuple, points[i]))
    return points


def consecutive_cart(array, stepsize=100):
    '''
    将cartesian坐标系下的点list，根据x和y进行分组
    array: 经过area 参数过滤的点的笛卡尔坐标系下的坐标np.array
    stepsize: 前后两个元素的x或y的差'''

    scan_filter_array = list(np.hsplit(array, 2))  # 这里也可以用list 的zip(*list)方法
    scan_filter_x = np.ravel(scan_filter_array[0])
    scan_filter_y = np.ravel(scan_filter_array[1])
    x_split = np.split(scan_filter_x,
                       np.where(np.diff(scan_filter_y) >= stepsize)[0] + 1)
    y_split = np.split(scan_filter_y,
                       np.where(np.diff(scan_filter_y) >= stepsize)[0] + 1)
    x_split = np.split(scan_filter_x,
                       np.where(np.diff(scan_filter_x) >= stepsize)[0] + 1)
    y_split = np.split(scan_filter_y,
                       np.where(np.diff(scan_filter_x) >= stepsize)[0] + 1)
    points = [n for n in range(len(x_split))]
    for i in range(len(x_split)):
        points[i] = list(zip(x_split[i], y_split[i]))
    return points


def continuous_filter(list, blob_size=200):
    '''去除list中只有少于等于一个元素的子列表，并去除尺寸小于blob_size的点集'''

    if len(list) > 1 and math.hypot(
            list[0][0] - list[len(list) - 1][0],
            list[0][1] - list[len(list) - 1][1]) >= int(blob_size):
        return True
    else:
        return False


def midpoint_finder(point_list):
    '''对分组的点集，每组取得一个平均值'''

    sum_x = 0
    sum_y = 0
    mid_points = [n for n in range(len(point_list))]
    for i in range(len(point_list)):
        for j in range(len(point_list[i])):
            sum_x = sum_x + point_list[i][j][0]
            sum_y = sum_y + point_list[i][j][1]
        mid_points[i] = [
            sum_x / len(point_list[i]), sum_y / len(point_list[i])
        ]
        sum_x = 0
        sum_y = 0
    return mid_points


def cart_fliter(cart_list):
    if cart_list[0] > int(area_left) and cart_list[0] < int(
            area_right) and cart_list[1] > int(
                area_near) and cart_list[1] < int(area_far):
        return True
    else:
        return False


def range_plot(range_list):  # 转换扫描范围笛卡尔坐标系点为极坐标，直接转换做了极轴和y轴的对齐
    range_polar = [(), (), (), ()]
    for i in range(len(range_list)):
        range_polar[i] = cart2pol(range_list[i][0], range_list[i][1])
    pointTheta, pointR = zip(*range_polar)  # Rhi和theta直接互换，对齐坐标系
    return list(pointTheta), list(pointR)


def change_state():
    global start_flag
    if start_flag is True:
        start_flag = False
    else:
        start_flag = True


def print_osc_msg():
    global msg_text
    global OSC_msg_raw
    msg_text.insert(1.0,
                    "/blob " + ' '.join(str(e) for e in OSC_msg_raw) + "\n")
    msg_text.tag_add("YELLOW", "1.0")


def update():
    global rad_selected
    global plot_on_off_rad
    global osc_on_off_rad
    global range_point
    global laser
    global plot
    global text
    global errFlag
    global plot_marker
    global start_time
    global mode_filter
    global mode_send
    global plot_limit
    global blob_size_threshold
    global area_far, area_left, area_near, area_right
    global OSC_msg_raw
    midpoints = []
    scan_pol_filter = [] 
    if errFlag is False:
        try:
            timestamp, scan = laser.get_filtered_dist(start=180, end=900, dmax=10000)
            # timestamp 是一个int类型，scan是[弧度，距离]的2维列表集合；
            # start=180，end=900，是step的范围，lx10的step是1080steps，1080*1/6得到180，1080*5/6=900，只需要正前方的0°~180°的识别范围
            # print(scan[0][0]) 得到弧度值 -1.57,对应极坐标的270°/-90°
            # print(scan[720][0]) 得到弧度值1.57，对应90°
            # print(scan.size) 所占字节数 2162
            # print(scan.ndim) 维度 2
            # print(scan.shape)  #形状  (1081,2)
            # scan 是一个 1081行2列的List
            # 极坐标转为笛卡尔坐标，return为笛卡尔坐标系list
            scan_cart = list(map(pol2cart_tuple, scan))
            scan_cart_filter = list(filter(
                cart_fliter, scan_cart))  # 根据area range值，得到需要的扫描区域的笛卡尔坐标系点list
            if len(scan_cart_filter) > 1:
                scan_pol_filter = list(map(
                    cart2pol_tuple, scan_cart_filter))  # 过滤后的扫描区笛卡尔坐标系值，转为极坐标点list
                if mode_filter.get() == 1:
                    points = consecutive_pol(
                        np.asarray(scan_pol_filter), int(marker_distance_interval),
                        float(marker_angual_interval)
                    )  # 把得到的点分组，根据rho的间隙，如果超过100mm，就分为另外一组，返回笛卡尔坐标系的点
                else:
                    points = consecutive_cart(
                        np.asarray(scan_cart_filter), int(marker_distance_interval))
                points = list(
                    filter(lambda x: continuous_filter(x, blob_size_threshold),
                           points))
                midpoints = midpoint_finder(points)
                if osc_on_off_rad.get() == 1:
                    OSC_msg_raw.clear()
                    OSC_msg_raw.append(len(midpoints))
                    for i in range(len(midpoints)):
                        if mode_send.get() == 1:
                            #为了配合之前的C++雷达代码，协议按照之前的方式来写。
                            deltaX = (abs(float(area_right))-abs(float(area_left)))/2.0
                            x = (float(midpoints[i][0])-deltaX)/float(abs(float(area_right)-float(area_left)))
                            y = ((float(midpoints[i][1])-float(area_near)))/float(abs(float(area_far)-float(area_near)))
                            OSC_msg_raw.append(x)
                            OSC_msg_raw.append(y)
                            OSC_msg_raw.append(float(0.0))
                        # 下面是原始物理位置的发送代码
                        elif mode_send.get() == 0:
                            OSC_msg_raw.append(float(midpoints[i][0]))
                            OSC_msg_raw.append(float(midpoints[i][1]))
                            OSC_msg_raw.append(float(0.0))
                    msg = oscbuildparse.OSCMessage("/blob", None, OSC_msg_raw)
                    osc_send(msg, "osc")
                    osc_process()
            if plot_on_off_rad.get() == 1:
                if len(midpoints) > 0:
                    points_pol = list(map(cart2pol_tuple, midpoints))
                    plot_marker.set_marker("+")
                    plot_marker.set_linestyle(" ")
                    plot_marker.set_markersize(50)
                    plot_marker.set_data(*np.array(points_pol).T)
                else:
                    plot_marker.set_data([], [])
                point_R, point_theta = range_plot(
                    range_point)  # 把笛卡尔坐标转为极坐标，返回两个list
                point_R.append(point_R[0])  # 把第一个元素再添加进list，为了显示时闭合。
                point_theta.append(point_theta[0])
                if rad_selected.get() == 2:  # line mode
                    plot.set_linestyle('--')
                    plot.set_marker("_")
                elif rad_selected.get() == 1:  # dot mode
                    plot.set_marker(",")
                    plot.set_linestyle(" ")
                    plot.set_markersize(16)
                if autoscale_rad.get() == 1:  # 自动缩放开启
                    if len(midpoints)>0:
                        plot.set_data(*np.array(scan_pol_filter).T)  # 根据area range值，过滤后的数据再转回极坐标，显示在图像上。
                    # ax.relim()
                        ax.set_rlim(0, plot_limit, 1)
                    # ax.autoscale_view(True,True,True)
                elif autoscale_rad.get() == 0:
                    ax.set_rmax(10000)
                    plot.set_data(
                        *scan.T
                    )  # line mode *scan.T 意思是将scan转置矩阵后，unpack ,把每个元素都代进函数
                plot_range.set_data([x - np.pi / 2 for x in point_theta],
                                    point_R)  # transform plot cw 90 degree
            text.set_text('Sensor Time: %d ms \nPorgram Time: %d s' %
                    (timestamp, time.time() - start_time))
            plt.draw()
        except Exception as e:
            msg_text.insert(1.0, "[UPDATE_ERR] "+str(e)+time.strftime("%X")+"\n")
            msg_text.tag_add("RED", "1.0")
    win.after(10, update)


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
    global marker_distance_interval
    global marker_angual_interval
    global range_point
    global plot_limit
    global blob_size_threshold

    cf = configparser.ConfigParser()
    cf.read("./config.conf")
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
    marker_angual_interval = cf.get("MARKER", "marker_angual_interval")
    marker_distance_interval = cf.get("MARKER", "marker_distance_interval")
    blob_size_threshold = cf.get("MARKER", "blob_size_threshold")
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
    sVar_marker_distance_interval.set(marker_distance_interval)
    sVar_marker_angual_interval.set(marker_angual_interval)
    range_point = [(area_left, area_far), (area_right, area_far),
                   (area_right, area_near), (area_left, area_near)]
    plot_limit = get_plot_limit(area_left, area_right, area_far)
    sVar_blob_size_threshold.set(blob_size_threshold)


def write_conf():
    global sensor_ip
    global sensor_port
    global osc_host_ip
    global osc_host_port
    global area_left
    global area_right
    global map_left
    global map_right
    global blob_size_threshold
    global marker_angual_interval
    global marker_distance_interval
    global laser
    global errFlag

    cf = configparser.ConfigParser()
    sensor_ip = sVar_sensor_ip.get()
    sensor_port = sVar_sensor_port.get()
    osc_host_ip = sVar_osc_host_ip.get()
    osc_host_port = sVar_osc_host_port.get()
    area_left = sVar_area_left.get()
    area_right = sVar_area_right.get()
    area_far = sVar_area_far.get()
    area_near = sVar_area_near.get()
    map_left = sVar_map_left.get()
    map_right = sVar_map_right.get()
    map_near = sVar_map_near.get()
    map_far = sVar_map_far.get()
    marker_angual_interval =sVar_marker_angual_interval.get()
    marker_distance_interval = sVar_marker_distance_interval.get()
    blob_size_threshold = sVar_blob_size_threshold.get()
    cf.read("config.conf")
    cf.set("SENSOR", "sensor_ip", sensor_ip)
    cf.set("SENSOR", "sensor_port", str(sensor_port))
    cf.set("OSC", "OSC_host_ip", osc_host_ip)
    cf.set("OSC", "OSC_host_port", str(osc_host_port))
    cf.set("AREA", "area_left", str(area_left))
    cf.set("AREA", "area_right", str(area_right))
    cf.set("AREA", "area_near", str(area_near))
    cf.set("AREA", "area_far", str(area_far))
    cf.set("AREA", "map_near", str(map_near))
    cf.set("AREA", "map_far", str(map_far))
    cf.set("AREA", "map_left", str(map_left))
    cf.set("AREA", "map_right", str(map_right))
    cf.set("MARKER", "marker_distance_interval", str(marker_distance_interval))
    cf.set("MARKER", "marker_angual_interval", str(marker_angual_interval))
    cf.set("MARKER", "blob_size_threshold", str(blob_size_threshold))

   # TO DO 
    #osc_terminate()
    #osc_startup()
    #osc_udp_client(str(osc_host_ip), int(osc_host_port), "osc")

    try:
        laser.close()
        laser = HokuyoLX(
                addr=(str(sensor_ip), int(sensor_port)),
                info=False,
                buf=1024,
                time_tolerance=1000,
                convert_time=False)
        msg_text.insert(1.0,
                    "[MSG]Sensor connected  " + time.strftime("%X") + "\n")
        msg_text.tag_remove("RED", "1.0")
        errFlag = False

    except Exception as e:
        msg_text.insert(1.0,
                        "[ERR]Sensor connect failed " + time.strftime("%X") + "\n")
        msg_text.tag_add("RED", "1.0")
        errFlag = True

    with open('config.conf', 'w') as configfile:
        cf.write(configfile)
    read_conf()


np.set_printoptions(suppress=True)

sys.setrecursionlimit(10000)  # python会报一个递归错误，这里设置最大递归数量 update  是一个递归函数

start_time = time.time()  # 测试运行时间
win = tk.Tk()  # 顶级容器
rad_selected = tk.IntVar()
osc_on_off_rad = tk.IntVar()
plot_on_off_rad = tk.IntVar()
autoscale_rad = tk.IntVar()
mode_filter = tk.IntVar()
mode_send = tk.IntVar()
mode_send.set(1)
autoscale_rad.set(0)
plot_on_off_rad.set(3)
osc_on_off_rad.set(1)
mode_filter.set(1)

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
sVar_marker_angual_interval = tk.StringVar(win, value=marker_angual_interval)
sVar_marker_distance_interval = tk.StringVar(
    win, value=marker_distance_interval)
sVar_blob_size_threshold = tk.StringVar(win, value=blob_size_threshold)

sensor_fr = tk.LabelFrame(win, text=" SENSOR NETWORK SETTING", bg='snow')
sensor_fr.grid(column=0, row=0, padx=2, pady=4)
osc_fr = tk.LabelFrame(win, text=" OSC NETWORK SETTING", bg='mint cream')
osc_fr.grid(column=0, row=1, padx=2, pady=4)
area_fr = tk.LabelFrame(win, text=" DETECTION RANGE", bg='papaya whip')
area_fr.grid(column=0, row=2, padx=2, pady=4)
map_fr = tk.LabelFrame(win, text=" MAP RANGE", bg='misty rose')
map_fr.grid(column=0, row=3, padx=2, pady=4)
marker_fr = tk.LabelFrame(win, text=" BLOB SETTING", bg='thistle1')
marker_fr.grid(column=0, row=4, padx=2, pady=4)
control_fr = tk.LabelFrame(win, text="YES!Quill", bg='light cyan')
control_fr.grid(column=0, row=5, padx=2, pady=4)
msg_fr = tk.LabelFrame(win, text="MSG", bg='light cyan')
msg_fr.grid(column=0, row=6, padx=2, pady=4)

read_conf()
# Start the system.
osc_startup()
# Make client channels to send packets
osc_udp_client(str(osc_host_ip), int(osc_host_port), "osc")

matplotlib.use('tkagg')
# plt.style.use("ggplot")
# plt.style.use('fivethirtyeight')
# plt.style.use('seaborn-white')
plt.style.use('seaborn-white')

plt.ion()
fig = plt.figure(figsize=(8, 8), dpi=100)
fig.canvas.set_window_title("By INT++ ")
ax = plt.subplot(111, projection='polar')
ax.set_title('Scanning Figure', fontstyle='italic')
ax.set_theta_zero_location('N')
plot = ax.plot([], [], ',')[0]
plot_marker = ax.plot([], [], ',')[0]
plot_range = ax.plot([], [], '--')[0]
text = plt.text(0, 1, '', transform=ax.transAxes)
ax.set_rmax(10000)
ax.grid(True)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

sensor_ip_L = tk.Label(
    sensor_fr, text="Sensor IP ", width=20, bg='snow').grid(
        column=0, row=0)
sensor_port_L = tk.Label(
    sensor_fr, text="Sensor Port", width=20, bg='snow').grid(
        column=0, row=1)
osc_ip_L = tk.Label(
    osc_fr, text="OSC Host IP ", width=20, bg='mint cream').grid(
        column=0, row=2)
osc_port_L = tk.Label(
    osc_fr, text="OSC Host Port", width=20, bg='mint cream').grid(
        column=0, row=3)
area_left_L = tk.Label(
    area_fr, text="Area Left(mm)", width=20, bg='papaya whip').grid(
        column=0, row=0)
area_right_L = tk.Label(
    area_fr, text="Area Right(mm)", width=20, bg='papaya whip').grid(
        column=0, row=1)
area_near_L = tk.Label(
    area_fr, text="Area Near(mm)", width=20, bg='papaya whip').grid(
        column=0, row=2)
area_far_L = tk.Label(
    area_fr, text="Area Far(mm)", width=20, bg='papaya whip').grid(
        column=0, row=3)
map_left_L = tk.Label(
    map_fr, text="Map Left", width=20, bg='misty rose').grid(
        column=0, row=0)
map_right_L = tk.Label(
    map_fr, text="Map Right", width=20, bg='misty rose').grid(
        column=0, row=1)
map_near_L = tk.Label(
    map_fr, text="Map Near", width=20, bg='misty rose').grid(
        column=0, row=2)
map_far_L = tk.Label(
    map_fr, text="Map Far", width=20, bg='misty rose').grid(
        column=0, row=3)
marker_angual_interval_L = tk.Label(
    marker_fr, text="Angular Interval(Radians)", width=20, bg='thistle1').grid(
        column=0, row=0)
marker_distance_interval_L = tk.Label(
    marker_fr, text="Distance Interval(mm) ", width=20, bg='thistle1').grid(
        column=0, row=1)
blob_size_threshold_L = tk.Label(
    marker_fr, text="Size Threshold(mm) ", width=20, bg='thistle1').grid(
        column=0, row=2)

sensor_ip_E = tk.Entry(
    sensor_fr, textvariable=sVar_sensor_ip, width=20, bg='snow').grid(
        column=1, row=0, sticky='w')
sensor_port_E = tk.Entry(
    sensor_fr, textvariable=sVar_sensor_port, width=20, bg='snow').grid(
        column=1, row=1, sticky='w')
osc_ip_E = tk.Entry(
    osc_fr, textvariable=sVar_osc_host_ip, width=20, bg='mint cream').grid(
        column=1, row=2, sticky='w')
osc_port_E = tk.Entry(
    osc_fr, textvariable=sVar_osc_host_port, width=20, bg='mint cream').grid(
        column=1, row=3, sticky='w')
area_left_E = tk.Entry(
    area_fr, textvariable=sVar_area_left, width=20, bg='papaya whip').grid(
        column=1, row=0, sticky='w')
area_right_E = tk.Entry(
    area_fr, textvariable=sVar_area_right, width=20, bg='papaya whip').grid(
        column=1, row=1, sticky='w')
area_near_E = tk.Entry(
    area_fr, textvariable=sVar_area_near, width=20, bg='papaya whip').grid(
        column=1, row=2, sticky='w')
area_far_E = tk.Entry(
    area_fr, textvariable=sVar_area_far, width=20, bg='papaya whip').grid(
        column=1, row=3, sticky='w')
map_left_E = tk.Entry(
    map_fr, textvariable=sVar_map_left, width=20, bg='misty rose').grid(
        column=1, row=0, sticky='w')
map_right_E = tk.Entry(
    map_fr, textvariable=sVar_map_right, width=20, bg='misty rose').grid(
        column=1, row=1, sticky='w')
map_near_E = tk.Entry(
    map_fr, textvariable=sVar_map_left, width=20, bg='misty rose').grid(
        column=1, row=2, sticky='w')
map_far_E = tk.Entry(
    map_fr, textvariable=sVar_map_right, width=20, bg='misty rose').grid(
        column=1, row=3, sticky='w')
marker_angual_interval_E = tk.Entry(
    marker_fr,
    textvariable=sVar_marker_angual_interval,
    width=20,
    bg='thistle1').grid(
        column=1, row=0, sticky='w')
marker_distance_interval_E = tk.Entry(
    marker_fr,
    textvariable=sVar_marker_distance_interval,
    width=20,
    bg='thistle1').grid(
        column=1, row=1, sticky='w')
blob_size_threshold_E = tk.Entry(
    marker_fr, textvariable=sVar_blob_size_threshold, width=20,
    bg='thistle1').grid(
        column=1, row=2, sticky='w')

load_conf_btn = tk.Button(
    control_fr, text="Read Conf", command=read_conf, bg='light cyan', width=15)
write_conf_btn = tk.Button(
    control_fr, text="Set Conf", command=write_conf, bg='light cyan', width=15)
start_btn = tk.Button(
    win,
    text="Grap a single OSC",
    command=print_osc_msg,
    bg='gold',
    padx=2,
    pady=4)

pixel_rad = tk.Radiobutton(
    control_fr,
    text='  Dot   ',
    value=1,
    variable=rad_selected,
    bg='light cyan',
    width=15)
line_rad = tk.Radiobutton(
    control_fr,
    text='  Line  ',
    value=2,
    variable=rad_selected,
    bg='light cyan',
    width=15)
plot_on_rad = tk.Radiobutton(
    control_fr,
    text='Plot  ON',
    value=1,
    variable=plot_on_off_rad,
    bg='light cyan',
    width=15)
plot_off_rad = tk.Radiobutton(
    control_fr,
    text='Plot OFF',
    value=0,
    variable=plot_on_off_rad,
    bg='light cyan',
    width=15)
osc_on_rad = tk.Radiobutton(
    control_fr,
    text='OSC   ON',
    value=1,
    variable=osc_on_off_rad,
    bg='light cyan',
    width=15)
osc_off_rad = tk.Radiobutton(
    control_fr,
    text='OSC  OFF',
    value=0,
    variable=osc_on_off_rad,
    bg='light cyan',
    width=15)
autoscale_on_rad = tk.Radiobutton(
    control_fr,
    text='AutoScaleON',
    value=1,
    variable=autoscale_rad,
    bg='light cyan',
    width=15)
autoscale_off_rad = tk.Radiobutton(
    control_fr,
    text='AutoScaleOFF',
    value=0,
    variable=autoscale_rad,
    bg='light cyan',
    width=15)
polar_filter_rad = tk.Radiobutton(
    control_fr,
    text='PolarFilter',
    value=1,
    variable=mode_filter,
    bg='light cyan',
    width=15)
cart_filter_rad = tk.Radiobutton(
    control_fr,
    text='CartFilter',
    value=0,
    variable=mode_filter,
    bg='light cyan',
    width=15)
map_mode_rad = tk.Radiobutton(
    control_fr,
    text='MapMode',
    value=1,
    variable=mode_send,
    bg='light cyan',
    width=15)
raw_mode_rad = tk.Radiobutton(
    control_fr,
    text='RawMode',
    value=0,
    variable=mode_send,
    bg='light cyan',
    width=15)

msg_text = tk.Text(msg_fr, width=40, height=8)
msg_text.grid(column=0, row=0)

load_conf_btn.grid(column=0, row=6)
write_conf_btn.grid(column=1, row=6)
start_btn.grid(column=0, row=7)
pixel_rad.grid(column=0, row=0)
line_rad.grid(column=1, row=0)
plot_on_rad.grid(column=0, row=1)
plot_off_rad.grid(column=1, row=1)
osc_on_rad.grid(column=0, row=2)
osc_off_rad.grid(column=1, row=2)
autoscale_on_rad.grid(column=0, row=3)
autoscale_off_rad.grid(column=1, row=3)
polar_filter_rad.grid(column=0, row=4)
cart_filter_rad.grid(column=1, row=4)
map_mode_rad.grid(column=0, row=5)
raw_mode_rad.grid(column=1, row=5)
msg_text.tag_config('RED', background='red')

try:
    laser = HokuyoLX(
        addr=(str(sensor_ip), int(sensor_port)),
        info=False,
        buf=1024,
        time_tolerance=1000,
        convert_time=False)
    msg_text.insert(1.0,
                    "[MSG]Sensor connected  " + time.strftime("%X") + "\n")
    msg_text.tag_remove("RED", "1.0")

except Exception as e:
    msg_text.insert(1.0,
                    "[ERR]Sensor connect failed " + time.strftime("%X") + "\n")
    msg_text.tag_add("RED", "1.0")
    errFlag = True

win.after(10, update)
win.mainloop()
