import sys
import configparser
from hokuyolx import HokuyoLX
import numpy as np
import math
import time
from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse
import argparse

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
    global plot_limit
    global blob_size_threshold
    global OSC_msg_raw

    try:
        if errFlag is True:
            return
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
            points = consecutive_pol(np.asarray(scan_pol_filter), int(marker_distance_interval),float(marker_angual_interval))  # 把得到的点分组，根据rho的间隙，如果超过100mm，就分为另外一组，返回笛卡尔坐标系的点
            points = list(
                filter(lambda x: continuous_filter(x, blob_size_threshold),
                       points))
            midpoints = midpoint_finder(points)
            OSC_msg_raw.clear()
            OSC_msg_raw.append(len(midpoints))
            for i in range(len(midpoints)):
                OSC_msg_raw.append(float(midpoints[i][0]))
                OSC_msg_raw.append(float(midpoints[i][1]))
            msg = oscbuildparse.OSCMessage("/blob", None, OSC_msg_raw)
            osc_send(msg, "aclientname")
            osc_process()
            print(OSC_msg_raw)
    except Exception as e:
        print("Sensor msg rev failed\n")


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

    range_point = [(area_left, area_far), (area_right, area_far),
                   (area_right, area_near), (area_left, area_near)]
    plot_limit = get_plot_limit(area_left, area_right, area_far)

np.set_printoptions(suppress=True)
sys.setrecursionlimit(10000)  # python会报一个递归错误，这里设置最大递归数量 update  是一个递归函数
parser = argparse.ArgumentParser()
parser.add_argument("-l","--log",help="print blob positon")
if args.log:
    print("withLog")
parser.paser_argrs()

read_conf()
# Start the system.
osc_startup()
# Make client channels to send packets.
osc_udp_client("127.0.0.1", 12345, "aclientname")
try:
    laser = HokuyoLX(
        addr=('192.168.0.10', 10940),
        info=False,
        buf=1024,
        time_tolerance=1000,
        convert_time=False)
except Exception as e:
    print("[ERR]Sensor connect failed " + time.strftime("%X") + "\n")
    errFlag = True
while True:
    update()
