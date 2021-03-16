import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# 画图步骤
# 1. 定义x, y数据
# 2. 画图，设置样式
# 3. 设置坐标轴

'''
1. 绘制简单直线
'''
def lesson1():
    x = np.linspace(-3,3,50) # x范围是[-3,3]，取50个点
    y1 = 2 * x + 1
    y2 = x * 2

    plt.figure() # 之后所有的操作都属于这个fig，知道新建一个figure
    plt.plot(x, y1)
    plt.show()

    plt.figure(num=3, figsize=(8,5)) # 设置编号，窗口比例
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--') # 设置颜色，线宽，风格
    plt.show()

'''
2. 设置坐标轴的文字描述和标尺
'''
def lesson2():
    x = np.linspace(-3, 3, 50)  # x范围是[-3,3]，取50个点
    y1 = 2 * x + 1
    y2 = x * 2

    plt.figure() # 设置编号，窗口比例
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--') # 设置颜色，线宽，风格

    plt.xlim((-1, 2)) # 设置x轴的取值范围
    plt.ylim((-2, 3)) # 设置y轴的取值范围
    plt.xlabel('I am x') # 设置x坐标轴的label
    plt.ylabel('I am y') # 设置y坐标轴的label

    plt.xticks(np.linspace(-1, 2, 5)) # 设置x轴的刻度，范围[-1,2]，内有5个刻度
    plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$']) # 设置y轴的刻度和对应名称
    plt.show()

'''
3. 设置坐标轴的位置，设置边框，平移x轴和y轴的位置
'''
def lesson3():
    x = np.linspace(-3, 3, 50)  # x范围是[-3,3]，取50个点
    y1 = 2 * x + 1
    y2 = x * 2

    plt.figure()  # 设置编号，窗口比例
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')  # 设置颜色，线宽，风格

    plt.xlim((-1, 2))  # 设置x轴的取值范围
    plt.ylim((-2, 3))  # 设置y轴的取值范围
    plt.xlabel('I am x')  # 设置x坐标轴的label
    plt.ylabel('I am y')  # 设置y坐标轴的label

    plt.xticks(np.linspace(-1, 2, 5))  # 设置x轴的刻度，范围[-1,2]，内有5个刻度
    plt.yticks([-2, -1.8, -1, 1.22, 3],
               [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])  # 设置y轴的刻度和对应名称

    # gca = 'get current axis'
    ax = plt.gca()
    ax.spines['right'].set_color('none') # 让右边的坐标轴消失
    ax.spines['top'].set_color('none') # 让上面的坐标轴消失

    ax.xaxis.set_ticks_position('bottom') #用下面的坐标轴代替x
    # ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

    ax.spines['bottom'].set_position(('data', 0)) # 将x轴平移到y轴0的位置
    # the 1st is in 'outward' | 'axes' | 'data'
    # axes: percentage of y axis
    # data: depend on y data

    ax.yaxis.set_ticks_position('left') # 用左边的坐标轴代替y轴
    # ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]

    ax.spines['left'].set_position(('data', 0)) # 将y轴平移到x轴0的位置
    plt.show()
    pass

'''
4. 添加图例
'''
def lesson4():
    x = np.linspace(-3, 3, 50)  # x范围是[-3,3]，取50个点
    y1 = 2 * x + 1
    y2 = x * 2

    plt.figure()  # 设置编号，窗口比例
    l1,=plt.plot(x, y2, label='up') # 这个label就是标记
    l2,=plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='down')  # 设置颜色，线宽，风格

    plt.xlim((-1, 2))  # 设置x轴的取值范围
    plt.ylim((-2, 3))  # 设置y轴的取值范围
    plt.xlabel('I am x')  # 设置x坐标轴的label
    plt.ylabel('I am y')  # 设置y坐标轴的label

    plt.xticks(np.linspace(-1, 2, 5))  # 设置x轴的刻度，范围[-1,2]，内有5个刻度
    plt.yticks([-2, -1.8, -1, 1.22, 3],
               [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])  # 设置y轴的刻度和对应名称

    plt.legend(loc='best', handles=[l1,l2,], labels=['aaa', 'bbb']) # 显示图例
    # loc: upper | right | best
    # handles: 设置哪几个线
    # labels: handles这些线的名字
    plt.show()
    pass

'''
5. 添加标注：线上的某个点
'''
def lesson5():
    x = np.linspace(-3, 3, 50)  # x范围是[-3,3]，取50个点
    y = 2 * x + 1

    plt.figure(num=1,figsize=(8,5),)  # 设置编号，窗口比例
    plt.plot(x, y)

    # gca = 'get current axis'
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # 让右边的坐标轴消失
    ax.spines['top'].set_color('none')  # 让上面的坐标轴消失

    ax.xaxis.set_ticks_position('bottom')  # 用下面的坐标轴代替x
    ax.spines['bottom'].set_position(('data', 0))  # 将x轴平移到y轴0的位置

    ax.yaxis.set_ticks_position('left')  # 用左边的坐标轴代替y轴
    ax.spines['left'].set_position(('data', 0))  # 将y轴平移到x轴0的位置

    x0 = 1
    y0 = 2 * x0 + 1
    plt.scatter(x0, y0, s=50, color='b') # 绘制点，尺寸，颜色
    plt.plot([x0,x0],[y0,0], 'k--', lw=2.5) # 绘制线段，黑色 虚线 线宽

    # method 1
    # method 1:
    #####################
    plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), # 标注点
                 xycoords='data', xytext=(+30, -30), # 文本偏移
                 textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2")) # 箭头属性

    # method 2:
    ########################
    plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
             fontdict={'size': 16, 'color': 'r'}) # 直接打字

    plt.show()
    pass

'''
6. 当线遮挡住了xy轴上的label
'''
def lesson6():
    x = np.linspace(-3, 3, 50)
    y = 0.1 * x

    plt.figure()
    plt.plot(x, y, linewidth=10, zorder=1)  # set zorder for ordering the plot in plt 2.0.2 or higher
    plt.ylim(-2, 2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    """"""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        # set zorder for ordering the plot in plt 2.0.2 or higher
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, zorder=2)) # 背景色，边界色，不透明度
    plt.show()

'''
7. 绘制散点图
'''
def lesson7():
    n = 1024  # 数据量
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)  # 根据数据确定颜色

    plt.scatter(X, Y, s=75, c=T, alpha=.5) # 尺寸，颜色，不透明度

    plt.xlim(-1.5, 1.5)
    plt.xticks(())  # 忽略xticks
    plt.ylim(-1.5, 1.5)
    plt.yticks(())  # 忽略yticks

    plt.show()

'''
8. 绘制柱状图
'''
def lesson8():
    n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) *  np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) *  np.random.uniform(0.5, 1.0, n)

    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white') # 柱子的XY，背景色，边框色
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    for x, y in zip(X, Y1): # 设置标注
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom') # 标注位置，文本，水平对齐方式，垂直对齐方式

    for x, y in zip(X, Y2):
        # ha: horizontal alignment
        # va: vertical alignment
        plt.text(x, -y - 0.05, '%.2f' % y, ha='center', va='top')

    plt.xlim(-.5, n)
    plt.xticks(()) # 忽略x
    plt.ylim(-1.25, 1.25)
    plt.yticks(()) # 忽略y

    plt.show()
    pass

'''
9. 绘制等高线
'''
def lesson9():
    def f(x, y):
        # the height function
        return (1 - x / 2 + x ** 5 + y ** 3) *  np.exp(-x ** 2 - y ** 2)

    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y) # x y 对应网格

    # use plt.contourf to filling contours
    # X, Y and value for (X,Y) point
    plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot) # 绘制颜色

    # use plt.contour to add contour lines
    C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5) # 等高线
    # adding label
    plt.clabel(C, inline=True, fontsize=10) # 等高线标注

    plt.xticks(())
    plt.yticks(())
    plt.show()

'''
9. 绘制色块图，如热力图
'''
def lesson10():
    # image data
    a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
                  0.365348418405, 0.439599930621, 0.525083754405,
                  0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)

    """
    for the value of "interpolation", check this:
    http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
    for the value of "origin"= ['upper', 'lower'], check this:
    http://matplotlib.org/examples/pylab_examples/image_origin.html
    """
    plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower') # 绘图形式， 颜色对应表，反转或不反转
    plt.colorbar(shrink=.92) # 颜色条的显示比例

    plt.xticks(())
    plt.yticks(())
    plt.show()

'''
10. 3D图
'''
def lesson11():
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    # X, Y value
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    # height value
    Z = np.sin(R)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow')) # 行跨度和列跨度
    """
    ============= ================================================
            Argument      Description
            ============= ================================================
            * X* , * Y* , * Z*  Data values as 2D arrays
            * rstride*      Array row stride (step size), defaults to 10
            * cstride*      Array column stride (step size), defaults to 10
            * color*        Color of the surface patches
            * cmap*         A colormap for the surface patches.
            * facecolors*   Face colors for the individual patches
            * norm*         An instance of Normalize to map values to colors
            * vmin*         Minimum value to map
            * vmax*         Maximum value to map
            * shade*        Whether to shade the facecolors
            ============= ================================================
    """

    # I think this is different from plt12_contours
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow')) # 压缩到那个面，那个面的哪个值
    """
    ==========  ================================================
            Argument    Description
            ==========  ================================================
            * X* , * Y* ,   Data values as numpy.arrays
            * Z* 
            * zdir*       The direction to use: x, y or z (default)
            * offset*     If specified plot a projection of the filled contour
                        on this position in plane normal to zdir
            ==========  ================================================
    """

    ax.set_zlim(-2, 2)

    plt.show()

'''
12. 多合一显示
'''
def lesson12():
    # example 1:
    ###############################
    plt.figure(figsize=(6, 4))
    # plt.subplot(n_rows, n_cols, plot_num)
    plt.subplot(2, 2, 1)
    plt.plot([0, 1], [0, 1])

    plt.subplot(222)
    plt.plot([0, 1], [0, 2])

    plt.subplot(223)
    plt.plot([0, 1], [0, 3])

    plt.subplot(224)
    plt.plot([0, 1], [0, 4])

    plt.tight_layout()

    # example 2:
    ###############################
    plt.figure(figsize=(6, 4))
    # plt.subplot(n_rows, n_cols, plot_num)
    plt.subplot(2, 1, 1)
    # figure splits into 2 rows, 1 col, plot to the 1st sub-fig
    plt.plot([0, 1], [0, 1])

    plt.subplot(234)
    # figure splits into 2 rows, 3 col, plot to the 4th sub-fig
    plt.plot([0, 1], [0, 2])

    plt.subplot(235)
    # figure splits into 2 rows, 3 col, plot to the 5th sub-fig
    plt.plot([0, 1], [0, 3])

    plt.subplot(236)
    # figure splits into 2 rows, 3 col, plot to the 6th sub-fig
    plt.plot([0, 1], [0, 4])

    plt.tight_layout()
    plt.show()


'''
13. 多合一显示的3种方式，12是一种
'''
def lesson13():
    # method 1: subplot2grid
    ##########################
    plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)  # 多少行多少列，起点(0-indexed)，列跨度，行跨度默认1
    ax1.plot([1, 2], [1, 2])
    ax1.set_title('ax1_title')
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax4.scatter([1, 2], [2, 2])
    ax4.set_xlabel('ax4_x')
    ax4.set_ylabel('ax4_y')
    ax5 = plt.subplot2grid((3, 3), (2, 1))

    # method 2: gridspec
    #########################
    import matplotlib.gridspec as gridspec

    plt.figure()
    gs = gridspec.GridSpec(3, 3) # 通过索引的方式来绘制小图
    # use index from 0
    ax6 = plt.subplot(gs[0, :])
    ax7 = plt.subplot(gs[1, :2])
    ax8 = plt.subplot(gs[1:, 2])
    ax9 = plt.subplot(gs[-1, 0])
    ax10 = plt.subplot(gs[-1, -2])

    # method 3: easy to define structure
    ####################################
    f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax11.scatter([1, 2], [1, 2])

    plt.tight_layout()
    plt.show()

'''
14. 图中图
'''
def lesson131():
    fig = plt.figure()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 3, 4, 2, 5, 8, 6]

    # below are all percentage
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8 # 左下来定位，宽度和高度决定尺寸
    ax1 = fig.add_axes([left, bottom, width, height])  # 往图里添加ax，这个图就可以用ax进行定制
    ax1.plot(x, y, 'r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('title')

    ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
    ax2.plot(y, x, 'b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('title inside 1')

    # different method to add axes
    ####################################
    plt.axes([0.6, 0.2, 0.25, 0.25]) # 另一种添加ax的方式，这种方式不需要明确ax对象，后续操作都默认跟着plt.axes
    plt.plot(y[::-1], x, 'g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('title inside 2')

    plt.show()


'''
14. 绘制双坐标轴，两个图有相同的x轴，y轴不同
'''
def lesson14():
    x = np.arange(0, 10, 0.1)
    y1 = 0.05 * x ** 2
    y2 = -1 *  y1

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()  # 将ax1的y坐标轴进行镜面反向，X轴保持共享
    ax1.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')

    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y1 data', color='g')
    ax2.set_ylabel('Y2 data', color='b')

    plt.show()


'''
15. 绘制动画
'''
def lesson15():
    from matplotlib import animation
    fig, ax = plt.subplots()

    x = np.arange(0, 2 *  np.pi, 0.01)
    line, = ax.plot(x, np.sin(x))

    def animate(i):
        line.set_ydata(np.sin(x + i / 10.0))  # 更新数据
        return line, # 返回一个列表

    # Init only required for blitting to give a clean slate.
    def init():
        line.set_ydata(np.sin(x)) # 初始化函数
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    # blit=True dose not work on Mac, set blit=False
    # interval=更新频率
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,
                                  interval=20, blit=False)

    plt.show()

# 样例
def variant_rmse_taxi():
    '''
    taxi variant bar for rmse
    :return:
    '''
    n = 3
    X = np.arange(n)
    Y = [18.55,17.08,16.64]
    plt.figure(1, figsize=(5, 5))
    ax = plt.gca()

    # x y major label and minor label
    # xmajorLocator = plt.MultipleLocator(1)  # set x major label as 1 times
    # xminorLocator = plt.MultipleLocator(0.5)  # set x minor label as 0.5 times
    # ax.xaxis.set_major_locator(xmajorLocator) # set x major locator
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ymajorLocator = plt.MultipleLocator(0.5)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.5)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="both", linestyle='--', which='minor')
    plt.xlim((-1, n))
    plt.xticks(())
    plt.ylim(16,19)
    plt.yticks(np.linspace(16, 19, 7), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#2E75B6', edgecolor='k', hatch="--", label="GL-TCN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#4169E1', edgecolor='k', hatch="///", label="GL-TCN-nog", alpha=0.8)
    plt.bar(X[2], Y[2], color='#6495ED', edgecolor='k', hatch="\\\\\\", label="GL-TCN", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('Method', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    plt.legend(loc='best', fontsize=13)
    plt.savefig(r'D:\研究生\科研\方案和写作\写作：stgln\ieee-stgln\figure\v_r_taxi.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    pass


def len_taxi():
    x = np.linspace(1, 7, 7)
    global_y = [17.1, 16.64, 16.94, 17.02, 17.1, 17.2, 17.66]
    local_y = [17.29, 17.2, 17.02, 16.94, 16.7, 17.1, 17.66]

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ymajorLocator = plt.MultipleLocator(0.4)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.4)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.grid(axis="both", linestyle='--', which='minor')

    plt.plot(x, global_y, color='#FF8C00', linewidth=2.0, linestyle='-',  marker='v', label='global data')
    plt.plot(x, local_y, color='#4169E1', linewidth=2.0, linestyle='-', marker='o', label='local data')

    plt.xlim((0.5, 7.5))
    plt.ylim((16, 18))
    plt.xticks(np.linspace(1, 7, 7), fontsize=18)
    plt.yticks(np.linspace(15.6, 18, 7), fontsize=18)

    plt.xlabel('The length of data', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    plt.legend(loc='lower right', fontsize=13)
    plt.savefig('E:\\研究生\\科研\方案\\stgln\\ieee-stgln\\figure\\l_taxi.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def hyper_res():
    x = np.linspace(0,5,5)
    y = [17.28, 16.95, 17.07, 16.68, 17.33]

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ymajorLocator = plt.MultipleLocator(0.4)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.2)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.grid(axis="both", linestyle='--', which='minor')

    plt.plot(x, y, color='#4169E1', linewidth=2.0, linestyle='-',  marker='o')

    plt.xlim((-0.5, 5.5))
    plt.ylim((16.4, 17.4))
    plt.xticks(np.linspace(0,5,5), [1, 4, 8, 12, 24], fontsize=18)
    plt.yticks(np.linspace(16.4, 17.4, 6), fontsize=18)

    plt.xlabel('The number of residual units', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    plt.savefig('E:\\研究生\\科研\方案\\stgln\\ieee-stgln\\figure\\h_res.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def variant_rmse_bike():
    '''
    taxi variant bar for rmse
    :return:
    '''
    n = 3
    X = np.arange(n)
    Y = [5.91,5.61,5.46]

    plt.figure(1, figsize=(5, 5))
    ax = plt.gca()

    # x y major label and minor label
    # xmajorLocator = plt.MultipleLocator(1)  # set x major label as 1 times
    # xminorLocator = plt.MultipleLocator(0.5)  # set x minor label as 0.5 times
    # ax.xaxis.set_major_locator(xmajorLocator) # set x major locator
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ymajorLocator = plt.MultipleLocator(0.2)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.2)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="both", linestyle='--', which='minor')
    plt.xlim((-1, n))
    plt.xticks(())
    plt.ylim(5,6.2)
    plt.yticks(np.linspace(5, 6.2, 7), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#228B22', edgecolor='k', hatch="--", label="GL-TCN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#2E8B57', edgecolor='k', hatch="///", label="GL-TCN-nog", alpha=0.8)
    plt.bar(X[2], Y[2], color='#3CB371', edgecolor='k', hatch="\\\\\\", label="GL-TCN", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('Method', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    plt.legend(loc='best', fontsize=13)
    plt.savefig(r'D:\研究生\科研\方案和写作\写作：stgln\ieee-stgln\figure\v_r_bike.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def variant_mae_taxi():
    '''
    taxi variant bar for rmse
    :return:
    '''
    n = 3
    X = np.arange(n)
    Y = [15.16,14.95,14.57]
    plt.figure(1, figsize=(5, 5))
    ax = plt.gca()

    # x y major label and minor label
    # xmajorLocator = plt.MultipleLocator(1)  # set x major label as 1 times
    # xminorLocator = plt.MultipleLocator(0.5)  # set x minor label as 0.5 times
    # ax.xaxis.set_major_locator(xmajorLocator) # set x major locator
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ymajorLocator = plt.MultipleLocator(0.2)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.2)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="both", linestyle='--', which='minor')
    plt.xlim((-1, n))
    plt.xticks(())
    plt.ylim(14.2,15.4)
    plt.yticks(np.linspace(14.2, 15.4, 7), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#2E75B6', edgecolor='k', hatch="--", label="GL-TCN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#4169E1', edgecolor='k', hatch="///", label="GL-TCN-nog", alpha=0.8)
    plt.bar(X[2], Y[2], color='#6495ED', edgecolor='k', hatch="\\\\\\", label="GL-TCN", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('Method', fontsize=24)
    plt.ylabel('MAE', fontsize=24)

    plt.legend(loc='best', fontsize=13)
    plt.savefig(r'D:\研究生\科研\方案和写作\写作：stgln\ieee-stgln\figure\v_m_taxi.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def variant_mae_bike():
    '''
    taxi variant bar for rmse
    :return:
    '''
    n = 3
    X = np.arange(n)
    Y = [4.55,4.43,4.24]

    plt.figure(1, figsize=(5, 5))
    ax = plt.gca()

    # x y major label and minor label
    # xmajorLocator = plt.MultipleLocator(1)  # set x major label as 1 times
    # xminorLocator = plt.MultipleLocator(0.5)  # set x minor label as 0.5 times
    # ax.xaxis.set_major_locator(xmajorLocator) # set x major locator
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ymajorLocator = plt.MultipleLocator(0.1)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.1)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="both", linestyle='--', which='minor')
    plt.xlim((-1, n))
    plt.xticks(())
    plt.ylim(4.1,4.7)
    plt.yticks(np.linspace(4.1, 4.7, 7), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#228B22', edgecolor='k', hatch="--", label="GL-TCN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#2E8B57', edgecolor='k', hatch="///", label="GL-TCN-nog", alpha=0.8)
    plt.bar(X[2], Y[2], color='#3CB371', edgecolor='k', hatch="\\\\\\", label="GL-TCN", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('Method', fontsize=24)
    plt.ylabel('MAE', fontsize=24)

    plt.legend(loc='best', fontsize=13)
    plt.savefig(r'D:\研究生\科研\方案和写作\写作：stgln\ieee-stgln\figure\v_m_bike.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def len_bike():
    x = np.linspace(1, 7, 7)
    global_y = [5.82,5.32,5.14,5.21,5.27,5.32,5.45]
    local_y = [5.86,5.67,5.4,5.27,5.14,5.34,5.45]

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ymajorLocator = plt.MultipleLocator(0.2)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.2)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.grid(axis="both", linestyle='--', which='minor')

    plt.plot(x, global_y, color='#FF8C00', linewidth=2.0, linestyle='-',  marker='v', label='global data')
    plt.plot(x, local_y, color='#4169E1', linewidth=2.0, linestyle='-', marker='o', label='local data')

    plt.xlim((0.5, 7.5))
    plt.ylim((4.8, 6))
    plt.xticks(np.linspace(1, 7, 7), fontsize=18)
    plt.yticks(np.linspace(4.8, 6, 7), fontsize=18)

    plt.xlabel('The length of data', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    plt.legend(loc='lower right', fontsize=13)
    plt.savefig('E:\\研究生\\科研\方案\\stgln\\ieee-stgln\\figure\\l_bike.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def hyper_cnn():
    x = np.linspace(0,6,6)
    y = [17.48,17.32,16.69,16.78,17.24,17.42]

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ymajorLocator = plt.MultipleLocator(0.2)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.2)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.grid(axis="both", linestyle='--', which='minor')

    plt.plot(x, y, color='#000000', linewidth=2.0, linestyle='-',  marker='o')

    plt.xlim((-0.5, 6.5))
    plt.ylim((16.6, 17.6))
    plt.xticks(np.linspace(0, 6, 6), [2, 4, 6, 8, 10, 12], fontsize=18)
    plt.yticks(np.linspace(16.6, 17.6, 6), fontsize=18)

    plt.xlabel('The number of convolution layers', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    # plt.savefig('E:\\研究生\\科研\方案\\stgln\\ieee-stgln\\figure\\h_cnn.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_netl_mmd():
    '''
    画图 MMD bandwidth(0.5 1 2) sample_num 32 
    beijing guangzhou: 6.4477948975956 0.049495148465293945 | 0.4489 0.0393
    beijing tianjing: 9.485573235554057 0.04427947366169667 | 0.5444 0.0372
    guangzhou tianjing: 11.171122864466383 0.047595047771296875 | 0.5908 0.0386
    '''
    n = 3
    X = np.arange(n)
    # Y = [6.4477948975956,9.485573235554057,11.171122864466383]
    Y = [0.4489, 0.5444, 0.5908]

    ax = plt.gca()

    # x y major label and minor label
    # xmajorLocator = plt.MultipleLocator(1)  # set x major label as 1 times
    # xminorLocator = plt.MultipleLocator(0.5)  # set x minor label as 0.5 times
    # ax.xaxis.set_major_locator(xmajorLocator) # set x major locator
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.xaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    # ymajorLocator = plt.MultipleLocator(0.5)  # set y major label as 0.5 times
    # yminorLocator = plt.MultipleLocator(0.25)  # set y minor label as 0.25 times
    # ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    # ax.set_axisbelow(True) # grid below the charts

    # plt.grid(axis="both", linestyle='--', which='minor')
    plt.xlim((-1, n))
    plt.xticks([0, 1, 2], ['BJ->GZ', 'BJ->TJ', 'GZ->TJ'], fontsize=18)
    plt.ylim(0,0.62)
    plt.yticks(np.linspace(0, 0.60, 7), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '* ', 'o', 'O', '.')
    plt.bar(X[0], Y[0], width=0.8, color='#FFA500', edgecolor='w', label="BJ->GZ", alpha=0.8)#, hatch="--",)
    plt.bar(X[1], Y[1], width=0.8, color='#32CD32', edgecolor='w', label="BJ->TJ", alpha=0.8)#, hatch="///")
    plt.bar(X[2], Y[2], width=0.8, color='#6495ED', edgecolor='w', label="GZ->TJ", alpha=0.8)#, hatch="\\\\\\")

    plt.plot([-0.38, 0.38], [0.0393,0.0393], color='black', linewidth=2.0, linestyle='-')
    plt.plot([0.62, 1.38], [0.0372,0.0372], color='black', linewidth=2.0, linestyle='-')
    plt.plot([1.62, 2.38], [0.0386,0.0386], color='black', linewidth=2.0, linestyle='-')

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('City pair', fontsize=18)
    plt.ylabel('$MMD$', fontsize=18)

    #plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('E:\\研究生\\方案和写作\\写作：acdan\\figure\\mmd_city_data.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def variant_netl_rmse_3():
    '''
    acdan variant bar for rmse, only for three models
    :return:
    '''
    n = 13
    X = [0, 1, 2, 5, 6, 7, 10, 11, 12]
    Y = [0.2792, 0.2776, 0.274, 0.2856, 0.2825, 0.279, 0.2915, 0.2727, 0.2708]

    plt.figure(1, figsize=(9, 5))
    ax = plt.gca()

    ymajorLocator = plt.MultipleLocator(0.01)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.005)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="y", linestyle='--', which='major')
    plt.xlim((-1, n))
    plt.xticks([1, 6, 11], ['BJ->GZ', 'BJ->TJ', 'GZ->TJ'], fontsize=18)
    plt.ylim(0.26,0.29)
    plt.yticks(np.linspace(0.26, 0.30, 9), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#FFA500', edgecolor='k', hatch="--", label="ACDAN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#32CD32', edgecolor='k', hatch="///", label="ACDAN-sa", alpha=0.8)
    plt.bar(X[2], Y[2], color='#6495ED', edgecolor='k', hatch="\\\\\\", label="ACDAN", alpha=0.8)

    plt.bar(X[3], Y[3], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[4], Y[4], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[5], Y[5], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)

    plt.bar(X[6], Y[6], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[7], Y[7], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[8], Y[8], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('City pair', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    plt.legend(loc='best', fontsize=16)
    plt.savefig('D:\\研究生\\科研\\方案和写作\写作：acdan\\figure\\v_rmse_3.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def variant_netl_rmse_4():
    '''
    acdan variant bar for rmse, four models(add ACDAN-wd)
    :return:
    '''
    n = 16
    X = [0, 1, 2, 3,
         6, 7, 8, 9,
         12, 13, 14, 15]
    Y = [0.2792, 0.2776, 0.274, 0.2792,
         0.2856, 0.2825, 0.279, 0.286,
         0.2915, 0.2727, 0.2708, 0.2909]

    plt.figure(1, figsize=(9, 5))
    ax = plt.gca()

    ymajorLocator = plt.MultipleLocator(0.01)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.005)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="y", linestyle='--', which='major')
    plt.xlim((-1, n))
    plt.xticks([1.5, 7.5, 13.5], ['BJ->GZ', 'BJ->TJ', 'GZ->TJ'], fontsize=18)
    plt.ylim(0.26,0.29)
    plt.yticks(np.linspace(0.26, 0.30, 9), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#FFA500', edgecolor='k', hatch="--", label="ACDAN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#32CD32', edgecolor='k', hatch="///", label="ACDAN-sa", alpha=0.8)
    plt.bar(X[2], Y[2], color='#6495ED', edgecolor='k', hatch="\\\\\\", label="ACDAN", alpha=0.8)
    plt.bar(X[3], Y[3], color='#696969', edgecolor='k', hatch="...", label="ACDAN-nod", alpha=0.8)

    plt.bar(X[4], Y[4], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[5], Y[5], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[6], Y[6], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)
    plt.bar(X[7], Y[7], color='#696969', edgecolor='k', hatch="...", alpha=0.8)

    plt.bar(X[8], Y[8], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[9], Y[9], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[10], Y[10], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)
    plt.bar(X[11], Y[11], color='#696969', edgecolor='k', hatch="...", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('City pair', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)

    plt.legend(loc='best', fontsize=16)
    plt.savefig('D:\\研究生\\科研\\方案和写作\写作：acdan\\figure\\v_rmse_4.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def variant_netl_ndcg_3():
    '''
    acdan variant bar for ndcg@10, with three model
    :return:
    '''
    n = 13
    X = [0, 1, 2, 5, 6, 7, 10, 11, 12]
    Y = [0.8553, 0.8701, 0.89, 0.9306, 0.9306, 0.9337, 0.9149, 0.9216, 0.9266]

    plt.figure(1, figsize=(9, 5))
    ax = plt.gca()

    ymajorLocator = plt.MultipleLocator(0.01)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.01)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="y", linestyle='--', which='major')
    plt.xlim((-1, n))
    plt.xticks([1, 6, 11], ['BJ->GZ', 'BJ->TJ', 'GZ->TJ'], fontsize=18)
    plt.ylim(0.84,0.94)
    plt.yticks(np.linspace(0.84, 0.94, 11), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#FFA500', edgecolor='k', hatch="--", label="ACDAN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#32CD32', edgecolor='k', hatch="///", label="ACDAN-sa", alpha=0.8)
    plt.bar(X[2], Y[2], color='#6495ED', edgecolor='k', hatch="\\\\\\", label="ACDAN", alpha=0.8)

    plt.bar(X[3], Y[3], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[4], Y[4], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[5], Y[5], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)

    plt.bar(X[6], Y[6], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[7], Y[7], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[8], Y[8], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('City pair', fontsize=24)
    plt.ylabel('NDCG@10', fontsize=24)

    plt.legend(loc='best', fontsize=16)
    plt.savefig('D:\\研究生\\科研\\方案和写作\写作：acdan\\figure\\v_ndcg_3.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def variant_netl_ndcg_4():
    '''
    acdan variant bar for ndcg@10, with four models(ACDAN-wd)
    :return:
    '''
    n = 16
    X = [0, 1, 2, 3,
         6, 7, 8, 9,
         12, 13, 14, 15]
    Y = [0.8553, 0.8701, 0.89, 0.8522,
         0.9306, 0.9306, 0.9337, 0.9216,
         0.9149, 0.9216, 0.9266, 0.9052]

    plt.figure(1, figsize=(9, 5))
    ax = plt.gca()

    ymajorLocator = plt.MultipleLocator(0.01)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.01)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%5.1f'))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter('%5.1f'))

    ax.set_axisbelow(True) # grid below the charts

    plt.grid(axis="y", linestyle='--', which='major')
    plt.xlim((-1, n))
    plt.xticks([1.5, 7.5, 13.5], ['BJ->GZ', 'BJ->TJ', 'GZ->TJ'], fontsize=18)
    plt.ylim(0.84,0.94)
    plt.yticks(np.linspace(0.84, 0.94, 11), fontsize=18)

    # hatch pattern ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')
    plt.bar(X[0], Y[0], color='#FFA500', edgecolor='k', hatch="--", label="ACDAN-min", alpha=0.8)
    plt.bar(X[1], Y[1], color='#32CD32', edgecolor='k', hatch="///", label="ACDAN-sa", alpha=0.8)
    plt.bar(X[2], Y[2], color='#6495ED', edgecolor='k', hatch="\\\\\\", label="ACDAN", alpha=0.8)
    plt.bar(X[3], Y[3], color='#696969', edgecolor='k', hatch="...", label="ACDAN-nod", alpha=0.8)

    plt.bar(X[4], Y[4], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[5], Y[5], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[6], Y[6], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)
    plt.bar(X[7], Y[7], color='#696969', edgecolor='k', hatch="...", alpha=0.8)

    plt.bar(X[8], Y[8], color='#FFA500', edgecolor='k', hatch="--", alpha=0.8)
    plt.bar(X[9], Y[9], color='#32CD32', edgecolor='k', hatch="///", alpha=0.8)
    plt.bar(X[10], Y[10], color='#6495ED', edgecolor='k', hatch="\\\\\\", alpha=0.8)
    plt.bar(X[11], Y[11], color='#696969', edgecolor='k', hatch="...", alpha=0.8)

    # label_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 18,}
    # plt.xlabel('Method', label_font)
    plt.xlabel('City pair', fontsize=24)
    plt.ylabel('NDCG@10', fontsize=24)

    plt.legend(loc='best', fontsize=16)
    plt.savefig('D:\\研究生\\科研\\方案和写作\写作：acdan\\figure\\v_ndcg_4.jpg', dpi=300, bbox_inches='tight')
    plt.show()

def alpha_netl_ndcg():
    x = [0, 0.3, 0.5, 0.8, 1.0]
    bj_gz = [0.8512, 0.8553, 0.8669, 0.89, 0.8615]
    bj_tj = [0.9216, 0.9226, 0.9306, 0.9337, 0.9216]
    gz_tj = [0.9052, 0.9146, 0.9266, 0.9149, 0.9149]

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ymajorLocator = plt.MultipleLocator(0.04)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.04)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.grid(axis="y", linestyle='--', which='major')

    plt.plot(x, bj_gz, color='#FF8C00', linewidth=2.0, linestyle='-',  marker='^', label='BJ->GZ')
    plt.plot(x, bj_tj, color='#4169E1', linewidth=2.0, linestyle='-', marker='o', label='BJ->TJ')
    plt.plot(x, gz_tj, color='#DC143C', linewidth=2.0, linestyle='-', marker='*', label='GZ->TJ')

    plt.xlim((-0.1, 1.1))
    plt.ylim((0.8, 1.0))
    plt.xticks(np.linspace(0, 1, 6), fontsize=18)
    plt.yticks(np.linspace(0.8, 1, 6), fontsize=18)

    plt.xlabel(r'$\alpha$', fontsize=18)
    plt.ylabel('NDCG@10', fontsize=18)

    plt.legend(loc='upper right', fontsize=12)
    plt.savefig('E:\\研究生\\方案和写作\\写作：acdan\\figure\\alpha_ndcg.png', dpi=300, bbox_inches='tight')
    plt.show()

def alpha_netl_rmse():
    x = [0, 0.3, 0.5, 0.8, 1.0]
    bj_gz = [0.2779, 0.274, 0.2758, 0.2767, 0.2909]
    bj_tj = [0.2879, 0.279, 0.2842, 0.2858, 0.2941]
    gz_tj = [0.2817, 0.2779, 0.2769, 0.2693, 0.29]

    plt.figure(figsize=(6,4))
    ax = plt.gca()
    ymajorLocator = plt.MultipleLocator(0.05)  # set y major label as 0.5 times
    yminorLocator = plt.MultipleLocator(0.05)  # set y minor label as 0.25 times
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)

    plt.grid(axis="y", linestyle='--', which='major')

    plt.plot(x, bj_gz, color='#FF8C00', linewidth=2.0, linestyle='-',  marker='^', label='BJ->GZ')
    plt.plot(x, bj_tj, color='#4169E1', linewidth=2.0, linestyle='-', marker='o', label='BJ->TJ')
    plt.plot(x, gz_tj, color='#DC143C', linewidth=2.0, linestyle='-', marker='*', label='GZ->TJ')

    plt.xlim((-0.1, 1.1))
    plt.ylim((0.25, 0.3))
    plt.xticks(np.linspace(0, 1, 6), fontsize=14)
    plt.yticks(np.linspace(0.25, 0.3, 6), fontsize=14)

    plt.xlabel(r'$\alpha$', fontsize=18)
    plt.ylabel('RMSE', fontsize=18)

    plt.legend(loc='lower right', fontsize=12)
    plt.savefig('E:\\研究生\\方案和写作\\写作：acdan\\figure\\alpha_rmse.png', dpi=300, bbox_inches='tight')
    plt.show()


def tsne_example():
    from time import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import offsetbox
    from sklearn import (manifold, datasets, decomposition, ensemble,
                         discriminant_analysis, random_projection)
    digits = datasets.load_digits(n_class=10)  # 导入数据集
    X = digits.data
    y = digits.target
    n_samples, n_features = X.shape
    n_neighbors = 30



    color_list=['purple','red','blue','navy','orange','gray',
                'black','plum','gold','darkgreen','seagreen']

    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        plt.figure()
        ax = plt.subplot(111)

        #去掉边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                     #color=plt.cm.Set3(y[i]/10),
                     color=color_list[y[i]],
                     fontdict={'weight': 'bold', 'size': 11})
        if hasattr(offsetbox, 'AnnotationBbox'):  # (用于判断对象是否包含对应的属性)
            shown_images = np.array([[1., 1.]])
        # for i in range(digits.data.shape[0]):
        #     dist = np.sum((X[i] - shown_images) ** 2, 1)
        #     if np.min(dist) < 4e-3:
        #         continue
        #     shown_images = np.r_[shown_images, [X[i]]]  # 按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
        #     imagebox = offsetbox.AnnotationBbox(
        #         offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
        #         X[i])
        # ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

        plt.savefig('E:\\研究生\\毕业论文\\毕业论文\\图\\tsne_example.jpg', dpi=300, bbox_inches='tight')
        plt.show()

    # ----------------------------------------------------------------------

    n_img_per_row = 15
    img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 10 * i + 1
        for j in range(n_img_per_row):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('E:\\研究生\\毕业论文\\毕业论文\\图\\tsne_digit.jpg', dpi=300, bbox_inches='tight')
    # plt.title('A selection from the 64-dimensional digits dataset')
    # 计算PCA
    # print("Computing PCA projection")
    # t0 = time()
    # X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    # plot_embedding(X_pca,
    #               "Principal Components projection of the digits (time %.2fs)" %
    #               (time() - t0))
    # 计算t-SNE
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne)#,"t-SNE embedding of the digits (time %.2fs)" % (time() - t0))

    pass

def relu():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    x = np.linspace(-15, 15, 1500)
    relu = np.zeros(1500)
    for i in range(750, 1500):
        relu[i] = x[i]

    plt.figure()
    ax = plt.subplot(111) #保留两位小数
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #plt.plot(x, sigmod, '--')
    plt.plot(x, relu, '-', c='b')
    plt.legend(["ReLU"],fontsize=14)
    plt.grid()
    plt.savefig('E:\\研究生\\毕业论文\\毕业论文\\图\\relu.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def sigmoid():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    x = np.linspace(-15, 15, 1500)
    sigmoid = 1 / (1 + np.exp(-x))

    plt.figure()
    ax = plt.subplot(111)  # 保留两位小数
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.plot(x, sigmoid, '-', c='b')
    plt.legend(["Sigmoid"],fontsize=14)
    plt.grid()
    plt.savefig('E:\\研究生\\毕业论文\\毕业论文\\图\\sigmoid.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def tanh():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    x = np.linspace(-15, 15, 1500)
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    plt.figure()
    ax = plt.subplot(111)  # 保留两位小数
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.plot(x, tanh, '-', c='b')
    plt.legend(["Tanh"],fontsize=14)
    plt.grid()
    plt.savefig('E:\\研究生\\毕业论文\\毕业论文\\图\\tanh.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def leakrelu():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    x = np.linspace(-15, 15, 1500)
    leakrelu = np.zeros(1500)
    for i in range(750, 1500):
        leakrelu[i] = x[i]
    for i in range(0,750):
        leakrelu[i] = 0.1 * x[i]

    plt.figure()
    ax = plt.subplot(111)  # 保留两位小数
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #plt.plot(x, sigmod, '--')
    plt.plot(x, leakrelu, '-', c='b')
    plt.legend(["Leak ReLU"],fontsize=14)
    plt.grid()
    plt.savefig('E:\\研究生\\毕业论文\\毕业论文\\图\\leakrelu.jpg', dpi=300, bbox_inches='tight')
    plt.show()
    pass

if __name__ == '__main__':
    #relu()
    sigmoid()
    #tanh()
    #leakrelu()
