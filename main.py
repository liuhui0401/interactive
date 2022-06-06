import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel, QPushButton, QLineEdit, QInputDialog, QMessageBox
from PyQt5.QtGui import QPainter, QColor, QBrush, QPalette, QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt
from PIL import Image
import sys
sys.path.append('bin/Real_time')
import bin.Real_time.test as test
content_pic = ['', '']
style_pic = ['', '']
 
class FirstUi(QWidget):
    def __init__(self):
        super(FirstUi, self).__init__()
        self.init_ui()
 
    def init_ui(self):
        # 设置标题
        self.setWindowTitle('Style Transfer')
        # 画布大小为1920*1200
        self.resize(1920, 1200)
        # 设置背景
        background = QPalette()
        #设置字体颜色
    
        self.setStyleSheet("QLabel{;}"
                   "QLabel{color:rgb(242,192,86,255)}"
                   "QLabel:hover{;}")

        
        # 设置背景图
        background.setBrush(self.backgroundRole(), QBrush(
            QPixmap('./bin/background1.png').scaled(self.width(), self.height())))
        self.setPalette(background)
        # 设置icon
        self.setWindowIcon(QIcon('./bin/icon.png'))
        # 以下部分为该交互界面中的文本
        label = QLabel('欢迎使用本项目，本项目可将给定图片转换成四种指定风格之一，\n同时用户也可以自定义风格图片，请选择：', self)
        label.setFont(QFont("宋体", 20, QFont.Bold))
        label.move(220, 150)

        self.btn1 = QPushButton('我要将图片转换成指定风格', self)
        self.btn1.setFont(QFont("宋体", 20, QFont.Bold))
        self.btn1.setStyleSheet('background-color:rgb(0,0,0,100);\ncolor:rgb(242,192,86,255)') 
        self.btn1.setGeometry(100, 500, 1800, 200)
        self.btn1.clicked.connect(self.slot_btn_function1)

        self.btn2 = QPushButton('我要自定义图片风格', self)
        self.btn2.setFont(QFont("宋体", 20, QFont.Bold))
        self.btn2.setStyleSheet('background-color:rgb(0,0,0,100);\ncolor:rgb(242,192,86,255)')
        self.btn2.setGeometry(100, 800, 1800, 200)
        self.btn2.clicked.connect(self.slot_btn_function2)
 
    def slot_btn_function1(self):
        self.hide()
        self.s = SecondUi()
        self.s.show()
    def slot_btn_function2(self):
        self.hide()
        self.s = ThirdUi()
        self.s.show()
 
class SecondUi(QWidget):
    def __init__(self):
        super(SecondUi, self).__init__()
        self.init_ui()
 
    def init_ui(self):
        self.setAcceptDrops(True)
        # 设置标题
        self.setWindowTitle('Style Transfer')
        # 画布大小为1920*1200
        self.resize(1920, 1200)
        # 设置背景
        background = QPalette()
        #设置字体颜色
        self.setStyleSheet("QLabel{;}"
                   "QLabel{color:rgb(242,192,86,255)}"
                   "QLabel:hover{;}")
        # 设置背景图
        background.setBrush(self.backgroundRole(), QBrush(
            QPixmap('./bin/background2.jpg').scaled(self.width(), self.height())))
        self.setPalette(background)
        # 设置icon
        self.setWindowIcon(QIcon('./bin/icon.png'))
        # 以下部分为该交互界面中的文本
        label1 = QLabel('将要转换的内容图片拖入下面矩形框中', self)
        label1.setFont(QFont("宋体", 12, QFont.Bold))

        label1.move(220, 150)

        label3 = QLabel('确定', self)
        label3.setFont(QFont("宋体", 20, QFont.Bold))

        label3.move(910, 1080)
        label4 = QLabel('内容图像', self)
        label4.setFont(QFont("宋体", 40, QFont.Bold))

        label4.move(280, 500)

        label6 = QLabel('brush', self)
        label6.setFont(QFont("宋体", 40, QFont.Bold))

        label6.move(1200, 200)
        label7 = QLabel('mosaic', self)
        label7.setFont(QFont("宋体", 40, QFont.Bold))

        label7.move(1200, 400)
        label8 = QLabel('starry', self)
        label8.setFont(QFont("宋体", 40, QFont.Bold))

        label8.move(1200, 600)
        label9 = QLabel('stair', self)
        label9.setFont(QFont("宋体", 40, QFont.Bold))

        label9.move(1200, 800)
        label2 = QLabel('请选择你想转换的风格', self)
        label2.setFont(QFont("宋体", 12, QFont.Bold))

        label2.move(1175, 150)

        # 风格选择，0为空由上到下对应1，2，3，4
        self.style_choose = 0
        self.btn = QPushButton('返回', self)
        self.btn.setGeometry(1250, 1050, 250, 100)
        self.btn.setFont(QFont("宋体", 20, QFont.Bold))
        self.btn.setStyleSheet('background-color:rgb(0,0,0,100);\ncolor:rgb(242,192,86,255)')
        self.btn.clicked.connect(self.slot_btn_function)
    #返回按钮
    def slot_btn_function(self):
        content_pic[0]=''
        content_pic[1]=''
        style_pic[0]=''
        style_pic[1]=''
        self.hide()
        self.f = FirstUi()
        self.f.show()
    # 图片拖拽
    def dragEnterEvent(self, evn):

        evn.accept()
    # 将图片拖进来后松开鼠标左键时鼠标的坐标以及是否读入内容或风格图片

    def dropEvent(self, evn):
        # print(evn.pos())
        if evn.pos().x() > 200 and evn.pos().x() < 800 and evn.pos().y() > 200 and evn.pos().y() < 1000:
            content_pic[0] = evn.mimeData().text()[:]

    def mousePressEvent(self, e):  # 重载一下鼠标点击事件
        # 左键按下
        if e.buttons() == Qt.LeftButton:
            # 坐标判断
            if e.pos().x() > 860 and e.pos().x() < 1060 and e.pos().y() > 1050 and e.pos().y() < 1150:

                # 确保内容图片和风格图片都不为空
                if content_pic[0] == '':
                    self.show_message1()

                elif self.style_choose == 0:
                    self.show_message2()
                elif self.style_choose==1:
                    content_pic[1] = content_pic[0][8:]
                    test.main(content_pic[1],'./bin/brush.jpg','./bin/Real_time/model_state.pth','./bin/target.jpg')
                    img=Image.open('./bin/target.jpg')
                    img.show()
                elif self.style_choose==2:
                    content_pic[1] = content_pic[0][8:]
                    test.main(content_pic[1],'./bin/mosaic.jpg','./bin/Real_time/model_state.pth','./bin/target.jpg')
                    img=Image.open('./bin/target.jpg')
                    img.show()
                elif self.style_choose==3:
                    content_pic[1] = content_pic[0][8:]
                    test.main(content_pic[1],'./bin/starry.jpg','./bin/Real_time/model_state.pth','./bin/target.jpg')
                    img=Image.open('./bin/target.jpg')
                    img.show()
                elif self.style_choose==4:
                    content_pic[1] = content_pic[0][8:]
                    test.main(content_pic[1],'./bin/trial.jpg','./bin/Real_time/model_state.pth','./bin/target.jpg')
                    img=Image.open('./bin/target.jpg')
                    img.show()
            elif e.pos().x() > 1120 and e.pos().x() < 1570 and e.pos().y() > 200 and e.pos().y() < 305:
                self.style_choose = 1
                self.update()
            elif e.pos().x() > 1120 and e.pos().x() < 1570 and e.pos().y() > 400 and e.pos().y() < 505:
                self.style_choose = 2
                self.update()
            elif e.pos().x() > 1120 and e.pos().x() < 1570 and e.pos().y() > 600 and e.pos().y() < 705:
                self.style_choose = 3
                self.update()
            elif e.pos().x() > 1120 and e.pos().x() < 1570 and e.pos().y() > 800 and e.pos().y() < 905:
                self.style_choose = 4
                self.update()
    # 绘图，使界面更加美观

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawRectangles(qp)
        if self.style_choose == 1:
            qp.setBrush(QColor(255, 0, 0))
            qp.drawRect(1120, 200, 450, 105)
        elif self.style_choose == 2:
            qp.setBrush(QColor(255, 0, 0))
            qp.drawRect(1120, 400, 450, 105)
        elif self.style_choose == 3:
            qp.setBrush(QColor(255, 0, 0))
            qp.drawRect(1120, 600, 450, 105)
        elif self.style_choose == 4:
            qp.setBrush(QColor(255, 0, 0))
            qp.drawRect(1120, 800, 450, 105)
        qp.end()

    # 画矩形
    def drawRectangles(self, qp):

        # 内容图像框
        # Qcolor前三个参数是RGB,第四个是透明度
        qp.setBrush(QColor(0, 0, 255, 100))
        qp.drawRect(200, 200, 600, 800)
        qp.setBrush(QColor(255, 0, 0, 150))
        qp.drawRect(200, 130, 600, 60)
        qp.setBrush(QColor(255, 0, 0, 150))
        qp.drawRect(1120, 130, 450, 60)
        # 风格图像框

        qp.setBrush(QColor(0, 0, 0, 100))
        qp.drawRect(1120, 200, 450, 105)
        qp.setBrush(QColor(0, 0, 0, 100))
        qp.drawRect(1120, 400, 450, 105)
        qp.setBrush(QColor(0, 0, 0, 100))
        qp.drawRect(1120, 600, 450, 105)
        qp.setBrush(QColor(0, 0, 0, 100))
        qp.drawRect(1120, 800, 450, 105)
        # 确定框
        qp.setBrush(QColor(0, 0, 0, 100))
        qp.drawRect(860, 1050, 200, 100)

    # 内容图片为空提示
    def show_message1(self):
        QMessageBox.information(self, "错误提示", "内容图片不能为空！",
                                QMessageBox.Ok)
    # 风格图片为空提示

    def show_message2(self):
        QMessageBox.information(self, "错误提示", "风格图片不能为空！",
                                QMessageBox.Ok)

class ThirdUi(QWidget):
    # 初始化
    def __init__(self):
        super(ThirdUi, self).__init__()
        self.init_ui()
    def init_ui(self):
        # 允许拖入图片
        self.setAcceptDrops(True)
        # 画布大小为1920*1200
        self.resize(1920, 1200)
        # 设置标题
        self.setWindowTitle('Style Transfer')
         # 设置背景
        background = QPalette()
        #设置字体颜色
        self.setStyleSheet("QLabel{;}"
                   "QLabel{color:rgb(242,192,86,255)}"
                   "QLabel:hover{;}")
        # 设置背景图
        background.setBrush(self.backgroundRole(), QBrush(
            QPixmap('./bin/background3.jpg').scaled(self.width(), self.height())))
        self.setPalette(background)
        # 设置icon
        self.setWindowIcon(QIcon('./bin/icon.png'))
        # 以下部分为该交互界面中的文本
        # 标签1内容
        label1 = QLabel('将要转换的内容图片拖入下面矩形框中', self)
        # 标签1字体
        label1.setFont(QFont("宋体", 12, QFont.Bold))
        # label1.setAutoFillBackground(True)
        # 标签1位置
        label1.move(220, 150)
        label2 = QLabel('将目标风格图片拖入下面矩形框中', self)
        label2.setFont(QFont("宋体", 12, QFont.Bold))
        # label2.setAutoFillBackground(True)
        label2.move(1175, 150)
        label3 = QLabel('确定', self)
        label3.setFont(QFont("宋体", 20, QFont.Bold))
        # label3.setAutoFillBackground(True)
        label3.move(910, 1080)
        label4 = QLabel('内容图像', self)
        label4.setFont(QFont("宋体", 40, QFont.Bold))
        # label4.setAutoFillBackground(True)
        label4.move(280, 500)

        label5 = QLabel('风格图像', self)
        label5.setFont(QFont("宋体", 40, QFont.Bold))
        # label5.setAutoFillBackground(True)
        label5.move(1200, 500)
        self.btn = QPushButton('返回', self)
        self.btn.setGeometry(1250, 1050, 250, 100)
        self.btn.setFont(QFont("宋体", 20, QFont.Bold))
        self.btn.setStyleSheet('background-color:rgb(0,0,0,100);\ncolor:rgb(242,192,86,255)')
        self.btn.clicked.connect(self.slot_btn_function)
        #self.QLabl = QLabel(self)
    #返回按钮
    def slot_btn_function(self):
        content_pic[0]=''
        content_pic[1]=''
        style_pic[0]=''
        style_pic[1]=''
        self.hide()
        self.f = FirstUi()
        self.f.show()
    # 图片拖拽
    def dragEnterEvent(self, evn):
        # self.QLabl.setText(evn.mimeData().text())
        # evn.mimeData().text()中存的是图片绝对路径
        # print(evn.mimeData().text())
        # print(evn.pos())
        evn.accept()
    # 将图片拖进来后松开鼠标左键时鼠标的坐标以及是否读入内容或风格图片

    def dropEvent(self, evn):
        # print(evn.pos())
        #拖拽到内容位置
        if evn.pos().x() > 200 and evn.pos().x() < 800 and evn.pos().y() > 200 and evn.pos().y() < 1000:
            content_pic[0] = evn.mimeData().text()[:]
            # print('content_pic:'+content_pic[0])
        #拖拽到风格位置
        elif evn.pos().x() > 1120 and evn.pos().x() < 1720 and evn.pos().y() > 200 and evn.pos().y() < 1000:
            style_pic[0] = evn.mimeData().text()[:]
            # print('style_pic:'+style_pic[0])

    def mousePressEvent(self, e):  # 重载一下鼠标点击事件
        # 左键按下
        if e.buttons() == Qt.LeftButton:
            # 坐标判断
            if e.pos().x() > 860 and e.pos().x() < 1060 and e.pos().y() > 1050 and e.pos().y() < 1150:
                # print(content_pic[0])
                # print(style_pic[0])
                # 确保内容图片和风格图片都不为空
                if content_pic[0] == '':
                    self.show_message1()

                elif style_pic[0] == '':
                    self.show_message2()
                else:
                    content_pic[1] = content_pic[0][8:]
                    style_pic[1] = style_pic[0][8:]
                    test.main(content_pic[1],style_pic[1],'./bin/Real_time/model_state.pth','./bin/target.jpg')
                    img=Image.open('./bin/target.jpg')
                    img.show()
    # 绘图，使界面更加美观
    def paintEvent(self, e):

        qp = QPainter()
        qp.begin(self)
        self.drawRectangles(qp)
        qp.end()

    # 画矩形
    def drawRectangles(self, qp):

        #col = QColor(0, 0, 0)
        # col.setNamedColor('#d4d4d4')
        # qp.setPen(col)
        # 内容图像框
        # Qcolor前三个参数是RGB,第四个是透明度
        qp.setBrush(QColor(0, 0, 255, 100))
        #矩形坐标
        qp.drawRect(200, 200, 600, 800)
        # 风格图像框
        qp.setBrush(QColor(255, 0, 0, 150))
        qp.drawRect(200, 130, 600, 60)
        qp.setBrush(QColor(255, 0, 0, 150))
        qp.drawRect(1120, 130, 600, 60)
        #qp.setBrush(QColor(255, 80, 0, 160))
        qp.setBrush(QColor(0, 0, 255, 100))
        qp.drawRect(1120, 200, 600, 800)
        # 确定框
        qp.setBrush(QColor(0, 0, 0, 100))
        qp.drawRect(860, 1050, 200, 100)
    # 内容图片为空提示

    def show_message1(self):
        QMessageBox.information(self, "错误提示", "内容图片不能为空！",
                                QMessageBox.Ok)
    # 风格图片为空提示

    def show_message2(self):
        QMessageBox.information(self, "错误提示", "风格图片不能为空！",
                                QMessageBox.Ok)
 
def main():
    app = QApplication(sys.argv)
    w = FirstUi()
    w.show()
    sys.exit(app.exec_())
 
 
if __name__ == '__main__':
    main()