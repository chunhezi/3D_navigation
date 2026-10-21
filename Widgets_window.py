from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(677, 712)
        self.button_open_serial = QtWidgets.QPushButton(Form)
        self.button_open_serial.setGeometry(QtCore.QRect(20, 100, 131, 21))
        self.button_open_serial.setObjectName("button_open_serial")
        self.button_exit = QtWidgets.QPushButton(Form)
        self.button_exit.setGeometry(QtCore.QRect(20, 190, 131, 21))
        self.button_exit.setObjectName("button_exit")
        self.label_baud = QtWidgets.QLabel(Form)
        self.label_baud.setGeometry(QtCore.QRect(20, 40, 54, 21))
        self.label_baud.setObjectName("label_baud")
        self.comboBox_baud = QtWidgets.QComboBox(Form)
        self.comboBox_baud.setGeometry(QtCore.QRect(80, 40, 69, 22))
        self.comboBox_baud.setObjectName("comboBox_baud")
        self.button_refresh_serial = QtWidgets.QPushButton(Form)
        self.button_refresh_serial.setGeometry(QtCore.QRect(20, 130, 131, 21))
        self.button_refresh_serial.setObjectName("button_refresh_serial")
        self.button_camera = QtWidgets.QPushButton(Form)
        self.button_camera.setGeometry(QtCore.QRect(20, 70, 131, 21))
        self.button_camera.setObjectName("button_camera")
        self.label_port = QtWidgets.QLabel(Form)
        self.label_port.setGeometry(QtCore.QRect(20, 10, 51, 21))
        self.label_port.setObjectName("label_port")
        self.comboBox_port = QtWidgets.QComboBox(Form)
        self.comboBox_port.setGeometry(QtCore.QRect(80, 10, 69, 22))
        self.comboBox_port.setObjectName("comboBox_port")

        self.label_show_camera = QtWidgets.QLabel(Form)
        self.label_show_camera.setGeometry(QtCore.QRect(20, 220, 640, 480))
        self.label_show_camera.setText("")
        self.label_show_camera.setObjectName("label_show_camera")
        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(179, 9, 481, 171))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 479, 169))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")

        # ===== Arduino 泵串口选择区（放在 scrollArea 顶部工具栏）=====
        self.label_arduino_port = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_arduino_port.setGeometry(QtCore.QRect(8, 6, 80, 18))
        self.label_arduino_port.setObjectName("label_arduino_port")

        self.comboBox_arduino_port = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBox_arduino_port.setGeometry(QtCore.QRect(90, 4, 90, 22))
        self.comboBox_arduino_port.setObjectName("comboBox_arduino_port")

        self.button_open_arduino = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.button_open_arduino.setGeometry(QtCore.QRect(190, 4, 90, 22))
        self.button_open_arduino.setObjectName("button_open_arduino")

        # ===== 测试泵按钮 =====
        self.button_test_pump = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.button_test_pump.setGeometry(QtCore.QRect(286, 4, 90, 22))
        self.button_test_pump.setObjectName("button_test_pump")

        # ===== 原来的日志框 textEdit 下移，避免被工具栏遮挡 =====
        self.textEdit = QtWidgets.QTextEdit(self.scrollAreaWidgetContents)
        self.textEdit.setGeometry(QtCore.QRect(0, 30, 481, 141))
        self.textEdit.setObjectName("textEdit")

        # ====== ====== ====== ====== ====== ======

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.button_up = QtWidgets.QPushButton(Form)
        self.button_up.setGeometry(QtCore.QRect(180, 190, 81, 21))
        self.button_up.setObjectName("button_up")
        self.button_motion = QtWidgets.QPushButton(Form)
        self.button_motion.setGeometry(QtCore.QRect(20, 160, 131, 21))
        self.button_motion.setObjectName("button_motion")
        self.button_down = QtWidgets.QPushButton(Form)
        self.button_down.setGeometry(QtCore.QRect(280, 190, 81, 21))
        self.button_down.setObjectName("button_down")
        self.button_left = QtWidgets.QPushButton(Form)
        self.button_left.setGeometry(QtCore.QRect(380, 190, 81, 21))
        self.button_left.setObjectName("button_left")
        self.button_right = QtWidgets.QPushButton(Form)
        self.button_right.setGeometry(QtCore.QRect(480, 190, 81, 21))
        self.button_right.setObjectName("button_right")
        self.button_stop = QtWidgets.QPushButton(Form)
        self.button_stop.setGeometry(QtCore.QRect(580, 190, 81, 21))
        self.button_stop.setObjectName("button_stop")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Magnetic Drive Desktop"))
        self.button_open_serial.setText(_translate("Form", "打开串口"))
        self.button_exit.setText(_translate("Form", "Exit"))
        self.label_baud.setText(_translate("Form", "波特率"))
        self.button_refresh_serial.setText(_translate("Form", "刷新串口"))
        self.button_camera.setText(_translate("Form", "打开相机"))
        self.label_port.setText(_translate("Form", "串口端口"))

        self.label_arduino_port.setText(_translate("Form", "Arduino端口"))
        self.button_open_arduino.setText(_translate("Form", "打开泵串口"))

        self.button_up.setText(_translate("Form", "up"))
        self.button_motion.setText(_translate("Form", "start motion"))
        self.button_down.setText(_translate("Form", "down"))
        self.button_left.setText(_translate("Form", "left"))
        self.button_right.setText(_translate("Form", "right"))
        self.button_stop.setText(_translate("Form", "stop"))
