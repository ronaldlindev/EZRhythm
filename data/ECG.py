import serial
import time
from datetime import datetime
ser = serial.Serial('/dev/ttyACM0', 9600 , timeout = 5)
name = datetime.now()
file = open(f'{name}.txt','a')


while True:
    input_str = ser.readline().decode('utf-8').strip()
    file.write(input_str + ',')
    print(input_str)
    time.sleep(0.008)
