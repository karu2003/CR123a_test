import machine
import utime

sensor_temp = machine.ADC(2)
conversion_factor = 3.3 / (65535)

while True:
    reading = sensor_temp.read_u16() * conversion_factor

    print(reading)
    utime.sleep(0.25)  