from machine import Pin, ADC
import dht
import time
import json
import uselect
import sys

# ----------------------
# SENSOR SETUP
# ----------------------
dht_sensor = dht.DHT11(Pin(15))

mq_sensor = ADC(26)  # MQ-2 or MQ-135 analog output

# ----------------------
# MOTOR SETUP (L298N)
# ----------------------
IN1 = Pin(10, Pin.OUT)
IN2 = Pin(11, Pin.OUT)
IN3 = Pin(12, Pin.OUT)
IN4 = Pin(13, Pin.OUT)

is_moving = False
last_command_time = time.ticks_ms()
WATCHDOG_TIMEOUT_MS = 2000

def forward():
    global is_moving, last_command_time
    IN1.high()
    IN2.low()
    IN3.high()
    IN4.low()
    is_moving = True
    last_command_time = time.ticks_ms()

def stop():
    global is_moving
    IN1.low()
    IN2.low()
    IN3.low()
    IN4.low()
    is_moving = False

# ----------------------
# CALIBRATION VALUES
# (YOU MUST ADJUST IN REAL ENVIRONMENT)
# ----------------------
CLEAN_AIR = 3000
POLLUTED_AIR = 8000

# ----------------------
# FUNCTION: ETHYLENE INDEX (PROXY)
# ----------------------
def get_ethylene_index(raw_value):
    index = (raw_value - CLEAN_AIR) * 100 / (POLLUTED_AIR - CLEAN_AIR)

    if index < 0:
        index = 0
    elif index > 100:
        index = 100

    return round(index, 2)

# ----------------------
# SERIAL CONTROL SETUP
# ----------------------
poll_obj = uselect.poll()
poll_obj.register(sys.stdin, uselect.POLLIN)

def check_serial():
    global last_command_time
    if poll_obj.poll(0):
        line = sys.stdin.readline().strip()
        if not line:
            return
        
        last_command_time = time.ticks_ms()
        if line == "forward":
            forward()
        elif line == "stop":
            stop()

# ----------------------
# CONTINUOUS MONITORING & CONTROL
# ----------------------
last_sensor_time = time.ticks_ms()
sensor_interval = 2000  # 2 seconds

while True:
    # 1. Poll serial input
    check_serial()

    # 2. Watchdog check
    if is_moving:
        if time.ticks_diff(time.ticks_ms(), last_command_time) > WATCHDOG_TIMEOUT_MS:
            stop()

    # 3. Read sensors periodically
    current_time = time.ticks_ms()
    if time.ticks_diff(current_time, last_sensor_time) >= sensor_interval:
        last_sensor_time = current_time

        # ---- DHT11 ----
        try:
            dht_sensor.measure()
            temp = dht_sensor.temperature()
            hum = dht_sensor.humidity()
        except:
            temp = None
            hum = None

        # ---- GAS SENSOR ----
        gas_raw = mq_sensor.read_u16()
        ethylene_index = get_ethylene_index(gas_raw)

        # ---- FINAL DATA ----
        data = {
            "temperature": temp,
            "humidity": hum,
            "gas_raw": gas_raw,
            "ethylene_index": ethylene_index
        }

        print(json.dumps(data))

    # Small delay to prevent 100% CPU lockup
    time.sleep(0.05)
