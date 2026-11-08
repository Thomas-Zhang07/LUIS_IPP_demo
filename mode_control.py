import serial

ser = serial.Serial('COM9', 38400)
# C8: 200 Normal mode
# C9: 201 C4
# CA: 202 Liitle star
# CB: 203 200hz
data = bytes(
    [0xC9] +
    [0xFE] +
    [0x10] * 256 +
    [0xFD]
)

ser.write(data)
ser.flush()
ser.close()

print("done")