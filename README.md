# PSK
# Aim
Write a simple Python program for the modulation and demodulation of PSK and QPSK.
# Tools required
# Program
```
Program for PSK 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
# Butterworth low-pass filter for demodulation
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)
# Parameters
fs = 1000                # Sampling frequency
f_carrier = 50           # Carrier frequency
bit_rate = 10            # Data rate (bits per second)
T = 1                    # Total time duration in seconds
t = np.linspace(0, T, int(fs * T), endpoint=False)
# Message signal (binary data)
bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
message_signal = np.repeat(bits, bit_duration)
# PSK Modulation (0 -> 0 phase, 1 -> 180° phase shift)
carrier = np.sin(2 * np.pi * f_carrier * t)
psk_signal = np.sin(2 * np.pi * f_carrier * t + np.pi * message_signal)
# PSK Demodulation
demodulated = psk_signal * carrier
filtered_signal = butter_lowpass_filter(demodulated, f_carrier, fs)
decoded_bits = (filtered_signal[::bit_duration] < 0).astype(int)
# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Message Signal (Binary)', color='b')
plt.title('Message Signal')
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(t, carrier, label='Carrier Signal', color='g')
plt.title('Carrier Signal')
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(t, psk_signal, label='PSK Modulated Signal', color='r')
plt.title('PSK Modulated Signal')
plt.ylabel('Amplitude')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.step(np.arange(len(decoded_bits)), decoded_bits, label='Decoded Bits', color='r', marker='x')
plt.title('Decoded Bits')
plt.xlabel('Time')
plt.ylabel('Bit Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```
```
Program for QPSK

import numpy as np
pb = [] # Parity matrix
Ik = [] # I_K Matrix
p = []
m = []
h = []
h_dis = []
r_code = []
err = []
col = int(input("Enter the Parity bits : "))
row = int(input("Enter the Message bits : "))
# Generator matrix
for i in range (row):
    p = list(map(int, input(f"Enter the row values : {i+1} (Separated by space) : ").split()))  
    pb.append(p)
p_mat = np.array(pb, dtype=int)
Ik=np.eye(row, dtype=int) # Diagonal Matrix
g_mat = np.hstack((p_mat,Ik)) # Generator Matris

# Codeword length and parity bit length
n, k = g_mat.T.shape
# Possible Message Bits
m = np.array([[1 if (i >> (k - j - 1)) & 1 else 0 for j in range(k)] for i in range(2**k)])
# Codewords and Hamming weights
c = np.mod(np.dot(m, g_mat), 2)
for i, row in enumerate(c):
    h_dis1 = np.sum(row)  # Count number of 1's in the row
    h_dis.append(h_dis1)
h_mat = np.array(h_dis).reshape(1,-1)
#h_mat = np.hstack(h_mat)
d_min = np.min(np.sum(c[1:], axis=1))
# H matrix (Parity-check matrix)
h = p_mat[:, :3]
hp = np.hstack((np.eye(n-k, dtype=int), h.T))
ht = hp.T
print('')
print('The Generator Matrix is: ')
#for r in p_mat: 
#    print(" ".join(map(str, r)))
#for r in Ik: 
#    print(" ".join(map(str, r)))
for r in g_mat: 
    print(" ".join(map(str, r)))
print('')
print(f'Message Bits  Codeword   Hamming Weight')
code_word = np.hstack((m, c, h_mat.T))
for r in range(code_word.shape[0]):
    format_row = " ".join(map(str, code_word[r, :k])) + '\t' + " ".join(map(str, code_word[r, k:n+k])) + '\t' + str(code_word[r, -1])
    print(format_row)
print('')
print(f'Minimum Hamming distance : {d_min}')
# Parity Check matrix
print('')
print(f'Parity Check Matrix')
for r in hp:
    print(" ".join(map(str, r)))
print('')
print(f'Parity Check Matrix Transpose')
for r in ht:
    print(" ".join(map(str, r)))
#Receive codeword
rc = list(map(int, input(f"Enter the error codeword : ").split()))  
r_code.append(rc)
r_c = np.array(r_code)
#Syndrome Calculation
e = np.mod(np.dot(r_c, ht), 2)

#print('')
#print(f'Received codeword Matrix')
#for r in r_c:
#    print(" ".join(map(str, r)))
print('')
print(f"Syndeome of given received codeword is : " + " ".join(map(str, e[0])))
print('')
print(f'Syndrome Matrix')
for i in range(n):
    combined_row = np.concatenate((ht[i, :], np.eye(n, dtype=int)[i,:]))
    formatted_row = " ".join(map(str, combined_row[:3])) + '\t' + " ".join(map(str, combined_row[k:]))
    print(f'{formatted_row}')
# Find the Error position
for i in range(n):
    if np.array_equal(e[0], ht[i, :]):
        err = np.eye(n, dtype=int)[i,:]
print(f"The error postion is : " + " ".join(map(str, err)))
# Correct the error in the received codeword
add = err + rc
print(f"The correct codeword is : " + " " .join(map(str,add)))

```
# Output Waveform
```
PSK
<img width="1110" height="746" alt="Screenshot 2025-10-25 151425" src="https://github.com/user-attachments/assets/29eb2a62-d7b8-4d1e-8680-ca94f24a9397" />

```
```
QPSK

<img width="1584" height="803" alt="Screenshot 2025-10-25 153541" src="https://github.com/user-attachments/assets/ea7fbc16-80e1-42a1-a92d-cf8f3ef75fb8" />


```
# Results
```
Attach the output waveform
```
# Hardware experiment output waveform.
