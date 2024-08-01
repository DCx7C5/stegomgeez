import cv2
import numpy as np
from pywt import dwt2, idwt2

from stegomgeez.helper import bin_str2str, str2bin_str


def find_delimiter(message, delimiter):
    delimiter_idx = message.find(delimiter)
    if delimiter_idx != -1:
        message += message[:delimiter_idx]
    return message


def dct_encode(image_path, message, output_path, delimiter='#####'):
    """Encode data using Discrete Cosine Transform"""

    # Load image and convert to YCrCb color space
    image = cv2.imread(image_path)
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)

    # Convert message to binary
    message += delimiter
    binary_message = str2bin_str(message)

    # Perform DCT on the Y channel
    dct_y = cv2.dct(np.float32(y_channel))

    # Encode binary message into DCT coefficients
    idx = 0
    for i in range(dct_y.shape[0]):
        for j in range(dct_y.shape[1]):
            if idx < len(binary_message):
                dct_y[i, j] = int(dct_y[i, j]) & ~1 | int(binary_message[idx])
                idx += 1
            else:
                break

    # Perform inverse DCT
    encoded_y = cv2.idct(dct_y)
    encoded_ycrcb_image = cv2.merge((encoded_y, cr_channel, cb_channel))
    encoded_image = cv2.cvtColor(encoded_ycrcb_image, cv2.COLOR_YCrCb2BGR)

    # Save the encoded image
    cv2.imwrite(output_path, np.uint8(encoded_image))


def dct_decode(image_path, delimiter='#####'):
    """Decode data using Discrete Cosine Transform"""
    # Load image and convert to YCrCb color space
    image = cv2.imread(image_path)
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel, _, _ = cv2.split(ycrcb_image)

    # Perform DCT on the Y channel
    dct_y = cv2.dct(np.float32(y_channel))

    # Extract binary message from DCT coefficients
    binary_message = ''
    for i in range(dct_y.shape[0]):
        for j in range(dct_y.shape[1]):
            binary_message += str(int(dct_y[i, j]) & 1)

    # Convert binary message to string
    message = bin_str2str(binary_message)

    # Find the delimiter
    return find_delimiter(message, delimiter)


def dft_encode(image_path, message, output_path, delimiter='#####'):
    """Encode data using Discrete Fourier Transform"""
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert message to binary
    message += delimiter
    binary_message = str2bin_str(message)

    # Perform DFT
    dft_image = np.fft.fft2(image)
    dft_image_shifted = np.fft.fftshift(dft_image)

    # Encode binary message into DFT coefficients
    idx = 0
    for i in range(dft_image_shifted.shape[0]):
        for j in range(dft_image_shifted.shape[1]):
            if idx < len(binary_message):
                real_part = dft_image_shifted[i, j].real
                dft_image_shifted[i, j] = complex(real_part - real_part % 2 + int(binary_message[idx]),
                                                  dft_image_shifted[i, j].imag)
                idx += 1
            else:
                break

    # Perform inverse DFT
    dft_image_ishifted = np.fft.ifftshift(dft_image_shifted)
    encoded_image = np.fft.ifft2(dft_image_ishifted)
    encoded_image = np.uint8(np.abs(encoded_image))

    # Save the encoded image
    cv2.imwrite(output_path, encoded_image)


def dft_decode(image_path, delimiter='#####'):
    """Decode data using Discrete Fourier Transform"""
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dft_image = np.fft.fft2(image)
    dft_image_shifted = np.fft.fftshift(dft_image)

    # Extract binary message from DFT coefficients
    binary_message = ''
    for i in range(dft_image_shifted.shape[0]):
        for j in range(dft_image_shifted.shape[1]):
            binary_message += str(int(dft_image_shifted[i, j].real) % 2)

    # Convert binary message to string
    message = bin_str2str(binary_message)

    # Find the delimiter
    return find_delimiter(message, delimiter)


def wavelet_encode(image_path, message, output_path, delimiter='#####'):
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert message to binary
    message += delimiter  # Delimiter
    binary_message = str2bin_str(message)

    # Perform wavelet transform
    coeffs = dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Encode binary message into wavelet coefficients
    idx = 0
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            if idx < len(binary_message):
                LL[i, j] = int(LL[i, j]) & ~1 | int(binary_message[idx])
                idx += 1
            else:
                break

    # Perform inverse wavelet transform
    coeffs = LL, (LH, HL, HH)
    encoded_image = idwt2(coeffs, 'haar')
    encoded_image = np.uint8(encoded_image)

    # Save the encoded image
    cv2.imwrite(output_path, encoded_image)


def wavelet_decode(image_path, delimiter='#####'):
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    coeffs = dwt2(image, 'haar')
    LL, (_, _, _) = coeffs

    # Extract binary message from wavelet coefficients
    binary_message = ''
    for i in range(LL.shape[0]):
        for j in range(LL.shape[1]):
            binary_message += str(int(LL[i, j]) & 1)

    # Convert binary message to string
    message = bin_str2str(binary_message)

    # Find the delimiter
    return find_delimiter(message, delimiter)


def detect_fq_steganography(image_path):
    """Detect hidden data in the frequency domain using statistical analysis"""
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dft_image = np.fft.fft2(image)
    dft_image_shifted = np.fft.fftshift(dft_image)

    # Analyze the least significant bits of the DFT coefficients
    lsb_values = []
    for i in range(dft_image_shifted.shape[0]):
        for j in range(dft_image_shifted.shape[1]):
            lsb_values.append(int(dft_image_shifted[i, j].real) % 2)

    # Perform statistical analysis on LSB values
    lsb_mean = np.mean(lsb_values)
    lsb_variance = np.var(lsb_values)

    print("LSB Mean:", lsb_mean)
    print("LSB Variance:", lsb_variance)

    # Heuristic: If LSB mean is significantly different from 0.5, suspect steganography
    if abs(lsb_mean - 0.5) > 0.1:
        print("Potential hidden data detected")
    else:
        print("No hidden data detected")
