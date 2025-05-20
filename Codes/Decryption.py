import math
import numpy as np
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import sys

def char_to_binary(c):
    return format(ord(c),'08b')

def binary_to_char(n):
    return int(n, 2)

def calculate_entropy(data_list):
    length = len(data_list)
    # Count the frequency of each unique element
    freq = Counter(data_list)
    # Compute entropy
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())

    return entropy

mio=3.999999674
x=0.997674
seedForDNADecode=0.9996548
def getPseduRandomRule():
   global x
   x=(mio*x*(1-x))
   return int(x*8+1)

def getPseduRandomRuleForDecode():
   global seedForDNADecode
   seedForDNADecode=(mio*seedForDNADecode*(1-seedForDNADecode))
   return int(seedForDNADecode*8+1)

def getPseduRandomPos(n):#returns a number between 0 and n-1
   global x
   x=(mio*x*(1-x))
   return int(x*n)

def getPseduRandomTwoThirdBitFilp(temp):
    global x
    x = mio * x * (1 - x)
    length = len(temp)

    for i in range(int(length * 2 / 3)):
        pos = getPseduRandomPos(length)
        if pos >= length:
            print(f"Invalid pos: {pos}, length: {length}")
            sys.exit()
        if temp[pos] == '0':
            temp[pos] = '1'
        else:
            temp[pos] = '0'

def key_scramble(key, target_length):
    reverse_temp = list(key[::-1])
    getPseduRandomTwoThirdBitFilp(reverse_temp)
    large_key = reverse_temp

    while len(large_key) < target_length:
        reverse_temp = list(large_key[::-1])
        getPseduRandomTwoThirdBitFilp(reverse_temp)
        large_key+=reverse_temp
    return large_key[:target_length]

def DNADecode(DNA):
    Bits = ''
    j = 0
    for i in range(4):
        rule=getPseduRandomRule()
        if rule == 1:
            if DNA[i] == 'A':
                Bits += '00'
            elif DNA[i] == 'G':
                Bits += '01'
            elif DNA[i] == 'C':
                Bits += '10'
            elif DNA[i] == 'T':
                Bits += '11'
            j += 2
        elif rule == 2:
            if DNA[i] == 'A':
                Bits += '00'
            elif DNA[i] == 'C':
                Bits += '01'
            elif DNA[i] == 'G':
                Bits += '10'
            elif DNA[i] == 'T':
                Bits += '11'
            j += 2
        elif rule == 3:
            if DNA[i] == 'T':
                Bits += '00'
            elif DNA[i] == 'G':
                Bits += '01'
            elif DNA[i] == 'C':
                Bits += '10'
            elif DNA[i] == 'A':
                Bits += '11'
            j += 2
        elif rule == 4:
            if DNA[i] == 'T':
                Bits += '00'
            elif DNA[i] == 'C':
                Bits += '01'
            elif DNA[i] == 'G':
                Bits += '10'
            elif DNA[i] == 'A':
                Bits += '11'
            j += 2
        elif rule == 5:
            if DNA[i] == 'C':
                Bits += '00'
            elif DNA[i] == 'T':
                Bits += '01'
            elif DNA[i] == 'A':
                Bits += '10'
            elif DNA[i] == 'G':
                Bits += '11'
            j += 2
        elif rule == 6:
            if DNA[i] == 'C':
                Bits += '00'
            elif DNA[i] == 'A':
                Bits += '01'
            elif DNA[i] == 'T':
                Bits += '10'
            elif DNA[i] == 'G':
                Bits += '11'
            j += 2
        elif rule == 7:
            if DNA[i] == 'G':
                Bits += '00'
            elif DNA[i] == 'T':
                Bits += '01'
            elif DNA[i] == 'A':
                Bits += '10'
            elif DNA[i] == 'C':
                Bits += '11'
            j += 2
        else:
            if DNA[i] == 'G':
                Bits += '00'
            elif DNA[i] == 'A':
                Bits += '01'
            elif DNA[i] == 'T':
                Bits += '10'
            elif DNA[i] == 'C':
                Bits += '11'
            j += 2
    return Bits

def DNAEncode2(bits):
    DNA = ''
    j = 0
    for i in range(0, 8, 2):
        rule=getPseduRandomRuleForDecode()
        if rule == 1:
            if bits[i:i+2] == '00':
                DNA += 'A'
            elif bits[i:i+2] == '01':
                DNA += 'G'
            elif bits[i:i+2] == '10':
                DNA += 'C'
            elif bits[i:i+2] == '11':
                DNA += 'T'
            j += 1
        elif rule == 2:
            if bits[i:i+2] == '00':
                DNA += 'A'
            elif bits[i:i+2] == '01':
                DNA += 'C'
            elif bits[i:i+2] == '10':
                DNA += 'G'
            elif bits[i:i+2] == '11':
                DNA += 'T'
            j += 1
        elif rule == 3:
            if bits[i:i+2] == '00':
                DNA += 'T'
            elif bits[i:i+2] == '01':
                DNA += 'G'
            elif bits[i:i+2] == '10':
                DNA += 'C'
            elif bits[i:i+2] == '11':
                DNA += 'A'
            j += 1
        elif rule == 4:
            if bits[i:i+2] == '00':
                DNA += 'T'
            elif bits[i:i+2] == '01':
                DNA += 'C'
            elif bits[i:i+2] == '10':
                DNA += 'G'
            elif bits[i:i+2] == '11':
                DNA += 'A'
            j += 1
        elif rule == 5:
            if bits[i:i+2] == '00':
                DNA += 'C'
            elif bits[i:i+2] == '01':
                DNA += 'T'
            elif bits[i:i+2] == '10':
                DNA += 'A'
            elif bits[i:i+2] == '11':
                DNA += 'G'
            j += 1
        elif rule == 6:
            if bits[i:i+2] == '00':
                DNA += 'C'
            elif bits[i:i+2] == '01':
                DNA += 'A'
            elif bits[i:i+2] == '10':
                DNA += 'T'
            elif bits[i:i+2] == '11':
                DNA += 'G'
            j += 1
        elif rule == 7:
            if bits[i:i+2] == '00':
                DNA += 'G'
            elif bits[i:i+2] == '01':
                DNA += 'T'
            elif bits[i:i+2] == '10':
                DNA += 'A'
            elif bits[i:i+2] == '11':
                DNA += 'C'
            j += 1
        elif rule == 8:
            if bits[i:i+2] == '00':
                DNA += 'G'
            elif bits[i:i+2] == '01':
                DNA += 'A'
            elif bits[i:i+2] == '10':
                DNA += 'T'
            elif bits[i:i+2] == '11':
                DNA += 'C'
            j += 1
    return DNA

def DNAEncode(bits):
    DNA = ''
    j = 0
    for i in range(0, 8, 2):
        rule=getPseduRandomRule()
        if rule == 1:
            if bits[i:i+2] == '00':
                DNA += 'A'
            elif bits[i:i+2] == '01':
                DNA += 'G'
            elif bits[i:i+2] == '10':
                DNA += 'C'
            elif bits[i:i+2] == '11':
                DNA += 'T'
            j += 1
        elif rule == 2:
            if bits[i:i+2] == '00':
                DNA += 'A'
            elif bits[i:i+2] == '01':
                DNA += 'C'
            elif bits[i:i+2] == '10':
                DNA += 'G'
            elif bits[i:i+2] == '11':
                DNA += 'T'
            j += 1
        elif rule == 3:
            if bits[i:i+2] == '00':
                DNA += 'T'
            elif bits[i:i+2] == '01':
                DNA += 'G'
            elif bits[i:i+2] == '10':
                DNA += 'C'
            elif bits[i:i+2] == '11':
                DNA += 'A'
            j += 1
        elif rule == 4:
            if bits[i:i+2] == '00':
                DNA += 'T'
            elif bits[i:i+2] == '01':
                DNA += 'C'
            elif bits[i:i+2] == '10':
                DNA += 'G'
            elif bits[i:i+2] == '11':
                DNA += 'A'
            j += 1
        elif rule == 5:
            if bits[i:i+2] == '00':
                DNA += 'C'
            elif bits[i:i+2] == '01':
                DNA += 'T'
            elif bits[i:i+2] == '10':
                DNA += 'A'
            elif bits[i:i+2] == '11':
                DNA += 'G'
            j += 1
        elif rule == 6:
            if bits[i:i+2] == '00':
                DNA += 'C'
            elif bits[i:i+2] == '01':
                DNA += 'A'
            elif bits[i:i+2] == '10':
                DNA += 'T'
            elif bits[i:i+2] == '11':
                DNA += 'G'
            j += 1
        elif rule == 7:
            if bits[i:i+2] == '00':
                DNA += 'G'
            elif bits[i:i+2] == '01':
                DNA += 'T'
            elif bits[i:i+2] == '10':
                DNA += 'A'
            elif bits[i:i+2] == '11':
                DNA += 'C'
            j += 1
        elif rule == 8:
            if bits[i:i+2] == '00':
                DNA += 'G'
            elif bits[i:i+2] == '01':
                DNA += 'A'
            elif bits[i:i+2] == '10':
                DNA += 'T'
            elif bits[i:i+2] == '11':
                DNA += 'C'
            j += 1
    return DNA

def DNAXOR(DNA1, DNA2):
    DNA = ''
    for i in range(4):
        if ((DNA1[i] == 'A' and DNA2[i] == 'A') or
            (DNA1[i] == 'T' and DNA2[i] == 'T') or
            (DNA1[i] == 'C' and DNA2[i] == 'C') or
            (DNA1[i] == 'G' and DNA2[i] == 'G')):
            DNA += 'A'
        elif ((DNA1[i] == 'A' and DNA2[i] == 'T') or
              (DNA1[i] == 'T' and DNA2[i] == 'A')):
            DNA += 'T'
        elif ((DNA1[i] == 'A' and DNA2[i] == 'C') or
              (DNA1[i] == 'C' and DNA2[i] == 'A')):
            DNA += 'C'
        elif ((DNA1[i] == 'A' and DNA2[i] == 'G') or
              (DNA1[i] == 'G' and DNA2[i] == 'A')):
            DNA += 'G'
        elif ((DNA1[i] == 'T' and DNA2[i] == 'C') or
              (DNA1[i] == 'C' and DNA2[i] == 'T')):
            DNA += 'G'
        elif ((DNA1[i] == 'T' and DNA2[i] == 'G') or
              (DNA1[i] == 'G' and DNA2[i] == 'T')):
            DNA += 'C'
        elif ((DNA1[i] == 'C' and DNA2[i] == 'G') or
              (DNA1[i] == 'G' and DNA2[i] == 'C')):
            DNA += 'T'
    return DNA



def encryption(ScrambleKeyBits, image_path):
    """Applies binary segments to the image's RGB channels and prints pixel values."""
    image = Image.open(image_path)
    image_array = np.array(image)
    print(image_array[275][275][0])

    height, width, channels = image_array.shape
    processed_image = np.zeros(shape=(height,width,channels), dtype=np.uint8)
    processed_image = np.copy(image)

    index = 0
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                if index < len(ScrambleKeyBits):
                    key_segment = ScrambleKeyBits[index*8:(index+1)*8]
                    key_segment = ''.join(key_segment)
                    original_value = processed_image[i, j, k]
                    original_bin = format(original_value, '08b')

                    key_dna=DNAEncode(key_segment)
                    ori_dna=DNAEncode2(original_bin)
                    xored_dna=DNAXOR(key_dna,ori_dna)
                    decoded_bin=DNADecode(xored_dna)
                    cipher_value = int(decoded_bin, 2)

                    processed_image[i, j, k] = cipher_value
                    index += 1
                    #print(index)
    #processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    print(processed_image[275][275][0])

    return processed_image

def main():

    image_path = '/home/encrypted_image.png'
    image = Image.open(image_path)
    image_array = np.array(image)

    image_row, image_col, channel = image_array.shape
    target_length = image_row * image_col * channel * 8

    txt = "Bangladesh"
    binary_string = ''.join(char_to_binary(c) for c in txt)
    #binary_string="101"
    #print(len(binary_string))
    #proposed Algorithm

    scrambled_key = key_scramble(binary_string, target_length)
    cipherImage= encryption(scrambled_key, image_path)
    plt.imshow(cipherImage)
    Image.fromarray(cipherImage).save('/home/decrypt_image.jpeg')



    # Compute histogram using NumPy
    #hist, bins = np.histogram(cipherImage.ravel(), bins=256, range=[0,256])

    # Plot the histogram
    #plt.plot(hist, color='black')
    #plt.title("Histogram (NumPy Method)")
    #plt.xlabel("Pixel Intensity")
    #plt.ylabel("Frequency")


    # img = [1, 2, 2]

    # txt = "Bangladesh"
    # binary_string = ''.join(char_to_binary(c) for c in txt)
    # #binary_string="101"
    # #print(len(binary_string))
    # #proposed Algorithm
    # scrambled_key = key_scramble(binary_string, 25165824)
    # # Current Algorithm
    # inscrambled_key = inkey_scramble(binary_string, 25165824)
    # #print(inscrambled_key)
    # print(len(inscrambled_key),len(scrambled_key))

    # # conversion to 8 bit ascii int value
    # # proposed
    # conv = [binary_to_char(scrambled_key[i:i + 8]) for i in range(0, len(scrambled_key), 8)]
    # #new
    # conv1 = [binary_to_char(inscrambled_key[i:i + 8]) for i in range(0, len(inscrambled_key), 8)]

    # # list size (should be key size/8)
    # print(str(len(conv1))+' '+str(len(conv)))
    # #first 100 letters print to check consistancy
    # # character = ''.join(map(chr, conv))
    # # for i in range(min(100, len(character))):
    # #      print(character[i])

    # print("Entropy of original Algorithm:", calculate_entropy(conv1))
    # print("Entropy of proposed Algorithm:", calculate_entropy(conv))

main()
