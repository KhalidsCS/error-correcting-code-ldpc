import numpy as np
import math

SNR_dB= 10
SNR = 10**(SNR_dB/10)
RATE=4/7
DATA_SIZE=7
SIGMA_SQ= 1/(2*RATE*SNR)
SIGMA=np.sqrt(SIGMA_SQ)
rng=np.random.default_rng()
NOISE=rng.normal(loc=0.0, scale=SIGMA, size=DATA_SIZE)
NOISE_IDX=0
PARITY_CHECK_MATRIX = np.array([[1, 1, 0, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0, 0, 1]])
TANNER_BIT_TO_CHECK= np.array([[idx for idx in range(PARITY_CHECK_MATRIX.shape[1]) if PARITY_CHECK_MATRIX[arr,idx]==1] for arr in range(PARITY_CHECK_MATRIX.shape[0])])

def calculate_check_to_bit(noisy_message):
    tanner_check_to_bit=np.zeros(noisy_message.shape)
    for i in range(TANNER_BIT_TO_CHECK.shape[0]):
        sorted_abs_values_idx= sorted([x for x in TANNER_BIT_TO_CHECK[i]], key= lambda x: abs(noisy_message[x]))
        sign= -1 if math.prod([noisy_message[x] for x in sorted_abs_values_idx])<0 else 1
        for j in sorted_abs_values_idx:
            current_sign= -1 if noisy_message[j]<0 else 1
            if j>0:
                tanner_check_to_bit[j]+= abs(TANNER_BIT_TO_CHECK[i, 0])*sign*current_sign*.75
            elif noisy_message.shape[0]>1:
                tanner_check_to_bit[j]+= abs(TANNER_BIT_TO_CHECK[i, 1])*sign*current_sign*.75
    return tanner_check_to_bit

def llr(noisy_message):
    global TANNER_BIT_TO_CHECK, NOISE_IDX
    tanner_check_to_bit=calculate_check_to_bit(noisy_message)
    noisy_message+=tanner_check_to_bit
    NOISE_IDX+=7

def generate_gaussian_noise(message):
    global NOISE, NOISE_IDX
    signed_message=np.where(message==1,-1,1)
    noisy_message= signed_message+NOISE[NOISE_IDX:7]
    NOISE_IDX+=7
    return noisy_message

def iterative_ldpc(noisy_message): 
    while True:
        output_message=np.where(noisy_message<0,1,0)
        llr(noisy_message)
        if np.all(np.where(noisy_message<0,1,0)==output_message):
            break
    return noisy_message

def main():
    message=np.array([1, 0, 1, 0, 1, 0, 1])
    noisy_message=generate_gaussian_noise(message)
    output_message=iterative_ldpc(noisy_message)
    print(output_message)
    if np.all(np.where(output_message<0,1,0)==message):
        print("Message decoded successfully")
    else:
        print("Message decoded incorrectly")
        

if __name__ == "__main__":
    main()
