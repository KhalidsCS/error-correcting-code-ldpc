import numpy as np
import math

# Prototype LDPC decoder for the (7, 4) Hamming code on an AWGN channel.
# A 7-bit word is sent with BPSK (0 -> +1, 1 -> -1), Gaussian noise is added,
# then check nodes push updates back onto the soft values until the hard
# decision stops changing.

# Eb/N0 in dB, and the same ratio on a linear scale.
SNR_dB= 10
SNR = 10**(SNR_dB/10)
# Four information bits and three parity bits: rate k/n = 4/7.
RATE=4/7
DATA_SIZE=7
# Noise variance for unit-energy BPSK: sigma^2 = N0/2 = 1 / (2 * R * Eb/N0).
SIGMA_SQ= 1/(2*RATE*SNR)
SIGMA=np.sqrt(SIGMA_SQ)
rng=np.random.default_rng()
# One pre-drawn noise block. NOISE_IDX is the cursor into that block.
NOISE=rng.normal(loc=0.0, scale=SIGMA, size=DATA_SIZE)
NOISE_IDX=0
# Parity-check matrix H (3 checks x 7 bits). A 1 means that bit is in that check.
# The last three columns are the identity, so those bits are the parity bits.
PARITY_CHECK_MATRIX = np.array([[1, 1, 0, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0, 1, 0],
                        [0, 1, 1, 1, 0, 0, 1]])
# Tanner graph, check side: row i lists the bit indices connected to check i.
TANNER_BIT_TO_CHECK= np.array([[idx for idx in range(PARITY_CHECK_MATRIX.shape[1]) if PARITY_CHECK_MATRIX[arr,idx]==1] for arr in range(PARITY_CHECK_MATRIX.shape[0])])

def calculate_check_to_bit(noisy_message):
    # One check-to-bit update. Each check looks at the soft values of its bits
    # and returns an additive correction of the same length as the received word.
    tanner_check_to_bit=np.zeros(noisy_message.shape)
    for i in range(TANNER_BIT_TO_CHECK.shape[0]):
        # Least reliable bits (smallest magnitude) come first.
        sorted_abs_values_idx= sorted([x for x in TANNER_BIT_TO_CHECK[i]], key= lambda x: abs(noisy_message[x]))
        # Sign of the whole check: -1 when an odd number of its bits are negative.
        sign= -1 if math.prod([noisy_message[x] for x in sorted_abs_values_idx])<0 else 1
        for j in range(len(sorted_abs_values_idx)):
            current_sign= -1 if noisy_message[j]<0 else 1
            # sign * current_sign drops this bit's own sign, leaving the product
            # of the other bits in the check. 0.75 scales that contribution.
            # Bit 0 uses the second neighbor index as the magnitude; every other
            # bit uses the first neighbor index.
            if j>0:
                tanner_check_to_bit[sorted_abs_values_idx[j]]+= abs(noisy_message[sorted_abs_values_idx[0]])*sign*current_sign*.75
            elif len(sorted_abs_values_idx)>1:
                tanner_check_to_bit[sorted_abs_values_idx[j]]+= abs(noisy_message[sorted_abs_values_idx[1]])*sign*current_sign*.75
    return tanner_check_to_bit

def llr(noisy_message):
    # Fold the check-to-bit correction into the soft values in place.
    global TANNER_BIT_TO_CHECK, NOISE_IDX, DATA_SIZE
    tanner_check_to_bit=calculate_check_to_bit(noisy_message)
    noisy_message+=tanner_check_to_bit
    NOISE_IDX+=DATA_SIZE

def generate_gaussian_noise(message):
    # Modulate the bits and add the next DATA_SIZE samples from the noise block.
    global NOISE, NOISE_IDX, DATA_SIZE
    signed_message=np.where(message==1,-1,1)
    noisy_message= signed_message+NOISE[NOISE_IDX:NOISE_IDX+DATA_SIZE]
    NOISE_IDX+=DATA_SIZE
    return noisy_message

def iterative_ldpc(noisy_message): 
    max_iterations=100
    iterations=0
    # Keep updating until a pass no longer changes the hard decision
    # (negative -> 1, otherwise -> 0). Returns the soft values, not the bits.
    while iterations<max_iterations:
        llr(noisy_message)
        if np.all((PARITY_CHECK_MATRIX @ np.where(noisy_message<0,1,0) )%2==0):
            break
        iterations+=1
    return noisy_message

def main():
    # Send one example word, decode it, and compare the hard decision to the original.
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
