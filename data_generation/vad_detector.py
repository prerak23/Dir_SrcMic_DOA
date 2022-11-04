import numpy as np
import h5py
import webrtcvad

vad = webrtcvad.Vad(2)


def float_to_pcm16(audio):
    import numpy

    ints = (audio * 32767).astype(numpy.int16)
    little_endian = ints.astype("<u2")
    buf = little_endian.tobytes()
    return buf


sample_rate = 16000
frame_duration = 30  # in ms
number_of_sample = 20 * 16

path_file = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/ICASSP/voicehome2_arr/D1_0000/noisy_mixtures/D1_0000_aggregated_mixture.hdf5"

mix_file = h5py.File(path_file)


hit_ = 0
total_sample = 0
for i in mix_file["room_nos"].keys():
    signal = mix_file["room_nos"][i]["nsmix_f"][0, 0, :]
    total_sample += 1
    print(i)
    for j in np.arange(100):
        x = signal[(j * number_of_sample) : (number_of_sample * (j + 1))]
        buf = float_to_pcm16(x)
        if vad.is_speech(buf, sample_rate):
            hit_ += 1
            break


print("Percentage of signal that have speech", (hit_ / total_sample))
