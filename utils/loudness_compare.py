import soundfile as sf
import pyloudnorm as pyln

full_condition_audio, rate = sf.read("full_condition.wav") # load audio (with shape (samples, channels))
sori_only_audio, rate = sf.read("sori_only.wav") # load audio (with shape (samples, channels))
meter = pyln.Meter(rate) # create BS.1770 meter
full_condition_loudness = meter.integrated_loudness(full_condition_audio) # measure loudness
sori_only_loudness = meter.integrated_loudness(sori_only_audio) # measure loudness

print(full_condition_loudness)
print(sori_only_loudness)
print(full_condition_loudness - sori_only_loudness)







