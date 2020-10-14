print('i like pie')

# sine_dry, sr_1 = sf.read(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine__dry.wav')
# sine_dist, sr_2 = sf.read(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine__distortion__Dist_Drv_100__Dist_Mode_05.wav')
# square_dry, sr_3 = sf.read(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square__dry.wav')
# square_dist, sr_4 = sf.read(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square__distortion__Dist_Drv_100__Dist_Mode_05.wav')
# sr = 44100
# assert sr_1 == sr_2 == sr_3 == sr_4 == sr
#
# n_fft = 2048
# hop_length = 512
# n_mels = 128
#
# sine_dry_mel = lr.feature.melspectrogram(
#     sine_dry,
#     sr=sr,
#     n_fft=n_fft,
#     hop_length=hop_length,
#     n_mels=n_mels
# )
# sine_dist_mel = lr.feature.melspectrogram(
#     sine_dist,
#     sr=sr,
#     n_fft=n_fft,
#     hop_length=hop_length,
#     n_mels=n_mels
# )
# square_dry_mel = lr.feature.melspectrogram(
#     square_dry,
#     sr=sr,
#     n_fft=n_fft,
#     hop_length=hop_length,
#     n_mels=n_mels
# )
# square_dist_mel = lr.feature.melspectrogram(
#     square_dist,
#     sr=sr,
#     n_fft=n_fft,
#     hop_length=hop_length,
#     n_mels=n_mels
# )
# sine_dist_m_dry = sine_dist_mel - sine_dry_mel
# sine_dry_m_dist = sine_dry_mel - sine_dist_mel
# square_dist_m_dry = square_dist_mel - square_dry_mel
# square_dry_m_dist = square_dry_mel - square_dist_mel
#
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(sine_dist_m_dry, sr=sr, n_fft=n_fft,
#                                         hop_length=hop_length)
# sf.write(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine_dist_m_dry.wav',
#     audio, sr)
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(sine_dry_m_dist, sr=sr, n_fft=n_fft,
#                                         hop_length=hop_length)
# sf.write(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine_dry_m_dist.wav',
#     audio, sr)
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(square_dist_m_dry, sr=sr, n_fft=n_fft,
#                                         hop_length=hop_length)
# sf.write(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square_dist_m_dry.wav',
#     audio, sr)
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(square_dry_m_dist, sr=sr, n_fft=n_fft,
#                                         hop_length=hop_length)
# sf.write(
#     '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square_dry_m_dist.wav',
#     audio, sr)
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(sine_dry_mel, sr=sr, n_fft=n_fft, hop_length=hop_length)
# sf.write('/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine__dry_inv_256.wav', audio, sr)
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(sine_dist_mel, sr=sr, n_fft=n_fft, hop_length=hop_length)
# sf.write('/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine__dist_inv_256.wav', audio, sr)
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(square_dry_mel, sr=sr, n_fft=n_fft, hop_length=hop_length)
# sf.write('/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square__dry_inv_256.wav', audio, sr)
# print('inv')
# audio = lr.feature.inverse.mel_to_audio(square_dist_mel, sr=sr, n_fft=n_fft, hop_length=hop_length)
# sf.write('/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square__dist_inv_256.wav', audio, sr)


# i = 1000000
# results = []
# for _ in range(i):
#     goal = 0.0
#     # goal = np.random.uniform(0.0, 1.0)
#     guess = np.random.uniform(0.0, 1.0)
#     mae = abs(goal - guess)
#     results.append(mae)
#
# print(len(results))
# print(np.mean(results))
# print(np.std(results))
# exit()
