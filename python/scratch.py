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


# x_id_to_y = {}
# log.info('Creating x_id_to_y')
# for x_id, mel_path, base_mel_path in tqdm(self.x_ids):
#     y = super()._create_y_batch([(x_id, mel_path, base_mel_path)])
#     y = [v[0] for v in y]
#     self.n_y_cols = len(y)
#     x_id_to_y[x_id] = y
#
# self.x_id_to_y = x_id_to_y

# def _create_y_batch(
#         self, batch_x_ids: List[Tuple[str, str, str]]) -> List[np.ndarray]:
#     log.info('Using fast _create_y_batch')
#     y = [[] for _ in range(self.n_y_cols)]
#
#     for x_id, _, _ in batch_x_ids:
#         row_y = self.x_id_to_y[x_id]
#         for all_v, row_v in zip(y, row_y):
#             all_v.append(row_v)
#
#     return y

# def __init__(self,
#              x_ids: List[Tuple[str, str, str]],
#              x_y_metadata: XYMetaData,
#              batch_size: int = 128,
#              shuffle: bool = True,
#              channel_mode: int = 1) -> None:
#     super().__init__(x_ids,
#                      x_y_metadata,
#                      batch_size=batch_size,
#                      shuffle=shuffle,
#                      channel_mode=channel_mode)


# if y_id in self.y_id_to_y_data:
#     y_data = self.y_id_to_y_data[y_id]
# else:
#     with np.load(os.path.join(self.y_dir, y_id)) as y_data:
#         y_data_copy = copy.deepcopy(dict(y_data))
#     self.y_id_to_y_data[y_id] = y_data_copy
#
#
# def speed_test():
#     import time
#     import numpy as np
#     from tqdm import tqdm
#
#     n = 10000
#     batch_size = 128
#     in_x = 128
#     in_y = 88
#     mel = np.ones((in_x, in_y))
#
#     start_time = time.time()
#
#     mels = np.empty((batch_size, in_x, in_y), dtype=np.float32)
#
#     for _ in tqdm(range(n)):
#         for idx in range(batch_size):
#             mels[idx, :, :] = mel
#
#         # mels = []
#         # for idx in range(batch_size):
#         #     mels.append(mel)
#         #
#         # mels = np.array(mels, dtype=np.float32)
#
#     end_time = time.time()
#     print(f'time elapsed = {end_time - start_time}')
