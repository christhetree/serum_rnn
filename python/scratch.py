print('i like pie')

# if __name__ == '__main__':
#     # crawl_presets(os.path.join(PRESETS_DIR, 'default_serum'), 'default_serum.json')
#     # exit()
#     # create_distortion_data(os.path.join(PRESETS_DIR, 'default_serum'))
#     # exit()
#
#     # data = np.load('../data/datasets/default_distortion_renders_10k.npz')
#     # renders = data['renders']
#     # params = data['params']
#     # mels = []
#     # print(renders.shape)
#     # print(params.shape)
#     #
#     # for render in tqdm(renders):
#     #     mel = get_mel_spec(renders[0],
#     #                        sr=44100,
#     #                        max_len_samples=66560,
#     #                        normalize_audio=True,
#     #                        normalize_mel=True)
#     #     mels.append(mel)
#     #
#     # mels = np.array(mels, dtype=np.float32)
#     # print(mels.shape)
#     # np.savez('../data/datasets/mels_10k_testing.npz', mels=mels, params=params)
#
#     # print(testing.shape)
#     # np.savez('../data/datasets/default_distortion_renders_100.npz', renders=renders[:100, :], params=params[:100, :])
#     # exit()
#     data = np.load('../data/datasets/mels_10k_testing.npz')
#     mels = data['mels']
#     params = data['params']
#     print(mels.shape)
#     print(mels.dtype)
#     print(params.shape)
#     print(params.dtype)
#     x = np.expand_dims(mels, axis=-1)
#     y = params[:, :3].astype(np.float32) / 100
#     print(x.shape)
#     print(x.dtype)
#     print(y.shape)
#     print(y.dtype)
#
#     cnn = build_cnn3_classifier()
#     cnn.summary()
#     cnn.compile(optimizer='adam', loss='mse')
#
#     cnn.fit(x, y, batch_size=64, epochs=10, validation_split=0.2, shuffle=True)
#
#     exit()
#
#     fxpA_path = os.path.join(
#         PRESETS_DIR, 'default.fxp')
#     # fxpA_path = os.path.join(
#     #     PRESETS_DIR, 'default__distortion_tube_drive50.fxp')
#     # fxpB_path = os.path.join(
#     #     PRESETS_DIR, 'default__distortion_tube_drive100.fxp')
#     fxpB_path = os.path.join(
#         PRESETS_DIR, 'default__distortion_soft-clip_drive50.fxp')
#     # fxpB_path = os.path.join(
#     #     PRESETS_DIR, 'default__distortion_lin-fold_drive50.fxp')
#     find_fxp_differences(fxpA_path, fxpB_path)
#     exit()
#
#     presets = load_presets('../data/presets/default_serum.json')
#     preset_names = list(presets.keys())
#
#     # engine, generator = setup_serum(os.path.join(PRESETS_DIR, 'default.fxp'))
#     # engine = setup_serum()
#     midi_note = 60
#     midi_velocity = 127
#     note_length = 1.5
#     render_length = 1.5
#
#     presets_root_dir = os.path.join(PRESETS_DIR, 'bass_subset')
#     file_ending = '.fxp'
#
#     preset_paths = []
#     for root, dirs, files in tqdm(os.walk(presets_root_dir)):
#         for f in files:
#             if f.endswith(file_ending):
#                 preset_paths.append(os.path.join(root, f))
#
#     log.info(f'Found {len(preset_paths)} files ending with {file_ending}')
#     # engine = setup_serum()
#     # engine = setup_serum(os.path.join(PRESETS_DIR, 'default.fxp'))
#
#     for path in preset_paths:
#         preset_name = ntpath.basename(path)
#         # preset = presets[preset_name]
#         # rm_preset = [(int(k), v) for k, v in preset.items()]
#         # patch = engine.get_patch()
#         # set_preset(engine, preset)
#         # engine.set_patch(rm_preset)
#         engine = setup_serum(path)
#         # engine.load_preset(path)
#         # load_preset(engine, path)
#         engine.render_patch(midi_note, midi_velocity, note_length,
#                             render_length, False)
#         # engine.write_to_wav(f'../out/{preset_name}_load_patch_rm.wav')
#         audio = np.array(engine.get_audio_frames())
#         print(audio.shape)
#         lr.output.write_wav(f'../out/{preset_name}_sr44100lr_lr.wav', audio,
#                             sr=44100, norm=False)
#
#     # for idx in tqdm(range(10)):
#     #     # random_preset_path = random.choice(preset_paths)
#     #     # print(random_preset_path)
#     #     # engine.load_preset(random_preset_path)
#     #     # random_patch = generator.get_random_patch()
#     #     # for i, v in random_patch:
#     #     #     engine.set_parameter(i, v)
#     #
#     #     # engine.set_patch(random_patch)
#     #     random_preset_name = random.choice(preset_names)
#     #     random_preset = presets[random_preset_name]
#     #     print(random_preset_name)
#     #     set_preset(engine, random_preset)
#     #     engine.render_patch(midi_note, midi_velocity, note_length,
#     #                         render_length, False)
#     #     audio = np.array(engine.get_audio_frames())
#     #     # print(audio.shape)
#     #     lr.output.write_wav(f'../out/test{idx}_lr.wav', audio, sr=RM_SR, norm=False)
#     #     # engine.write_to_wav(f'../out/test{idx}.wav')
#     #     # print(audio[:10])
#     #     # print(random_patch)
#     #     # engine.set_patch()
#     #     # print(engine.get_parameter(0))
#     # # engine.set_parameter(97, 0.25)
#     # # engine.set_parameter(99, 0.32)
#     # # engine.set_parameter(154, 1.0)
#     # # save_result = engine.save_preset(os.path.join(OUT_DIR, 'testing.fxp'))
#     # # print(save_result)

if np.random.random_sample() > 0.5:
    curr_v = np.random.normal(0.75,
                              0.125,
                              (self.n_params,)).astype(np.float32)
    self.curr_v = np.clip(curr_v, 0.0, 1.0)
else:
    curr_v = np.random.normal(0.25,
                              0.125,
                              (self.n_params,)).astype(np.float32)
    self.curr_v = np.clip(curr_v, 0.0, 1.0)



gen_config = {
    'root_dir': '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/audio_render_test',
    'n': 10000,
    'preset': 'default',
    'note_length': 1.0,
    'render_length': 1.0,
    'midi': [40],
    'vel': [127],
    'effects': [{
        'name': 'distortion',
        'Dist_Drv': [],
        'Dist_Mode': [],
    }, {
        'name': 'flanger',
        'Flg_Rate': [],
        'Flg_Dep': [],
        'Flg_Feed': [],
        'Flg_Stereo': [],
    }]
}

# data_path = os.path.join(DATASETS_DIR, 'renders_10k_testing_midi.npz')
# data = np.load(data_path)
# renders = data['renders']
# params = data['params']
# mels = []
# print(renders.shape)
# print(params.shape)
# exit()
#
# for render in tqdm(renders):
#     mel = get_mel_spec(render,
#                        sr=RM_SR,
#                        max_n_of_frames=44544,
#                        normalize_audio=True,
#                        normalize_mel=True)
#     mels.append(mel)
#
# mels = np.array(mels, dtype=np.float32)
# print(mels.shape)
# np.savez('../data/datasets/mels_10k_testing_midi.npz', mels=mels, params=params)


sine_dry, sr_1 = sf.read(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine__dry.wav')
sine_dist, sr_2 = sf.read(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine__distortion__Dist_Drv_100__Dist_Mode_05.wav')
square_dry, sr_3 = sf.read(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square__dry.wav')
square_dist, sr_4 = sf.read(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square__distortion__Dist_Drv_100__Dist_Mode_05.wav')
sr = 44100
assert sr_1 == sr_2 == sr_3 == sr_4 == sr

n_fft = 2048
hop_length = 512
n_mels = 128

sine_dry_mel = lr.feature.melspectrogram(
    sine_dry,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)
sine_dist_mel = lr.feature.melspectrogram(
    sine_dist,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)
square_dry_mel = lr.feature.melspectrogram(
    square_dry,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)
square_dist_mel = lr.feature.melspectrogram(
    square_dist,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)
sine_dist_m_dry = sine_dist_mel - sine_dry_mel
sine_dry_m_dist = sine_dry_mel - sine_dist_mel
square_dist_m_dry = square_dist_mel - square_dry_mel
square_dry_m_dist = square_dry_mel - square_dist_mel

print('inv')
audio = lr.feature.inverse.mel_to_audio(sine_dist_m_dry, sr=sr, n_fft=n_fft,
                                        hop_length=hop_length)
sf.write(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine_dist_m_dry.wav',
    audio, sr)
print('inv')
audio = lr.feature.inverse.mel_to_audio(sine_dry_m_dist, sr=sr, n_fft=n_fft,
                                        hop_length=hop_length)
sf.write(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/sine_dry_m_dist.wav',
    audio, sr)
print('inv')
audio = lr.feature.inverse.mel_to_audio(square_dist_m_dry, sr=sr, n_fft=n_fft,
                                        hop_length=hop_length)
sf.write(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square_dist_m_dry.wav',
    audio, sr)
print('inv')
audio = lr.feature.inverse.mel_to_audio(square_dry_m_dist, sr=sr, n_fft=n_fft,
                                        hop_length=hop_length)
sf.write(
    '/Users/christhetree/local_christhetree/audio_research/reverse_synthesis/data/square_dry_m_dist.wav',
    audio, sr)
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

# Squre compression
# -14.407989
# 17.579754
# -69.88389
# 39.220177

# Sine compression
# -45.556614
# 14.507765
# -70.14759
# 38.846474

# Square EQ
# -10.365809
# 16.198515
# -68.68618
# 48.045612

# Sine EQ
# -40.396572
# 15.43005
# -68.73795
# 48.020523

# # data = np.load('/Volumes/samsung_t5/reverse_synthesis/training_eq_l/square__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048/compressor__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_F__n_4986.npz',
# # data = np.load('/Volumes/samsung_t5/reverse_synthesis/training_eq_l/sine__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048/compressor__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_F__n_4981.npz',
# # data = np.load('/Volumes/samsung_t5/reverse_synthesis/training_eq_l/square__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048/eq__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_F__n_4994.npz',
# # data = np.load('/Volumes/samsung_t5/reverse_synthesis/training_eq_l/sine__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048/eq__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_F__n_4995.npz',
# data = np.load(
#     '/Volumes/samsung_t5/reverse_synthesis/training_eq_l/square__sr_44100__nl_1.00__rl_1.00__vel_127__midi_048/eq__gran_100/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_4994.npz',
#     allow_pickle=True)
# mels = data['mels']
# print(np.mean(mels))
# # print(np.mean(mels, axis=1))
# # print(np.mean(mels, axis=2))
# print(np.std(mels))
# print(np.min(mels))
# print(np.max(mels))
# exit()


# # datasets_dir = DATASETS_DIR
# datasets_dir = '/mnt/ssd01/christhetree/reverse_synthesis/data/datasets'
# x_data_path = os.path.join(datasets_dir, f'training_eq_l__{effect}.npz')
# x_npz_data = np.load(x_data_path)
#
# # x_base_path = f'audio_render_test/default__sr_44100__nl_1.00__rl_1.00__vel_127__midi_040/{effect}__gran_{gran}/processing/mel__sr_44100__frames_44544__n_fft_4096__n_mels_128__hop_len_512__norm_audio_F__norm_mel_T__n_{n}___base__{"_".join(base_effects)}.npz'
# # x_base_path = os.path.join(DATA_DIR, x_base_pathd)
# # x_base_data = np.load(x_base_path)
# # base_mels = x_base_data['mels']
# # log.info(f'base_effects = {base_effects}')
# # log.info(f'base_mels shape = {base_mels.shape}')
#
# x = x_npz_data['mels'][:N]
# in_x = x.shape[1]
# in_y = x.shape[2]
# log.info(f'mels shape = {x.shape}')
# log.info(f'in_x = {in_x}, in_y = {in_y}')
#
# base_mels = x_npz_data['base_mels'][:N]
# log.info(f'base_mels shape = {base_mels.shape}')
#
# x = np.concatenate([x, base_mels], axis=-1)
# # x = np.concatenate([base_mels, x], axis=-1)
# # x = np.concatenate([x, x], axis=-1)
# # log.info('Using same x, x')
# # x = np.concatenate([base_mels, base_mels], axis=-1)
# # log.info('Using same b, b')
# # x = x - base_mels
# log.info(f'x shape = {x.shape}')
#
# y_data_path = f'{os.path.splitext(x_data_path)[0]}__y_{params}.npz'
# y_data = prepare_y_model_data(y_data_path)
# y = [out[:N] for out in y_data.y_s]
