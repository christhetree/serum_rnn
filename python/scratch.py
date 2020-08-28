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
