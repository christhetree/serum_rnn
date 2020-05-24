import librenderman as rm

SAMPLE_RATE = 44100
BUFFER_SIZE = 512
FFT_SIZE = 512
SERUM_PATH = "/Library/Audio/Plug-Ins/VST/Serum.vst"


def setup_serum(path):
    engine = rm.RenderEngine(SAMPLE_RATE, BUFFER_SIZE, FFT_SIZE)
    engine.load_plugin(path)
    generator = rm.PatchGenerator(engine)
    return engine, generator


engine, generator = setup_serum(SERUM_PATH)
print(engine.get_plugin_parameters_description())
engine.load_preset('/Users/christhetree/local_christhetree/audio_research/lib/RenderMan/Builds/MacOSX/build/Debug/angle_grinder.fxp')
engine.set_parameter(0, 0.1)

midi_note = 40
midi_velocity = 127
note_length = 3.0
render_length = 4.0


engine.render_patch(midi_note, midi_velocity, note_length, render_length, False)
audio = engine.get_audio_frames()

save_result = engine.save_preset('/Users/christhetree/local_christhetree/audio_research/lib/RenderMan/Builds/MacOSX/build/Debug/derp6.fxp')
print(save_result)
print('After saving')
