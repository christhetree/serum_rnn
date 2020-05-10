import cProfile

import librenderman as rm
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
BUFFER_SIZE = 512
FFT_SIZE = 512
SERUM_PATH = "/Library/Audio/Plug-Ins/VST/Serum.vst"
# SERUM_PATH = "/Library/Audio/Plug-Ins/VST/SerumFX.vst"
# SERUM_PATH = "/Users/christhetree/local_christhetree/research/titech/reverse_synthesis/python/angle_grinder.fxp"


def setup_serum(path: str) -> (rm.RenderEngine, rm.PatchGenerator):
    engine = rm.RenderEngine(SAMPLE_RATE, BUFFER_SIZE, FFT_SIZE)
    engine.load_plugin(path)
    generator = rm.PatchGenerator(engine)
    return engine, generator


def get_random_patch(generator):
    return generator.get_random_patch()


def set_patch(engine, patch):
    engine.set_patch(patch)


def render_patch(engine, midi_note, midi_velocity, note_length, render_length) -> None:
    engine.render_patch(midi_note, midi_velocity, note_length, render_length)


def plot_audio(audio) -> None:
    plt.plot(audio)
    plt.xlabel('Time (frame count)')
    plt.show()


def main() -> None:
    num_trials = 0
    engine, generator = setup_serum(SERUM_PATH)

    print(engine.get_plugin_parameters_description())
    exit()

    for _ in range(num_trials):
        new_patch = get_random_patch(generator)
        # set_patch(engine, new_patch)

        midi_note = 40
        midi_velocity = 127
        note_length = 3.0
        render_length = 4.0

        render_patch(engine, midi_note, midi_velocity, note_length, render_length)
        audio = engine.get_audio_frames()

        plot_audio(audio)


if __name__ == '__main__':
    main()
    # cProfile.run('main()')
