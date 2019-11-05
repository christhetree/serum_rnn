from typing import Tuple, Any

import cProfile
import librenderman as rm
import matplotlib.pyplot as plt


SAMPLE_RATE = 44100
BUFFER_SIZE = 512
FFT_SIZE = 512
SERUM_PATH = "/Library/Audio/Plug-Ins/VST/Serum.vst"


def setup_serum(path: str) -> Tuple[Any, Any]:
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


def main() -> None:
    num_trials = 10
    engine, generator = setup_serum(SERUM_PATH)

    for _ in range(num_trials):
        new_patch = get_random_patch(generator)
        set_patch(engine, new_patch)

        midi_note = 40
        midi_velocity = 127
        note_length = 2.0
        render_length = 2.0

        render_patch(engine, midi_note, midi_velocity, note_length, render_length)
        audio = engine.get_audio_frames()

        # plt.plot(audio)
        # plt.xlabel('Time (frame count)')
        # plt.show()


if __name__ == '__main__':
    cProfile.run('main()')
