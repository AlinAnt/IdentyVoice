import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment


def preparation(before_path, data):
    
    result_folder = Path('data/splitted')

    for row in tqdm(data.iterrows()):
        client_id = row[1][0]
        path = row[1][1]

        audio_data = before_path / path
        segment = AudioSegment.from_mp3(audio_data)
        segment = segment.set_frame_rate(16000)

        audio = np.array(segment.get_array_of_samples())
        audio_size = 16000 * 3
        audio.resize(audio_size,  refcheck=False)

        result = result_folder / client_id
        if not result.exists():
            os.mkdir(result)

        result = result / (path.split('.')[0] + '.wav')

        audio_segment = AudioSegment(
            audio.tobytes(), 
            frame_rate=16000,
            sample_width=audio.dtype.itemsize, 
            channels=1
        )
        audio_segment.export(result, format='wav')