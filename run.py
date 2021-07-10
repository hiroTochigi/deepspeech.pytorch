import json
import os

import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
    audio_path_list = get_audio_list()
    for audio_path in audio_path_list:
        if audio_path.find('010-14-000.wav') > -1:
            output_path = "/analyze/transcript/" + audio_path.split('/')[-1].split('.')[0] + '.txt'
            print(output_path)
            cfg.audio_path=audio_path
            cfg.model.model_path="deepspeech.pytorch/librispeech_pretrained_v3.ckpt"
            result = transcribe(cfg=cfg)
            transcript = result["output"][0]["transcription"]
            with open(output_path, "w") as f:
                f.write(transcript.lower())

def get_audio_list():
    for root, dirs, audio_path_list in os.walk("/analyze/audio/"):
        if root == "/analyze/audio/":
            return [ root + audio_path for audio_path in audio_path_list ]

if __name__ == '__main__':
    #iterate_audio()
    hydra_main()
