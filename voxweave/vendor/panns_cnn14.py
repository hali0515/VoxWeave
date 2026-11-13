"""Self-contained PANNs Cnn14 inference network and AudioSet 527-class table.

Adapted from qiuqiangkong/audioset_tagging_cnn (MIT License, Copyright (c) 2019
Qiuqiang Kong) and torchlibrosa (MIT License, Copyright (c) 2019 Qiuqiang Kong).
Replaces the unmaintained ``panns-inference`` wrapper so VoxWeave never shells
out to ``wget``, never writes ``~/panns_data/``, never imports ``matplotlib``,
and loads checkpoints with ``weights_only=True``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AUDIOSET_LABELS: tuple[str, ...] = ('Speech',
 'Male speech, man speaking',
 'Female speech, woman speaking',
 'Child speech, kid speaking',
 'Conversation',
 'Narration, monologue',
 'Babbling',
 'Speech synthesizer',
 'Shout',
 'Bellow',
 'Whoop',
 'Yell',
 'Battle cry',
 'Children shouting',
 'Screaming',
 'Whispering',
 'Laughter',
 'Baby laughter',
 'Giggle',
 'Snicker',
 'Belly laugh',
 'Chuckle, chortle',
 'Crying, sobbing',
 'Baby cry, infant cry',
 'Whimper',
 'Wail, moan',
 'Sigh',
 'Singing',
 'Choir',
 'Yodeling',
 'Chant',
 'Mantra',
 'Male singing',
 'Female singing',
 'Child singing',
 'Synthetic singing',
 'Rapping',
 'Humming',
 'Groan',
 'Grunt',
 'Whistling',
 'Breathing',
 'Wheeze',
 'Snoring',
 'Gasp',
 'Pant',
 'Snort',
 'Cough',
 'Throat clearing',
 'Sneeze',
 'Sniff',
 'Run',
 'Shuffle',
 'Walk, footsteps',
 'Chewing, mastication',
 'Biting',
 'Gargling',
 'Stomach rumble',
 'Burping, eructation',
 'Hiccup',
 'Fart',
 'Hands',
 'Finger snapping',
 'Clapping',
 'Heart sounds, heartbeat',
 'Heart murmur',
 'Cheering',
 'Applause',
 'Chatter',
 'Crowd',
 'Hubbub, speech noise, speech babble',
 'Children playing',
 'Animal',
 'Domestic animals, pets',
 'Dog',
 'Bark',
 'Yip',
 'Howl',
 'Bow-wow',
 'Growling',
 'Whimper (dog)',
 'Cat',
 'Purr',
 'Meow',
 'Hiss',
 'Caterwaul',
 'Livestock, farm animals, working animals',
 'Horse',
 'Clip-clop',
 'Neigh, whinny',
 'Cattle, bovinae',
 'Moo',
 'Cowbell',
 'Pig',
 'Oink',
 'Goat',
 'Bleat',
 'Sheep',
 'Fowl',
 'Chicken, rooster',
 'Cluck',
 'Crowing, cock-a-doodle-doo',
 'Turkey',
 'Gobble',
 'Duck',
 'Quack',
 'Goose',
 'Honk',
 'Wild animals',
 'Roaring cats (lions, tigers)',
 'Roar',
 'Bird',
 'Bird vocalization, bird call, bird song',
 'Chirp, tweet',
 'Squawk',
 'Pigeon, dove',
 'Coo',
 'Crow',
 'Caw',
 'Owl',
 'Hoot',
 'Bird flight, flapping wings',
 'Canidae, dogs, wolves',
 'Rodents, rats, mice',
 'Mouse',
 'Patter',
 'Insect',
 'Cricket',
 'Mosquito',
 'Fly, housefly',
 'Buzz',
 'Bee, wasp, etc.',
 'Frog',
 'Croak',
 'Snake',
 'Rattle',
 'Whale vocalization',
 'Music',
 'Musical instrument',
 'Plucked string instrument',
 'Guitar',
 'Electric guitar',
 'Bass guitar',
 'Acoustic guitar',
 'Steel guitar, slide guitar',
 'Tapping (guitar technique)',
 'Strum',
 'Banjo',
 'Sitar',
 'Mandolin',
 'Zither',
 'Ukulele',
 'Keyboard (musical)',
 'Piano',
 'Electric piano',
 'Organ',
 'Electronic organ',
 'Hammond organ',
 'Synthesizer',
 'Sampler',
 'Harpsichord',
 'Percussion',
 'Drum kit',
 'Drum machine',
 'Drum',
 'Snare drum',
 'Rimshot',
 'Drum roll',
 'Bass drum',
 'Timpani',
 'Tabla',
 'Cymbal',
 'Hi-hat',
 'Wood block',
 'Tambourine',
 'Rattle (instrument)',
 'Maraca',
 'Gong',
 'Tubular bells',
 'Mallet percussion',
 'Marimba, xylophone',
 'Glockenspiel',
 'Vibraphone',
 'Steelpan',
 'Orchestra',
 'Brass instrument',
 'French horn',
 'Trumpet',
 'Trombone',
 'Bowed string instrument',
 'String section',
 'Violin, fiddle',
 'Pizzicato',
 'Cello',
 'Double bass',
 'Wind instrument, woodwind instrument',
 'Flute',
 'Saxophone',
 'Clarinet',
 'Harp',
 'Bell',
 'Church bell',
 'Jingle bell',
 'Bicycle bell',
 'Tuning fork',
 'Chime',
 'Wind chime',
 'Change ringing (campanology)',
 'Harmonica',
 'Accordion',
 'Bagpipes',
 'Didgeridoo',
 'Shofar',
 'Theremin',
 'Singing bowl',
 'Scratching (performance technique)',
 'Pop music',
 'Hip hop music',
 'Beatboxing',
 'Rock music',
 'Heavy metal',
 'Punk rock',
 'Grunge',
 'Progressive rock',
 'Rock and roll',
 'Psychedelic rock',
 'Rhythm and blues',
 'Soul music',
 'Reggae',
 'Country',
 'Swing music',
 'Bluegrass',
 'Funk',
 'Folk music',
 'Middle Eastern music',
 'Jazz',
 'Disco',
 'Classical music',
 'Opera',
 'Electronic music',
 'House music',
 'Techno',
 'Dubstep',
 'Drum and bass',
 'Electronica',
 'Electronic dance music',
 'Ambient music',
 'Trance music',
 'Music of Latin America',
 'Salsa music',
 'Flamenco',
 'Blues',
 'Music for children',
 'New-age music',
 'Vocal music',
 'A capella',
 'Music of Africa',
 'Afrobeat',
 'Christian music',
 'Gospel music',
 'Music of Asia',
 'Carnatic music',
 'Music of Bollywood',
 'Ska',
 'Traditional music',
 'Independent music',
 'Song',
 'Background music',
 'Theme music',
 'Jingle (music)',
 'Soundtrack music',
 'Lullaby',
 'Video game music',
 'Christmas music',
 'Dance music',
 'Wedding music',
 'Happy music',
 'Funny music',
 'Sad music',
 'Tender music',
 'Exciting music',
 'Angry music',
 'Scary music',
 'Wind',
 'Rustling leaves',
 'Wind noise (microphone)',
 'Thunderstorm',
 'Thunder',
 'Water',
 'Rain',
 'Raindrop',
 'Rain on surface',
 'Stream',
 'Waterfall',
 'Ocean',
 'Waves, surf',
 'Steam',
 'Gurgling',
 'Fire',
 'Crackle',
 'Vehicle',
 'Boat, Water vehicle',
 'Sailboat, sailing ship',
 'Rowboat, canoe, kayak',
 'Motorboat, speedboat',
 'Ship',
 'Motor vehicle (road)',
 'Car',
 'Vehicle horn, car horn, honking',
 'Toot',
 'Car alarm',
 'Power windows, electric windows',
 'Skidding',
 'Tire squeal',
 'Car passing by',
 'Race car, auto racing',
 'Truck',
 'Air brake',
 'Air horn, truck horn',
 'Reversing beeps',
 'Ice cream truck, ice cream van',
 'Bus',
 'Emergency vehicle',
 'Police car (siren)',
 'Ambulance (siren)',
 'Fire engine, fire truck (siren)',
 'Motorcycle',
 'Traffic noise, roadway noise',
 'Rail transport',
 'Train',
 'Train whistle',
 'Train horn',
 'Railroad car, train wagon',
 'Train wheels squealing',
 'Subway, metro, underground',
 'Aircraft',
 'Aircraft engine',
 'Jet engine',
 'Propeller, airscrew',
 'Helicopter',
 'Fixed-wing aircraft, airplane',
 'Bicycle',
 'Skateboard',
 'Engine',
 'Light engine (high frequency)',
 "Dental drill, dentist's drill",
 'Lawn mower',
 'Chainsaw',
 'Medium engine (mid frequency)',
 'Heavy engine (low frequency)',
 'Engine knocking',
 'Engine starting',
 'Idling',
 'Accelerating, revving, vroom',
 'Door',
 'Doorbell',
 'Ding-dong',
 'Sliding door',
 'Slam',
 'Knock',
 'Tap',
 'Squeak',
 'Cupboard open or close',
 'Drawer open or close',
 'Dishes, pots, and pans',
 'Cutlery, silverware',
 'Chopping (food)',
 'Frying (food)',
 'Microwave oven',
 'Blender',
 'Water tap, faucet',
 'Sink (filling or washing)',
 'Bathtub (filling or washing)',
 'Hair dryer',
 'Toilet flush',
 'Toothbrush',
 'Electric toothbrush',
 'Vacuum cleaner',
 'Zipper (clothing)',
 'Keys jangling',
 'Coin (dropping)',
 'Scissors',
 'Electric shaver, electric razor',
 'Shuffling cards',
 'Typing',
 'Typewriter',
 'Computer keyboard',
 'Writing',
 'Alarm',
 'Telephone',
 'Telephone bell ringing',
 'Ringtone',
 'Telephone dialing, DTMF',
 'Dial tone',
 'Busy signal',
 'Alarm clock',
 'Siren',
 'Civil defense siren',
 'Buzzer',
 'Smoke detector, smoke alarm',
 'Fire alarm',
 'Foghorn',
 'Whistle',
 'Steam whistle',
 'Mechanisms',
 'Ratchet, pawl',
 'Clock',
 'Tick',
 'Tick-tock',
 'Gears',
 'Pulleys',
 'Sewing machine',
 'Mechanical fan',
 'Air conditioning',
 'Cash register',
 'Printer',
 'Camera',
 'Single-lens reflex camera',
 'Tools',
 'Hammer',
 'Jackhammer',
 'Sawing',
 'Filing (rasp)',
 'Sanding',
 'Power tool',
 'Drill',
 'Explosion',
 'Gunshot, gunfire',
 'Machine gun',
 'Fusillade',
 'Artillery fire',
 'Cap gun',
 'Fireworks',
 'Firecracker',
 'Burst, pop',
 'Eruption',
 'Boom',
 'Wood',
 'Chop',
 'Splinter',
 'Crack',
 'Glass',
 'Chink, clink',
 'Shatter',
 'Liquid',
 'Splash, splatter',
 'Slosh',
 'Squish',
 'Drip',
 'Pour',
 'Trickle, dribble',
 'Gush',
 'Fill (with liquid)',
 'Spray',
 'Pump (liquid)',
 'Stir',
 'Boiling',
 'Sonar',
 'Arrow',
 'Whoosh, swoosh, swish',
 'Thump, thud',
 'Thunk',
 'Electronic tuner',
 'Effects unit',
 'Chorus effect',
 'Basketball bounce',
 'Bang',
 'Slap, smack',
 'Whack, thwack',
 'Smash, crash',
 'Breaking',
 'Bouncing',
 'Whip',
 'Flap',
 'Scratch',
 'Scrape',
 'Rub',
 'Roll',
 'Crushing',
 'Crumpling, crinkling',
 'Tearing',
 'Beep, bleep',
 'Ping',
 'Ding',
 'Clang',
 'Squeal',
 'Creak',
 'Rustle',
 'Whir',
 'Clatter',
 'Sizzle',
 'Clicking',
 'Clickety-clack',
 'Rumble',
 'Plop',
 'Jingle, tinkle',
 'Hum',
 'Zing',
 'Boing',
 'Crunch',
 'Silence',
 'Sine wave',
 'Harmonic',
 'Chirp tone',
 'Sound effect',
 'Pulse',
 'Inside, small room',
 'Inside, large room or hall',
 'Inside, public space',
 'Outside, urban or manmade',
 'Outside, rural or natural',
 'Reverberation',
 'Echo',
 'Noise',
 'Environmental noise',
 'Static',
 'Mains hum',
 'Distortion',
 'Sidetone',
 'Cacophony',
 'White noise',
 'Pink noise',
 'Throbbing',
 'Vibration',
 'Television',
 'Radio',
 'Field recording')


class _STFT(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 320, center: bool = True) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        out_channels = n_fft // 2 + 1
        self.conv_real = nn.Conv1d(
            1, out_channels, kernel_size=n_fft, stride=hop_length, bias=False
        )
        self.conv_imag = nn.Conv1d(
            1, out_channels, kernel_size=n_fft, stride=hop_length, bias=False
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x[:, None, :]
        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode="reflect")
        real = self.conv_real(x)[:, None, :, :].transpose(2, 3)
        imag = self.conv_imag(x)[:, None, :, :].transpose(2, 3)
        return real, imag


class _Spectrogram(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 320) -> None:
        super().__init__()
        self.stft = _STFT(n_fft=n_fft, hop_length=hop_length, center=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(x)
        return real.pow(2) + imag.pow(2)


class _LogmelFilterBank(nn.Module):
    def __init__(
        self, n_fft: int = 1024, n_mels: int = 64, ref: float = 1.0, amin: float = 1e-10
    ) -> None:
        super().__init__()
        self.ref = ref
        self.amin = amin
        self.melW = nn.Parameter(
            torch.zeros(n_fft // 2 + 1, n_mels), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = torch.matmul(x, self.melW)
        log_spec = 10.0 * torch.log10(torch.clamp(mel, min=self.amin))
        log_spec = log_spec - 10.0 * float(np.log10(max(self.amin, self.ref)))
        return log_spec


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        classes_num: int = 527,
    ) -> None:
        super().__init__()
        self.spectrogram_extractor = _Spectrogram(n_fft=window_size, hop_length=hop_size)
        self.logmel_extractor = _LogmelFilterBank(n_fft=window_size, n_mels=mel_bins)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(2, 2))
        x = self.conv_block5(x, pool_size=(2, 2))
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        embedding = F.relu_(self.fc1(x))
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))
        return clipwise_output, embedding


class PannsTagger:
    """Drop-in replacement for ``panns_inference.AudioTagging`` without side effects."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = device
        self.labels = list(AUDIOSET_LABELS)
        self.classes_num = len(AUDIOSET_LABELS)
        model = Cnn14(classes_num=self.classes_num)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        if device != "cpu":
            model = model.to(device)
        self.model = model

    def inference(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.as_tensor(audio, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            probs, emb = self.model(tensor)
        return probs.detach().cpu().numpy(), emb.detach().cpu().numpy()
