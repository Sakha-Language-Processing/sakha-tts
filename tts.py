import torch
import soundfile

from sys import argv, path
path.append("vits")
import commons
import utils
from models import SynthesizerTrn


def get_symbols_to_id(vocab_file):
    with open(vocab_file, encoding="utf-8") as f:
        data = (x.replace("\n", "") for x in f.readlines())
        return {s: i for i, s in enumerate(data)}


def filter_oov(text):
    return "".join(list(filter(lambda x: x in symbols_to_id, text)))


def text_to_sequence(text):
    sequence = []
    clean_text = text.strip()
    for symbol in clean_text:
        symbol_id = symbols_to_id[symbol]
        sequence += [symbol_id]
    return sequence


def get_text(text, hps):
    text = filter_oov(text.lower())
    text_norm = text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


LANG = "sah"
ckpt_dir = f"./{LANG}"
vocab_file = f"{ckpt_dir}/vocab.txt"
config_file = f"{ckpt_dir}/config.json"
hps = utils.get_hparams_from_file(config_file)
symbols_to_id = get_symbols_to_id(vocab_file)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net_g = SynthesizerTrn(
    len(symbols_to_id),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
)
net_g.to(device)
_ = net_g.eval()
g_pth = f"{ckpt_dir}/G_100000.pth"
_ = utils.load_checkpoint(g_pth, net_g, None)

stn_tst = get_text(argv[1], hps)
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    hyp = net_g.infer(
        x_tst, x_tst_lengths, noise_scale=.667,
        noise_scale_w=0.8, length_scale=1.0,
    )[0][0,0].cpu().float().numpy()

soundfile.write(argv[2], hyp, 16_000)
