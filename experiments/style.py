"""図の共通の約束（色・文字・目盛）。figs_e0_v2.py・figs_e1.py から使う"""
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm

for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NotoSansCJK-Bold" in f:
        fm.fontManager.addfont(f)
SURF = "#ffffff"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8c8b86"; GRID = "#e6e5e1"
sns.set_theme(style="white", rc={"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                                 "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                                 "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
                                 "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})

# ---------------------------------------------------------------- 色：系統で 4 色
BLUE, ORANGE, AQUA, GREY = "#186cce", "#e04e24", "#1f9f81", "#8f8f8c"
FAMILY = {"F": ("基準", GREY), "U": ("償わない", ORANGE), "V½": ("償わない", ORANGE), "V¾": ("償わない", ORANGE),
          "T0": ("共通の物差しで償う", AQUA), "T0+": ("共通の物差しで償う", AQUA),
          "T1": ("言い値で償う", BLUE), "T1h": ("言い値で償う", BLUE), "T1+": ("言い値で償う", BLUE)}
ORDER = ["F", "U", "V½", "V¾", "T0", "T0+", "T1", "T1h", "T1+"]
DESC = {"F": "固定（変えない）", "U": "単独・償わない", "V½": "過半の投票", "V¾": "3/4 の投票", "T0": "帳簿どおり・自分の手",
        "T0+": "帳簿どおり・他戸の手も", "T1": "言い値・自分の手", "T1h": "言い値＋余りの半分", "T1+": "言い値・他戸の手も"}
PATS = ["同質", "α=30", "α=3", "α=0.5", "二群", "少数の極端", "階の不一致"]
PAT_DESC = {"同質": "全員が平均の性向", "α=30": "ばらつき小", "α=3": "ばらつき中", "α=0.5": "ばらつき大", "二群": "広さ派 5・静けさ派 4",
            "少数の極端": "1 戸だけ近さ派", "階の不一致": "1F が日照派"}
DIV = LinearSegmentedColormap.from_list("div", ["#a32d2d", "#e34948", "#f0efec", "#186cce", "#24438f"])
SEQ_O = LinearSegmentedColormap.from_list("seqo", ["#fcede9", "#eea18a", "#e04e24", "#9b3619"])
SEQ_B = LinearSegmentedColormap.from_list("seqb", ["#f0f6fc", "#9bbfea", "#186cce", "#24438f"])


def hair(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def ink_on(rgb):
    r, g, b = rgb[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK

# ---------------------------------------------------------------- 第一段階からの記号：頭文字で中身を思い出せるように
DEEP = "#24438f"   # 改良版（言い値の系統の濃い青）
SYM = ["F", "U", "V50", "V75", "TR", "TR+", "TP", "TP+", "TP+貸清再"]
SYM_FAM = {"F": GREY, "U": ORANGE, "V50": ORANGE, "V75": ORANGE, "TR": AQUA, "TR+": AQUA, "TP": BLUE, "TP+": BLUE, "TP+貸清再": DEEP}
SYM_DESC = {"F": "固定", "U": "単独", "V50": "投票・過半", "V75": "投票・3/4", "TR": "物差しで償う・自戸の手",
            "TR+": "物差しで償う・他戸の手も", "TP": "言い値で償う・自戸の手", "TP+": "言い値で償う・他戸の手も",
            "TP+貸清再": "改良版（貸し借り・精算・再同意）"}
